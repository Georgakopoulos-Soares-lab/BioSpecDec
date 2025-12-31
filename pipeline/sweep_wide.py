from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from typing import Any, Dict, List, Optional

from .io_utils import append_jsonl, ensure_parent_dir
from .model_cache import ModelCache
from .runners import GenerationRequest, load_models, run_generation


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    if not keys:
        return [{}]
    vals = [grid[k] for k in keys]
    out: List[Dict[str, Any]] = []
    import itertools

    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return out


def _matches_condition(run_cfg: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    for k, expected in cond.items():
        if run_cfg.get(k) != expected:
            return False
    return True


def _satisfies_then(run_cfg: Dict[str, Any], then: Dict[str, List[Any]]) -> bool:
    for k, allowed in then.items():
        if allowed is None:
            continue
        if not isinstance(allowed, list):
            raise ValueError(f"constraint.then.{k} must be a list")
        if run_cfg.get(k) not in allowed:
            return False
    return True


def _passes_constraints(run_cfg: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    for c in constraints:
        if_cond = c.get("if") or {}
        then_cond = c.get("then") or {}
        if not isinstance(if_cond, dict) or not isinstance(then_cond, dict):
            raise ValueError("Each constraint must be { 'if': {...}, 'then': {...} }")
        if _matches_condition(run_cfg, if_cond):
            if not _satisfies_then(run_cfg, then_cond):
                return False
    return True


def _stable_group_id(run_cfg: Dict[str, Any]) -> str:
    excluded = {"method"}

    canonical = dict(run_cfg)
    if canonical.get("model_family") == "progen2" and canonical.get("progen2_draft_mode", "pretrained") == "pretrained":
        canonical.pop("draft_layers", None)
        canonical.pop("draft_layer_indices", None)

    payload = {k: canonical.get(k) for k in sorted(canonical.keys()) if k not in excluded}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _decode_ids(cm: Any, model_family: str, ids: List[int]) -> str:
    if model_family == "protgpt2":
        return cm.tokenizer.decode(ids, skip_special_tokens=False)
    if model_family == "progen2":
        return cm.tokenizer.decode(ids)
    raise ValueError(model_family)


def _suffix_from_record(cm: Any, model_family: str, rec: Dict[str, Any]) -> Optional[str]:
    try:
        prompt_text = _decode_ids(cm, model_family, list(rec["prompt_ids"]))
        full_text = str(rec.get("generated_text") or "")
        if full_text.startswith(prompt_text):
            return full_text[len(prompt_text) :]
        # Fallback: still return something rather than None
        return full_text
    except Exception:
        return None


def _totals_from_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total_tokens = float(sum(float(r.get("num_new_tokens") or 0.0) for r in records))
    total_time = float(sum(float(r.get("wall_time_s") or 0.0) for r in records))
    tps = (total_tokens / total_time) if total_time > 0 else 0.0
    return {"tps": tps, "tokens_total": total_tokens, "time_total": total_time}


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Protein wide sweep runner: for each prompt+hyperparams, runs target/draft/specdec and writes one wide CSV row. "
            "Optionally writes per-sample JSONL records."
        )
    )

    p.add_argument("--config", required=True, help="Path to a protein sweep JSON config")
    p.add_argument(
        "--output_csv",
        required=True,
        help="Wide-format CSV output (target/draft/specdec metrics + example suffixes).",
    )
    p.add_argument(
        "--output_jsonl",
        default="",
        help="Optional per-sample JSONL output (leave empty to disable).",
    )
    p.add_argument("--num_shards", type=int, default=1, help="Split runs into N shards (for multi-GPU).")
    p.add_argument("--shard_idx", type=int, default=0, help="Which shard to run: 0..N-1")
    p.add_argument(
        "--device_map",
        default="",
        help=(
            "Optional HF device_map override for ProGen2 (e.g. 'auto'). "
            "When set, allows sharding the model across all visible GPUs in a single process."
        ),
    )
    p.add_argument(
        "--target_device",
        default="",
        help="Optional device for the target model (e.g. 'cuda:0'). Overrides config.base.device for target only.",
    )
    p.add_argument(
        "--draft_device",
        default="",
        help="Optional device for the draft model (e.g. 'cuda:1'). Overrides config.base.device for draft only.",
    )

    args = p.parse_args(argv)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise ValueError("--shard_idx must be in [0, num_shards)")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base = cfg.get("base", {})
    grid = cfg.get("grid", {})
    constraints = cfg.get("constraints", [])

    if not isinstance(base, dict) or not isinstance(grid, dict):
        raise ValueError("config must include dicts: base, grid")
    if constraints and not isinstance(constraints, list):
        raise ValueError("config.constraints must be a list")

    # Wide runner always runs all three methods; drop method from sweep if present.
    base2 = dict(base)
    base2.pop("method", None)
    grid2 = dict(grid)
    grid2.pop("method", None)

    combos = _expand_grid(grid2)

    prompt_list = None
    if isinstance(grid.get("prompt_text"), list):
        prompt_list = list(grid.get("prompt_text"))

    cache = ModelCache()

    # Keep the DNAGPT-style schema for easy cross-model analysis.
    wide_columns = [
        "model_family",
        "target_model_name",
        "draft_model_name",
        "draft_mode",
        "draft_model_effective",
        "draft_num_layers_effective",
        "draft_layer_indices",
        "prompt_idx",
        "hg_row_index",
        "hg_id",
        "chrom",
        "start",
        "end",
        "prompt_len_bases",
        "prompt_len_tokens",
        "prefix_len_tokens",
        "target_context_len",
        "draft_context_len",
        "L",
        "accept_mode",
        "temperature",
        "top_k",
        "top_p",
        "target_tps",
        "draft_tps",
        "specdec_tps",
        "speedup_vs_target",
        "mean_accept_rate",
        "mean_accepted_prefix",
        "target_tokens_total",
        "draft_tokens_total",
        "specdec_tokens_total",
        "target_time_total",
        "draft_time_total",
        "specdec_time_total",
        "sample_target_suffix",
        "sample_draft_suffix",
        "sample_specdec_suffix",
        "sample_prompt_ids",
        "sample_target_new_ids",
        "sample_draft_new_ids",
        "sample_specdec_new_ids",
    ]

    ensure_parent_dir(args.output_csv)
    runs_written = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=wide_columns)
        writer.writeheader()
        out_f.flush()

        for run_idx, combo in enumerate(combos):
            if (run_idx % args.num_shards) != args.shard_idx:
                continue

            run_cfg = _merge(base2, combo)
            if constraints and (not _passes_constraints(run_cfg, constraints)):
                continue

            if runs_written == 0 or (run_idx % 10 == 0):
                print(
                    json.dumps(
                        {
                            "status": "starting_run",
                            "run_idx": run_idx,
                            "shard_idx": args.shard_idx,
                            "num_shards": args.num_shards,
                            "prompt_preview": str(run_cfg.get("prompt_text", ""))[:80],
                            "gamma": run_cfg.get("gamma"),
                            "draft_layers": run_cfg.get("draft_layers"),
                            "accept_mode": run_cfg.get("accept_mode"),
                        }
                    )
                )

            if "model_family" not in run_cfg:
                raise ValueError("run_cfg missing model_family")
            if "prompt_text" not in run_cfg:
                raise ValueError("run_cfg missing prompt_text")

            model_family = str(run_cfg["model_family"])
            prompt_text = str(run_cfg["prompt_text"])

            prompt_idx = 0
            if prompt_list is not None:
                try:
                    prompt_idx = int(prompt_list.index(prompt_text))
                except ValueError:
                    prompt_idx = 0

            sweep_group_id = _stable_group_id(run_cfg)

            # Shared request fields
            req_base = GenerationRequest(
                model_family=model_family,  # type: ignore[arg-type]
                method="specdec",  # placeholder; overwritten below
                prompt_text=prompt_text,
                max_new_tokens=int(run_cfg.get("max_new_tokens", run_cfg.get("num_tokens", 256))),
                num_samples=int(run_cfg.get("num_samples", 5)),
                temperature=float(run_cfg.get("temperature", 1.0)),
                top_k=int(run_cfg.get("top_k", 0)),
                top_p=float(run_cfg.get("top_p", 0.0)),
                gamma=int(run_cfg.get("gamma", 5)),
                accept_mode=str(run_cfg.get("accept_mode", "prob")),
                target_model_name=str(run_cfg.get("target_model_name", "")),
                draft_model_name=str(run_cfg.get("draft_model_name", "")),
                max_prompt_tokens=int(run_cfg.get("max_prompt_tokens", 0)),
                target_context_len=run_cfg.get("target_context_len"),
                draft_context_len=run_cfg.get("draft_context_len"),
                draft_layers=int(run_cfg.get("draft_layers", 8)),
                draft_layer_indices=run_cfg.get("draft_layer_indices"),
                tokenizer_name=run_cfg.get("tokenizer_name"),
                progen2_draft_mode=str(run_cfg.get("progen2_draft_mode", "pretrained")),
                device=run_cfg.get("device"),
                target_device=run_cfg.get("target_device") or (args.target_device if args.target_device else None),
                draft_device=run_cfg.get("draft_device") or (args.draft_device if args.draft_device else None),
                device_map=run_cfg.get("device_map") or (args.device_map if args.device_map else None),
                dtype=str(run_cfg.get("dtype", "float16")),
                seed=int(run_cfg.get("seed", 42)),
            )

            all_records: List[Dict[str, Any]] = []

            by_method: Dict[str, List[Dict[str, Any]]] = {}
            for method in ("target_baseline", "draft_baseline", "specdec"):
                req = GenerationRequest(**{**req_base.__dict__, "method": method})  # type: ignore[arg-type]
                recs = run_generation(cache, req)
                for r in recs:
                    r["run_idx"] = run_idx
                    r["sweep_group_id"] = sweep_group_id
                by_method[method] = recs
                all_records.extend(recs)

            if args.output_jsonl:
                append_jsonl(args.output_jsonl, all_records)

            # Load tokenizer once for suffix extraction.
            cm = load_models(cache, req_base)

            target_tot = _totals_from_records(by_method["target_baseline"])
            draft_tot = _totals_from_records(by_method["draft_baseline"])
            spec_tot = _totals_from_records(by_method["specdec"])

            speedup_vs_target = (spec_tot["tps"] / target_tot["tps"]) if target_tot["tps"] > 0 else 0.0

            acc_vals = [r.get("acceptance_rate") for r in by_method["specdec"] if r.get("acceptance_rate") is not None]
            mean_accept_rate: Optional[float]
            if acc_vals:
                mean_accept_rate = float(statistics.mean([float(x) for x in acc_vals]))
            else:
                mean_accept_rate = None

            L = req_base.gamma
            mean_accepted_prefix = (float(mean_accept_rate) * float(L)) if mean_accept_rate is not None else None

            sample_target_suffix = _suffix_from_record(cm, model_family, by_method["target_baseline"][0]) if by_method["target_baseline"] else None
            sample_draft_suffix = _suffix_from_record(cm, model_family, by_method["draft_baseline"][0]) if by_method["draft_baseline"] else None
            sample_specdec_suffix = _suffix_from_record(cm, model_family, by_method["specdec"][0]) if by_method["specdec"] else None

            import json as _json

            sample_prompt_ids = None
            if by_method["specdec"]:
                sample_prompt_ids = _json.dumps(list(by_method["specdec"][0].get("prompt_ids") or []))

            def _new_ids_json(method: str) -> str | None:
                if not by_method.get(method):
                    return None
                rec0 = by_method[method][0]
                return _json.dumps(list(rec0.get("new_ids") or []))

            sample_target_new_ids = _new_ids_json("target_baseline")
            sample_draft_new_ids = _new_ids_json("draft_baseline")
            sample_specdec_new_ids = _new_ids_json("specdec")

            prompt_len_tokens = int(by_method["specdec"][0]["prompt_len"]) if by_method["specdec"] else None

            # Draft metadata for paper/debugging: make the draft construction explicit in the wide CSV.
            if model_family == "progen2":
                draft_mode = str(getattr(req_base, "progen2_draft_mode", "pretrained"))
            else:
                # ProtGPT2 draft is always a truncated view of the target.
                draft_mode = "truncated"

            if draft_mode == "pretrained":
                draft_model_effective = req_base.draft_model_name
                draft_num_layers_effective = None
            else:
                draft_model_effective = req_base.target_model_name
                if req_base.draft_layer_indices is not None:
                    draft_num_layers_effective = int(len(req_base.draft_layer_indices))
                else:
                    draft_num_layers_effective = int(req_base.draft_layers)

            draft_layer_indices = json.dumps(req_base.draft_layer_indices or []) if draft_mode != "pretrained" else json.dumps([])

            # Populate the DNAGPT prompt identifier fields with protein-friendly values.
            hg_id = prompt_text if len(prompt_text) <= 80 else hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

            wide_row = {
                "model_family": model_family,
                "target_model_name": req_base.target_model_name,
                "draft_model_name": req_base.draft_model_name,
                "draft_mode": draft_mode,
                "draft_model_effective": draft_model_effective,
                "draft_num_layers_effective": draft_num_layers_effective,
                "draft_layer_indices": draft_layer_indices,
                "prompt_idx": prompt_idx,
                "hg_row_index": None,
                "hg_id": hg_id,
                "chrom": None,
                "start": None,
                "end": None,
                "prompt_len_bases": int(len(prompt_text)),
                "prompt_len_tokens": prompt_len_tokens,
                "prefix_len_tokens": 0,
                "target_context_len": req_base.target_context_len,
                "draft_context_len": req_base.draft_context_len,
                "L": L,
                "accept_mode": req_base.accept_mode,
                "temperature": req_base.temperature,
                "top_k": req_base.top_k,
                "top_p": req_base.top_p,
                "target_tps": target_tot["tps"],
                "draft_tps": draft_tot["tps"],
                "specdec_tps": spec_tot["tps"],
                "speedup_vs_target": speedup_vs_target,
                "mean_accept_rate": mean_accept_rate,
                "mean_accepted_prefix": mean_accepted_prefix,
                "target_tokens_total": target_tot["tokens_total"],
                "draft_tokens_total": draft_tot["tokens_total"],
                "specdec_tokens_total": spec_tot["tokens_total"],
                "target_time_total": target_tot["time_total"],
                "draft_time_total": draft_tot["time_total"],
                "specdec_time_total": spec_tot["time_total"],
                "sample_target_suffix": sample_target_suffix,
                "sample_draft_suffix": sample_draft_suffix,
                "sample_specdec_suffix": sample_specdec_suffix,
                "sample_prompt_ids": sample_prompt_ids,
                "sample_target_new_ids": sample_target_new_ids,
                "sample_draft_new_ids": sample_draft_new_ids,
                "sample_specdec_new_ids": sample_specdec_new_ids,
            }

            writer.writerow({k: wide_row.get(k) for k in wide_columns})
            out_f.flush()
            runs_written += 1

            if runs_written == 1 or (runs_written % 5 == 0):
                print(
                    json.dumps(
                        {
                            "runs_written": runs_written,
                            "last_prompt_idx": prompt_idx,
                            "shard_idx": args.shard_idx,
                            "num_shards": args.num_shards,
                            "output_csv": args.output_csv,
                        }
                    )
                )

    print(
        json.dumps(
            {
                "runs": runs_written,
                "output_csv": args.output_csv,
                "output_jsonl": (args.output_jsonl or None),
                "num_shards": args.num_shards,
                "shard_idx": args.shard_idx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
