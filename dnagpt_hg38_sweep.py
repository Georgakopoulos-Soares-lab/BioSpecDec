#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

_MODEL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
        return

    # stable column order: common keys first, then the rest
    preferred = [
        "run_idx",
        "sweep_group_id",
        "model_family",
        "method",
        "prompt_id",
        "hg_row_index",
        "hg_id",
        "chrom",
        "start",
        "end",
        "target_model_name",
        "draft_model_name",
        "target_weight",
        "draft_weight",
        "max_new_tokens",
        "num_samples",
        "temperature",
        "top_k",
        "top_p",
        "gamma",
        "accept_mode",
        "max_prompt_tokens",
        "target_context_len",
        "draft_context_len",
        "seed",
        "tokens_per_s_mean",
        "acceptance_rate_mean",
        "wall_time_s",
    ]

    keys = set().union(*(r.keys() for r in rows))
    fieldnames = [k for k in preferred if k in keys] + sorted([k for k in keys if k not in preferred])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return out


def _stable_group_id(payload: Dict[str, Any]) -> str:
    excluded = {"method", "prompt_text"}
    canonical = dict(payload)
    # Avoid hashing huge prompts; prompt_id is enough.
    if canonical.get("prompt_id"):
        canonical["prompt_text"] = canonical.get("prompt_id")

    blob = json.dumps(
        {k: canonical.get(k) for k in sorted(canonical.keys()) if k not in excluded},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _parse_csv_prompts(
    hg_csv: str,
    hg_prefix: str,
    num_prompts: int,
    hg_row_indices: Optional[List[int]],
    hg_ids: Optional[List[str]],
    seed: int,
) -> List[Dict[str, Any]]:
    with open(hg_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in hg_csv={hg_csv}")

    selected: List[Tuple[int, Dict[str, Any]]] = []

    if hg_ids:
        id_set = set(hg_ids)
        for idx, r in enumerate(rows):
            if r.get("id") in id_set:
                selected.append((idx, r))
        missing = [x for x in hg_ids if x not in {r.get("id") for _, r in selected}]
        if missing:
            raise ValueError(f"hg_ids not found in CSV: {missing[:10]}" + (" ..." if len(missing) > 10 else ""))

    elif hg_row_indices:
        for i in hg_row_indices:
            if i < 0 or i >= len(rows):
                raise ValueError(f"hg_row_index out of range: {i} (rows={len(rows)})")
            selected.append((i, rows[i]))

    else:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for i in indices[: max(1, int(num_prompts))]:
            selected.append((i, rows[i]))

    # If caller provided a candidate set (ids/row_indices) larger than num_prompts,
    # sample a stable subset so configs like "hg_row_indices=[...], num_prompts=5" behave as expected.
    if num_prompts and num_prompts > 0 and len(selected) > num_prompts:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:num_prompts]

    prompts: List[Dict[str, Any]] = []
    for prompt_idx, (row_idx, r) in enumerate(selected):
        seq = r.get("seq")
        if not seq:
            raise ValueError(f"Row {row_idx} missing 'seq'")

        hg_id = r.get("id") or f"row_{row_idx}"
        prompt_text = f"{hg_prefix}{seq}"

        prompts.append(
            {
                "prompt_idx": prompt_idx,
                "prompt_id": hg_id,
                "prompt_text": prompt_text,
                "prompt_len_bases": int(len(seq)),
                "hg_row_index": row_idx,
                "chrom": r.get("chrom"),
                "start": int(r["start"]) if r.get("start") not in (None, "") else None,
                "end": int(r["end"]) if r.get("end") not in (None, "") else None,
            }
        )

    return prompts


def _maybe_seed(seed: int) -> None:
    try:
        import random as _random

        _random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_models(run_cfg: Dict[str, Any]):
    import torch

    key = (
        "dnagpt",
        str(run_cfg.get("target_model_name")),
        str(run_cfg.get("draft_model_name")),
        os.path.abspath(str(run_cfg.get("target_weight"))),
        os.path.abspath(str(run_cfg.get("draft_weight"))),
        str(run_cfg.get("device")),
        str(run_cfg.get("dtype")),
    )

    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit

    from scoring_specdec_beam_search import get_model as dnagpt_get_model
    from scoring_specdec_beam_search import load_model as dnagpt_load_model

    device = run_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, str(run_cfg.get("dtype", "float16"))) if device == "cuda" else torch.float32

    target_model_name = str(run_cfg.get("target_model_name"))
    draft_model_name = str(run_cfg.get("draft_model_name"))
    target_weight = run_cfg.get("target_weight")
    draft_weight = run_cfg.get("draft_weight")

    if not target_weight or not draft_weight:
        raise ValueError("DNAGPT requires target_weight and draft_weight")

    _, tokenizer = dnagpt_get_model(target_model_name)

    target_model, _ = dnagpt_get_model(target_model_name)
    draft_model, _ = dnagpt_get_model(draft_model_name)

    target_model = dnagpt_load_model(target_model, target_weight, device=device, dtype=dtype)
    draft_model = dnagpt_load_model(draft_model, draft_weight, device=device, dtype=dtype)

    cm = {"target_model": target_model, "draft_model": draft_model, "tokenizer": tokenizer, "device": device, "dtype": str(dtype)}
    _MODEL_CACHE[key] = cm
    return cm


def _encode_prompt(tokenizer, prompt_text: str, device: str, max_prompt_tokens: int):
    import torch

    ids_1d = tokenizer.encode(prompt_text, device=str(device))
    input_ids = ids_1d.unsqueeze(0)
    if max_prompt_tokens and max_prompt_tokens > 0 and input_ids.size(1) > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
    return input_ids


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(ids[0].tolist())


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DNAGPT hg38 sweep runner (JSON config + hg38 CSV prompt sampling).")
    p.add_argument("--config", required=True, help="Path to a DNAGPT hg38 sweep JSON config")
    p.add_argument(
        "--output_csv",
        required=True,
        help=(
            "Wide-format per-run CSV (one row per prompt+hyperparams; includes target/draft/specdec columns). "
            "This is the recommended paper-ready output."
        ),
    )
    p.add_argument("--num_shards", type=int, default=1, help="Split runs into N shards (for multi-GPU).")
    p.add_argument("--shard_idx", type=int, default=0, help="Which shard to run: 0..N-1")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base = cfg.get("base", {})
    grid = cfg.get("grid", {})
    prompt_source = cfg.get("prompt_source", {})

    if not isinstance(base, dict) or not isinstance(grid, dict) or not isinstance(prompt_source, dict):
        raise ValueError("config must include dicts: base, grid, prompt_source")

    hg_csv = prompt_source.get("hg_csv", "hg38_sequences.csv")
    hg_prefix = prompt_source.get("hg_prefix", "<R>")
    num_prompts = int(prompt_source.get("num_prompts", 1))

    hg_row_indices = prompt_source.get("hg_row_indices")
    if isinstance(hg_row_indices, str):
        hg_row_indices = [int(x.strip()) for x in hg_row_indices.split(",") if x.strip()]

    hg_ids = prompt_source.get("hg_ids")
    if isinstance(hg_ids, str):
        hg_ids = [x.strip() for x in hg_ids.split(",") if x.strip()]

    seed = int(prompt_source.get("seed", base.get("seed", 42)))

    prompts = _parse_csv_prompts(
        hg_csv=hg_csv,
        hg_prefix=hg_prefix,
        num_prompts=num_prompts,
        hg_row_indices=hg_row_indices,
        hg_ids=hg_ids,
        seed=seed,
    )

    # The wide-format output always runs all three methods together per grid point.
    # If the config includes a method sweep, drop it to avoid redundant repeats.
    grid2 = dict(grid)
    grid2.pop("method", None)
    combos = _expand_grid(grid2) or [{}]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise ValueError("--shard_idx must be in [0, num_shards)")


    from scoring_specdec_beam_search import run_benchmarks_for_prompt

    wide_columns = [
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

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    runs_written = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=wide_columns)
        writer.writeheader()
        out_f.flush()

        run_unit_idx = 0
        for prompt in prompts:
            for combo in combos:
                if (run_unit_idx % args.num_shards) != args.shard_idx:
                    run_unit_idx += 1
                    continue
                run_unit_idx += 1

                run_cfg = _merge(base, combo)
                run_cfg["prompt_text"] = prompt["prompt_text"]
                run_cfg["prompt_id"] = prompt["prompt_id"]

                _maybe_seed(int(run_cfg.get("seed", 42)))

                cm = _load_models(run_cfg)
                target_model = cm["target_model"]
                draft_model = cm["draft_model"]
                tokenizer = cm["tokenizer"]
                device = cm["device"]

                max_prompt_tokens = int(run_cfg.get("max_prompt_tokens", 0))
                input_ids = _encode_prompt(tokenizer, run_cfg["prompt_text"], device=device, max_prompt_tokens=max_prompt_tokens)

                # Token counts
                prompt_len_tokens = int(input_ids.size(1))
                prefix_len_tokens = int(tokenizer.encode(hg_prefix, device=str(device)).numel())

                max_new_tokens = int(run_cfg.get("max_new_tokens", run_cfg.get("num_tokens", 256)))
                num_samples = int(run_cfg.get("num_samples", 5))

                temperature = float(run_cfg.get("temperature", 1.0))
                top_k = int(run_cfg.get("top_k", 0))
                top_p = float(run_cfg.get("top_p", 0.0))

                L = int(run_cfg.get("gamma", 5))
                accept_mode = str(run_cfg.get("accept_mode", "prob"))

                target_context_len = run_cfg.get("target_context_len")
                draft_context_len = run_cfg.get("draft_context_len")

                out = run_benchmarks_for_prompt(
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=tokenizer,
                    prompt_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_samples=num_samples,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    L=L,
                    accept_mode=accept_mode,
                    target_context_len=target_context_len,
                    draft_context_len=draft_context_len,
                    debug=False,
                    verbose=False,
                )

                # Paper-ready wide row
                wide_row = {
                    "prompt_idx": prompt.get("prompt_idx"),
                    "hg_row_index": prompt.get("hg_row_index"),
                    "hg_id": prompt.get("prompt_id"),
                    "chrom": prompt.get("chrom"),
                    "start": prompt.get("start"),
                    "end": prompt.get("end"),
                    "prompt_len_bases": prompt.get("prompt_len_bases"),
                    "prompt_len_tokens": prompt_len_tokens,
                    "prefix_len_tokens": prefix_len_tokens,
                    "target_context_len": target_context_len,
                    "draft_context_len": draft_context_len,
                    "L": L,
                    "accept_mode": accept_mode,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "target_tps": out.get("target_tps"),
                    "draft_tps": out.get("draft_tps"),
                    "specdec_tps": out.get("specdec_tps"),
                    "speedup_vs_target": out.get("speedup_vs_target"),
                    "mean_accept_rate": out.get("mean_accept_rate"),
                    "mean_accepted_prefix": out.get("mean_accepted_prefix"),
                    "target_tokens_total": out.get("target_tokens_total"),
                    "draft_tokens_total": out.get("draft_tokens_total"),
                    "specdec_tokens_total": out.get("specdec_tokens_total"),
                    "target_time_total": out.get("target_time_total"),
                    "draft_time_total": out.get("draft_time_total"),
                    "specdec_time_total": out.get("specdec_time_total"),
                    "sample_target_suffix": out.get("sample_target_suffix"),
                    "sample_draft_suffix": out.get("sample_draft_suffix"),
                    "sample_specdec_suffix": out.get("sample_specdec_suffix"),
                    "sample_prompt_ids": out.get("sample_prompt_ids"),
                    "sample_target_new_ids": out.get("sample_target_new_ids"),
                    "sample_draft_new_ids": out.get("sample_draft_new_ids"),
                    "sample_specdec_new_ids": out.get("sample_specdec_new_ids"),
                }

                writer.writerow({k: wide_row.get(k) for k in wide_columns})
                out_f.flush()
                runs_written += 1

                if runs_written == 1 or (runs_written % 5 == 0):
                    print(
                        json.dumps(
                            {
                                "runs_written": runs_written,
                                "last_prompt_idx": prompt.get("prompt_idx"),
                                "last_hg_id": prompt.get("prompt_id"),
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
                "prompts": len(prompts),
                "output_csv": args.output_csv,
                "num_shards": args.num_shards,
                "shard_idx": args.shard_idx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
