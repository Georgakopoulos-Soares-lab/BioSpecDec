from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
import hashlib
from typing import Any, Dict, Iterable, List, Tuple

from .io_utils import append_jsonl, read_jsonl, write_csv
from .model_cache import ModelCache
from .runners import GenerationRequest, run_generation


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a dict of {param: [values...]} into list of assignments."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
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
    """then is a dict of key -> allowed_values(list)."""
    for k, allowed in then.items():
        if allowed is None:
            continue
        if not isinstance(allowed, list):
            raise ValueError(f"constraint.then.{k} must be a list")
        if run_cfg.get(k) not in allowed:
            return False
    return True


def _passes_constraints(run_cfg: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    """constraints: list of {"if": {k:v,...}, "then": {k:[allowed...], ...}}."""
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
    """A stable ID used to align runs across different methods.

    We intentionally exclude 'method' and a few per-run-only fields so you can
    join baseline/specdec results for the same prompt + parameters.
    """
    excluded = {
        "method",
    }

    # Canonicalize away irrelevant knobs so baselines/specdec align even if
    # a grid mistakenly varies them.
    canonical = dict(run_cfg)
    if canonical.get("model_family") == "progen2" and canonical.get("progen2_draft_mode", "pretrained") == "pretrained":
        canonical.pop("draft_layers", None)
        canonical.pop("draft_layer_indices", None)

    payload = {k: canonical.get(k) for k in sorted(canonical.keys()) if k not in excluded}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _req_from_dict(d: Dict[str, Any]) -> GenerationRequest:
    # normalize draft_layer_indices
    d = dict(d)
    if isinstance(d.get("draft_layer_indices"), str):
        s = d["draft_layer_indices"].strip()
        d["draft_layer_indices"] = [int(x.strip()) for x in s.split(",") if x.strip()] if s else None

    # In ProGen2 pretrained draft mode, truncation controls must be ignored.
    if d.get("model_family") == "progen2" and str(d.get("progen2_draft_mode", "pretrained")) == "pretrained":
        d["draft_layer_indices"] = None
        # Keep a sentinel to make it obvious this is unused downstream.
        d["draft_layers"] = 0

    return GenerationRequest(
        model_family=d["model_family"],
        method=d["method"],
        prompt_text=d["prompt_text"],
        max_new_tokens=int(d.get("max_new_tokens", d.get("num_tokens", 256))),
        num_samples=int(d.get("num_samples", 5)),
        temperature=float(d.get("temperature", 1.0)),
        top_k=int(d.get("top_k", 0)),
        top_p=float(d.get("top_p", 0.0)),
        gamma=int(d.get("gamma", 5)),
        accept_mode=str(d.get("accept_mode", "prob")),
        target_model_name=str(d.get("target_model_name", "")),
        draft_model_name=str(d.get("draft_model_name", "")),
        target_weight=d.get("target_weight"),
        draft_weight=d.get("draft_weight"),
        max_prompt_tokens=int(d.get("max_prompt_tokens", 0)),
        target_context_len=d.get("target_context_len"),
        draft_context_len=d.get("draft_context_len"),
        draft_layers=int(d.get("draft_layers", 8)),
        draft_layer_indices=d.get("draft_layer_indices"),
        tokenizer_name=d.get("tokenizer_name"),
        progen2_draft_mode=str(d.get("progen2_draft_mode", "pretrained")),
        device=d.get("device"),
        dtype=str(d.get("dtype", "float16")),
        seed=int(d.get("seed", 42)),
    )


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Grid sweep runner for unified specdec experiments.")
    p.add_argument("--config", required=True, help="Path to a sweep JSON config")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--num_shards", type=int, default=1, help="Split runs into N shards (for multi-GPU).")
    p.add_argument("--shard_idx", type=int, default=0, help="Which shard to run: 0..N-1")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base = cfg.get("base", {})
    grid = cfg.get("grid", {})
    constraints = cfg.get("constraints", [])

    if constraints and not isinstance(constraints, list):
        raise ValueError("config.constraints must be a list")

    if not isinstance(grid, dict):
        raise ValueError("config.grid must be a dict of param -> [values]")

    combos = _expand_grid(grid)
    if not combos:
        combos = [{}]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise ValueError("--shard_idx must be in [0, num_shards)")

    cache = ModelCache()

    summary_rows: List[Dict[str, Any]] = []

    for run_idx, combo in enumerate(combos):
        if (run_idx % args.num_shards) != args.shard_idx:
            continue
        run_cfg = _merge(base, combo)

        if constraints and (not _passes_constraints(run_cfg, constraints)):
            continue

        req = _req_from_dict(run_cfg)

        sweep_group_id = _stable_group_id(run_cfg)

        start = time.time()
        records = run_generation(cache, req)

        # tag each per-sample record with sweep metadata for easy alignment
        for r in records:
            r["run_idx"] = run_idx
            r["sweep_group_id"] = sweep_group_id
        append_jsonl(args.output_jsonl, records)

        # one summary row per combo
        tps = [r["tokens_per_s"] for r in records]
        acc = [r["acceptance_rate"] for r in records if r.get("acceptance_rate") is not None]

        summary_rows.append(
            {
                "run_idx": run_idx,
                "sweep_group_id": sweep_group_id,
                "model_family": req.model_family,
                "method": req.method,
                "prompt_text": req.prompt_text,
                "target_model_name": req.target_model_name,
                "draft_model_name": req.draft_model_name,
                "max_new_tokens": req.max_new_tokens,
                "num_samples": req.num_samples,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "gamma": req.gamma,
                "accept_mode": req.accept_mode,
                "max_prompt_tokens": req.max_prompt_tokens,
                "draft_layers": req.draft_layers,
                "progen2_draft_mode": run_cfg.get("progen2_draft_mode", None),
                "seed": req.seed,
                "tokens_per_s_mean": sum(tps) / max(1, len(tps)),
                "acceptance_rate_mean": (sum(acc) / max(1, len(acc))) if acc else None,
                "wall_time_s": time.time() - start,
            }
        )

    write_csv(args.output_csv, summary_rows)

    print(
        json.dumps(
            {
                "runs": len(summary_rows),
                "output_jsonl": args.output_jsonl,
                "output_csv": args.output_csv,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
