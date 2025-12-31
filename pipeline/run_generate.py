from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List

from .io_utils import append_jsonl, write_csv
from .model_cache import ModelCache
from .runners import GenerationRequest, run_generation


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tps = [r["tokens_per_s"] for r in records if r.get("tokens_per_s") is not None]
    times = [r["wall_time_s"] for r in records if r.get("wall_time_s") is not None]
    nnew = [r["num_new_tokens"] for r in records if r.get("num_new_tokens") is not None]
    acc = [r["acceptance_rate"] for r in records if r.get("acceptance_rate") is not None]

    out: Dict[str, Any] = {
        "n": len(records),
        "tokens_per_s_mean": statistics.mean(tps) if tps else None,
        "tokens_per_s_p50": statistics.median(tps) if tps else None,
        "wall_time_s_mean": statistics.mean(times) if times else None,
        "num_new_tokens_mean": statistics.mean(nnew) if nnew else None,
        "acceptance_rate_mean": statistics.mean(acc) if acc else None,
    }
    return out


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Unified generation runner (baseline/specdec) that writes JSONL records.")

    p.add_argument("--model_family", required=True, choices=["dnagpt", "protgpt2", "progen2"])
    p.add_argument("--method", required=True, choices=["target_baseline", "draft_baseline", "specdec"])

    p.add_argument("--prompt", required=True)
    p.add_argument("--num_tokens", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=5)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.0)

    p.add_argument("--gamma", type=int, default=5)
    p.add_argument("--accept_mode", default="prob", choices=["prob", "pt_gt_pd", "match"])

    p.add_argument("--target_model_name", required=True)
    p.add_argument("--draft_model_name", default="")

    p.add_argument("--target_weight", default=None)
    p.add_argument("--draft_weight", default=None)

    p.add_argument("--max_prompt_tokens", type=int, default=0)
    p.add_argument("--target_context_len", type=int, default=None)
    p.add_argument("--draft_context_len", type=int, default=None)

    p.add_argument(
        "--draft_layers",
        type=int,
        default=8,
        help="Draft truncation depth (used for ProtGPT2 drafts and ProGen2 when --progen2_draft_mode=truncated; ignored for ProGen2 pretrained drafts).",
    )
    p.add_argument(
        "--draft_layer_indices",
        type=str,
        default="",
        help="Comma-separated layer indices for truncated drafts (ProtGPT2 or ProGen2 truncated mode). Ignored for ProGen2 pretrained drafts.",
    )

    p.add_argument("--tokenizer_name", type=str, default=None)

    # ProGen2 draft mode
    p.add_argument(
        "--progen2_draft_mode",
        default="pretrained",
        choices=["pretrained", "truncated"],
        help="For model_family=progen2: draft type (separate pretrained model, or truncated view of target).",
    )

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_jsonl", required=True, help="Append JSONL records here")
    p.add_argument("--output_csv", default=None, help="Write a one-row summary CSV here")

    args = p.parse_args(argv)

    draft_layer_indices = _parse_int_list(args.draft_layer_indices)

    # Enforce semantics: ProGen2 pretrained drafts always use the full pretrained model.
    # Truncation controls only apply when progen2_draft_mode=truncated.
    effective_draft_layers = args.draft_layers
    effective_draft_layer_indices = (draft_layer_indices if draft_layer_indices else None)
    if args.model_family == "progen2" and args.progen2_draft_mode == "pretrained":
        effective_draft_layers = 0
        effective_draft_layer_indices = None

    req = GenerationRequest(
        model_family=args.model_family,
        method=args.method,
        prompt_text=args.prompt,
        max_new_tokens=args.num_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        gamma=args.gamma,
        accept_mode=args.accept_mode,
        target_model_name=args.target_model_name,
        draft_model_name=args.draft_model_name,
        target_weight=args.target_weight,
        draft_weight=args.draft_weight,
        max_prompt_tokens=args.max_prompt_tokens,
        target_context_len=args.target_context_len,
        draft_context_len=args.draft_context_len,
        draft_layers=effective_draft_layers,
        draft_layer_indices=effective_draft_layer_indices,
        tokenizer_name=args.tokenizer_name,
        progen2_draft_mode=args.progen2_draft_mode,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )

    cache = ModelCache()

    records = run_generation(cache, req)
    append_jsonl(args.output_jsonl, records)

    summary = {
        "run_ts": time.time(),
        "model_family": args.model_family,
        "method": args.method,
        "prompt": args.prompt,
        "target_model_name": args.target_model_name,
        "draft_model_name": args.draft_model_name,
        "num_samples": args.num_samples,
        "num_tokens": args.num_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "gamma": args.gamma,
        "accept_mode": args.accept_mode,
        "max_prompt_tokens": args.max_prompt_tokens,
        "target_context_len": args.target_context_len,
        "draft_context_len": args.draft_context_len,
        "draft_layers": args.draft_layers,
        "draft_layer_indices": draft_layer_indices,
        "tokenizer_name": args.tokenizer_name,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        **_summarize(records),
    }

    if args.output_csv:
        write_csv(args.output_csv, [summary])

    # Print machine-readable summary as last line
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
