from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from .io_utils import ensure_parent_dir, read_jsonl, write_csv


def _resolve_device_dtype(device: Optional[str], dtype_str: str) -> Tuple[str, torch.dtype]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        dtype = getattr(torch, dtype_str)
    else:
        dtype = torch.float32
    return str(device), dtype


def _parse_list_cell(val: Any) -> List[int]:
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [int(x) for x in out]
    except Exception:
        pass
    return []


def _score_continuation_logprob(
    model: Any,
    input_ids: torch.Tensor,
    prompt_len: int,
    reduce: str = "mean",
) -> Tuple[float, int]:
    """Score log p(x_prompt+cont) but *only* over continuation tokens.

    input_ids: [1, T]
    prompt_len: number of prompt tokens (continuation starts at this index)

    We compute sum/mean over tokens t in [prompt_len, T-1] using logits at t-1.
    """

    if input_ids.size(1) <= max(1, prompt_len):
        return float("nan"), 0

    with torch.inference_mode():
        # Support both HF-style models (return object with .logits)
        # and DNAGPT-style models (return raw logits tensor).
        try:
            out = model(input_ids=input_ids)
        except TypeError:
            out = model(input_ids)

        logits = out.logits if hasattr(out, "logits") else out  # [1, T, V]

    logprobs = F.log_softmax(logits, dim=-1)

    # Predict token at position t from logits at t-1
    targets = input_ids[:, 1:]  # [1, T-1]
    pred = logprobs[:, :-1, :]  # [1, T-1, V]

    token_logp = torch.gather(pred, 2, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    start = max(0, prompt_len - 1)
    # token_logp indices correspond to targets positions 1..T-1
    # continuation tokens start at target index = prompt_len
    # which corresponds to token_logp index = prompt_len-1
    cont = token_logp[0, start:]

    if cont.numel() == 0:
        return float("nan"), 0

    if reduce == "sum":
        return float(cont.sum().item()), int(cont.numel())
    if reduce == "mean":
        return float(cont.mean().item()), int(cont.numel())

    raise ValueError(f"Unknown reduce={reduce}")


def _load_hf_causal_lm(model_name: str, device: str, dtype: torch.dtype, trust_remote_code: bool = False):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


@dataclass(frozen=True)
class _LoadedScorer:
    model: Any
    model_family: str
    model_name: str
    weight_path: Optional[str]


def _load_scoring_model(
    score_model_family: str,
    model_name: str,
    weight_path: Optional[str],
    device: str,
    dtype: torch.dtype,
) -> _LoadedScorer:
    if score_model_family == "dnagpt":
        if not weight_path:
            raise ValueError("DNAGPT scoring requires --weight_path")
        from scoring_specdec_beam_search import get_model as dnagpt_get_model
        from scoring_specdec_beam_search import load_model as dnagpt_load_model

        model, _tok = dnagpt_get_model(model_name)
        model = dnagpt_load_model(model, weight_path, device=device, dtype=dtype)
        return _LoadedScorer(model=model, model_family=score_model_family, model_name=model_name, weight_path=weight_path)

    if score_model_family == "protgpt2":
        model = _load_hf_causal_lm(model_name, device=device, dtype=dtype, trust_remote_code=False)
        return _LoadedScorer(model=model, model_family=score_model_family, model_name=model_name, weight_path=None)

    if score_model_family == "progen2":
        model = _load_hf_causal_lm(model_name, device=device, dtype=dtype, trust_remote_code=True)
        return _LoadedScorer(model=model, model_family=score_model_family, model_name=model_name, weight_path=None)

    raise ValueError(score_model_family)


def _infer_from_wide_csv(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    mf = None
    mn = None
    if "model_family" in df.columns:
        vals = [x for x in df["model_family"].dropna().unique().tolist() if str(x).strip()]
        if vals:
            mf = str(vals[0]).strip()
    if "target_model_name" in df.columns:
        vals = [x for x in df["target_model_name"].dropna().unique().tolist() if str(x).strip()]
        if vals:
            mn = str(vals[0]).strip()
    return mf, mn


def _score_wide_csv(
    input_csv: str,
    output_csv: str,
    scorer: _LoadedScorer,
    device: str,
    reduce: str,
) -> None:
    df = pd.read_csv(input_csv)

    required = [
        "sample_prompt_ids",
        "sample_target_new_ids",
        "sample_draft_new_ids",
        "sample_specdec_new_ids",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Wide CSV is missing required columns for scoring: "
            + ", ".join(missing)
            + ". Expected a wide sweep CSV (not JSONL)."
        )

    out_cols: Dict[str, List[float]] = {}

    for _, row in df.iterrows():
        prompt_ids = _parse_list_cell(row.get("sample_prompt_ids"))
        prompt_len = len(prompt_ids)

        for prefix, new_col in (
            ("target", "sample_target_new_ids"),
            ("draft", "sample_draft_new_ids"),
            ("specdec", "sample_specdec_new_ids"),
        ):
            new_ids = _parse_list_cell(row.get(new_col))
            full_ids = prompt_ids + new_ids

            input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
            logprob, cont_len = _score_continuation_logprob(scorer.model, input_ids, prompt_len=prompt_len, reduce=reduce)

            # Match existing scored-wide schema: *_suffix_logprob_mean, *_suffix_nll_mean, *_suffix_ppl
            # (The "mean" here refers to mean over continuation tokens.)
            lp_key = f"{prefix}_suffix_logprob_{reduce}"
            nll_key = f"{prefix}_suffix_nll_{reduce}"

            if reduce == "mean":
                lp_out_key = f"{prefix}_suffix_logprob_mean"
                nll_out_key = f"{prefix}_suffix_nll_mean"
                ppl_key = f"{prefix}_suffix_ppl"
                out_cols.setdefault(lp_out_key, []).append(float(logprob))
                out_cols.setdefault(nll_out_key, []).append(float((-logprob) if not math.isnan(logprob) else float("nan")))
                if math.isnan(logprob) or cont_len <= 0:
                    out_cols.setdefault(ppl_key, []).append(float("nan"))
                else:
                    out_cols.setdefault(ppl_key, []).append(float(math.exp(-logprob)))
            else:
                # Keep sum columns if requested; also compute ppl using per-token mean.
                out_cols.setdefault(lp_key, []).append(float(logprob))
                out_cols.setdefault(nll_key, []).append(float((-logprob) if not math.isnan(logprob) else float("nan")))
                ppl_key = f"{prefix}_suffix_ppl"
                if math.isnan(logprob) or cont_len <= 0:
                    out_cols.setdefault(ppl_key, []).append(float("nan"))
                else:
                    out_cols.setdefault(ppl_key, []).append(float(math.exp((-logprob) / cont_len)))

    for k, v in out_cols.items():
        df[k] = v

    df["scoring_model_family"] = scorer.model_family
    df["scoring_model_name"] = scorer.model_name
    df["scoring_weight_path"] = scorer.weight_path

    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Score continuation likelihoods. Supports:\n"
            "(1) JSONL from pipeline/run_generate.py (writes augmented JSONL + optional summary CSV), and\n"
            "(2) wide sweep CSVs (writes scored wide CSV with *_suffix_ppl columns)."
        )
    )

    # JSONL mode (pipeline.run_generate)
    p.add_argument("--input_jsonl", default=None)
    p.add_argument("--output_jsonl", default=None)

    # Wide CSV mode (sweeps)
    p.add_argument("--input_csv", default=None)
    p.add_argument("--output_csv", default=None, help="For wide CSV mode: scored CSV output. For JSONL mode: optional summary CSV.")

    p.add_argument("--score_model_family", default=None, choices=["dnagpt", "protgpt2", "progen2"])

    # Scoring model identity
    p.add_argument("--model_name", default=None)
    p.add_argument("--weight_path", default=None, help="DNAGPT checkpoint path (only for dnagpt)")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--reduce", default="mean", choices=["mean", "sum"])

    args = p.parse_args(argv)

    device, dtype = _resolve_device_dtype(args.device, args.dtype)

    # Auto-select mode.
    wide_mode = bool(args.input_csv)
    jsonl_mode = bool(args.input_jsonl)
    if wide_mode and jsonl_mode:
        raise SystemExit("Pass either --input_csv or --input_jsonl (not both).")
    if not wide_mode and not jsonl_mode:
        raise SystemExit("Must pass one of: --input_csv or --input_jsonl")

    if wide_mode:
        if not args.output_csv:
            raise SystemExit("Wide CSV mode requires --output_csv")

        df_head = pd.read_csv(args.input_csv, nrows=50)
        inferred_family, inferred_name = _infer_from_wide_csv(df_head)
        score_model_family = (args.score_model_family or inferred_family or "").strip()
        model_name = (args.model_name or inferred_name or "").strip()

        if not score_model_family:
            raise SystemExit("--score_model_family not provided and could not infer from wide CSV")
        if not model_name:
            raise SystemExit("--model_name not provided and could not infer from wide CSV")

        scorer = _load_scoring_model(
            score_model_family=score_model_family,
            model_name=model_name,
            weight_path=args.weight_path,
            device=device,
            dtype=dtype,
        )

        _score_wide_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            scorer=scorer,
            device=device,
            reduce=args.reduce,
        )

        print(
            json.dumps(
                {
                    "mode": "wide_csv",
                    "input_csv": args.input_csv,
                    "output_csv": args.output_csv,
                    "scoring_model_family": scorer.model_family,
                    "scoring_model_name": scorer.model_name,
                    "device": device,
                    "dtype": str(dtype),
                },
                indent=2,
            )
        )
        return

    # JSONL mode
    if not args.output_jsonl:
        raise SystemExit("JSONL mode requires --output_jsonl")
    if not args.score_model_family:
        raise SystemExit("JSONL mode requires --score_model_family")
    if not args.model_name:
        raise SystemExit("JSONL mode requires --model_name")

    scorer = _load_scoring_model(
        score_model_family=args.score_model_family,
        model_name=args.model_name,
        weight_path=args.weight_path,
        device=device,
        dtype=dtype,
    )

    model = scorer.model

    summary_rows: List[Dict[str, Any]] = []

    tps_by_method: Dict[str, List[float]] = {}
    lp_by_method: Dict[str, List[float]] = {}

    ensure_parent_dir(args.output_jsonl)
    records_scored = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for rec in read_jsonl(args.input_jsonl):
        # Expect these fields from run_generate.py
            prompt_len = int(rec["prompt_len"])
            full_ids = rec["full_ids"]

            input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)

            logprob, cont_len = _score_continuation_logprob(model, input_ids, prompt_len=prompt_len, reduce=args.reduce)
            nll = (-logprob) if not math.isnan(logprob) else float("nan")

            # Define perplexity as exp(mean NLL per continuation token).
            if math.isnan(logprob) or cont_len <= 0:
                ppl = float("nan")
            else:
                if args.reduce == "mean":
                    ppl = math.exp(-logprob)
                else:
                    ppl = math.exp((-logprob) / cont_len)

            rec2 = dict(rec)
            rec2[f"score_logprob_{args.reduce}"] = logprob
            rec2[f"score_nll_{args.reduce}"] = nll
            rec2["score_ppl"] = ppl
            rec2["scoring_model_family"] = args.score_model_family
            rec2["scoring_model_name"] = args.model_name
            rec2["scoring_weight_path"] = args.weight_path

            out_f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            records_scored += 1

            m = rec.get("method", "")
            tps = rec.get("tokens_per_s")
            if tps is not None:
                tps_by_method.setdefault(m, []).append(float(tps))
            if not math.isnan(logprob):
                lp_by_method.setdefault(m, []).append(float(logprob))

    if args.output_csv:
        for method, vals in tps_by_method.items():
            lps = lp_by_method.get(method, [])
            summary_rows.append(
                {
                    "method": method,
                    "n": len(vals),
                    "tokens_per_s_mean": statistics.mean(vals) if vals else None,
                    f"score_logprob_{args.reduce}_mean": statistics.mean(lps) if lps else None,
                }
            )
        write_csv(args.output_csv, summary_rows)

    print(
        json.dumps(
            {
                "mode": "jsonl",
                "scoring_model_family": args.score_model_family,
                "scoring_model_name": args.model_name,
                "device": device,
                "dtype": str(dtype),
                "records_scored": records_scored,
                "output_jsonl": args.output_jsonl,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
