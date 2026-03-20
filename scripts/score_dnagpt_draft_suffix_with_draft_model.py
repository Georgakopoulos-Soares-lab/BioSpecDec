#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from scoring_specdec_beam_search import get_model as dnagpt_get_model
from scoring_specdec_beam_search import load_model as dnagpt_load_model


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


def _resolve_device_dtype(device: Optional[str], dtype_str: str) -> Tuple[str, torch.dtype]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        dtype = getattr(torch, dtype_str)
    else:
        dtype = torch.float32
    return str(device), dtype


def _score_continuation_mean_logprob(model: Any, input_ids: torch.Tensor, prompt_len: int) -> Tuple[float, int]:
    """Mean logprob of continuation tokens (tokens after prompt).

    input_ids: [1, T]
    prompt_len: number of prompt tokens; continuation tokens are positions [prompt_len..T-1]

    Returns (mean_logprob, num_cont_tokens).
    """

    if input_ids.size(1) <= max(1, prompt_len):
        return float("nan"), 0

    with torch.inference_mode():
        out = model(input_ids)  # DNAGPT returns raw logits tensor [1, T, V]
    logits = out
    logprobs = F.log_softmax(logits, dim=-1)

    targets = input_ids[:, 1:]  # [1, T-1]
    pred = logprobs[:, :-1, :]  # [1, T-1, V]
    token_logp = torch.gather(pred, 2, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    # token_logp index i corresponds to log p(token at position i+1)
    # continuation starts at token position prompt_len, so index prompt_len-1
    start = max(0, prompt_len - 1)
    cont = token_logp[0, start:]
    if cont.numel() == 0:
        return float("nan"), 0

    return float(cont.mean().item()), int(cont.numel())


def _resolve_dnagpt_model_max_len(model: Any) -> int:
    max_len = getattr(model, "max_len", None)
    if isinstance(max_len, int) and max_len > 0:
        return max_len
    try:
        wpe = getattr(getattr(model, "transformer", None), "wpe", None)
        num = getattr(wpe, "num_embeddings", None)
        if isinstance(num, int) and num > 0:
            return num
    except Exception:
        pass
    raise ValueError("Could not resolve DNAGPT model max_len")


def _score_continuation_mean_logprob_sliding(
    *,
    full_ids: List[int],
    prompt_len: int,
    model: Any,
    device: str,
) -> Tuple[float, int]:
    """Mean logprob over continuation tokens using a bounded context (sliding window)."""

    if prompt_len < 1:
        return float("nan"), 0

    max_len = _resolve_dnagpt_model_max_len(model)
    T = len(full_ids)
    if T <= 1:
        return float("nan"), 0

    suffix_start_pos = prompt_len
    suffix_end_pos = T - 1
    if suffix_end_pos < suffix_start_pos:
        return float("nan"), 0

    stride = max(1, max_len // 2)
    collected: Dict[int, float] = {}

    # Iterate windows; each window predicts tokens at positions [begin+1 .. end-1].
    for i in range(0, T - 1, stride):
        end_loc = min(i + stride + 1, T)
        begin_loc = max(0, end_loc - max_len)
        seg = full_ids[begin_loc:end_loc]
        if len(seg) < 2:
            continue

        input_ids = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
        with torch.inference_mode():
            logits = model(input_ids)  # [1, L, V]

        # Compute token logprobs in fp32 for stability.
        logprobs_full = F.log_softmax(logits.float(), dim=-1)  # [1, L, V]
        logprobs_full = logprobs_full[:, :-1]  # [1, L-1, V]
        targets = input_ids[:, 1:]  # [1, L-1]
        token_logprobs = torch.gather(logprobs_full, 2, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

        seg_pred_start = begin_loc + 1
        seg_pred_end = end_loc - 1

        # Only take the new region to avoid duplicates.
        take_start = max(i + 1, seg_pred_start)
        if take_start > seg_pred_end:
            continue
        local_start = take_start - seg_pred_start
        local_len = seg_pred_end - take_start + 1
        vals = token_logprobs[0, local_start:local_start + local_len].detach().cpu().tolist()
        for offset, lp in enumerate(vals):
            collected[take_start + offset] = float(lp)

    vals = [collected.get(pos) for pos in range(suffix_start_pos, suffix_end_pos + 1)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), 0

    return float(sum(vals) / len(vals)), int(len(vals))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Score ONLY the draft-generated continuation tokens using the DNAGPT draft model. "
            "Reads a wide CSV (must include sample_prompt_ids + sample_draft_new_ids) and writes an augmented CSV."
        )
    )
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)

    p.add_argument("--draft_model_name", default="dna_gpt0.1b_m")
    p.add_argument("--draft_weight_path", default="DNAGPT/checkpoints/dna_gpt0.1b_m.pth")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")

    args = p.parse_args(argv)

    device, dtype = _resolve_device_dtype(args.device, args.dtype)

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input CSV not found: {args.input_csv}")
    if not os.path.exists(args.draft_weight_path):
        raise SystemExit(f"Draft weight_path not found: {args.draft_weight_path}")

    model, _tok = dnagpt_get_model(args.draft_model_name)
    model = dnagpt_load_model(model, args.draft_weight_path, device=device, dtype=dtype)
    model.eval()

    new_cols = [
        "target_suffix_logprob_mean__draft_model",
        "target_suffix_nll_mean__draft_model",
        "target_suffix_ppl__draft_model",
        "specdec_suffix_logprob_mean__draft_model",
        "specdec_suffix_nll_mean__draft_model",
        "specdec_suffix_ppl__draft_model",
        "draft_suffix_logprob_mean__draft_model",
        "draft_suffix_nll_mean__draft_model",
        "draft_suffix_ppl__draft_model",
        "draft_scoring_model_name",
        "draft_scoring_weight_path",
    ]

    with open(args.input_csv, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV appears to have no header")

        required = [
            "sample_prompt_ids",
            "sample_target_new_ids",
            "sample_specdec_new_ids",
            "sample_draft_new_ids",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                "Missing required columns: " + ", ".join(missing) + ". "
                "This script expects a wide CSV with token id columns."
            )

        out_fieldnames = list(reader.fieldnames)
        for c in new_cols:
            if c not in out_fieldnames:
                out_fieldnames.append(c)

        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()

            for i, row in enumerate(reader):
                prompt_ids = _parse_list_cell(row.get("sample_prompt_ids"))
                prompt_len = len(prompt_ids)

                for prefix, new_col in (
                    ("target", "sample_target_new_ids"),
                    ("specdec", "sample_specdec_new_ids"),
                    ("draft", "sample_draft_new_ids"),
                ):
                    new_ids = _parse_list_cell(row.get(new_col))
                    full_ids = prompt_ids + new_ids

                    # DNAGPT has a fixed positional length (usually 512). Some rows can exceed it
                    # (e.g. 256 prompt + 257 continuation). Use sliding-window scoring in that case.
                    max_len = _resolve_dnagpt_model_max_len(model)
                    if len(full_ids) > max_len:
                        mean_lp, cont_len = _score_continuation_mean_logprob_sliding(
                            full_ids=full_ids,
                            prompt_len=prompt_len,
                            model=model,
                            device=device,
                        )
                    else:
                        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
                        mean_lp, cont_len = _score_continuation_mean_logprob(model, input_ids, prompt_len=prompt_len)

                    row[f"{prefix}_suffix_logprob_mean__draft_model"] = mean_lp
                    row[f"{prefix}_suffix_nll_mean__draft_model"] = (float("nan") if math.isnan(mean_lp) else -mean_lp)
                    if math.isnan(mean_lp) or cont_len <= 0:
                        row[f"{prefix}_suffix_ppl__draft_model"] = float("nan")
                    else:
                        row[f"{prefix}_suffix_ppl__draft_model"] = float(math.exp(-mean_lp))

                row["draft_scoring_model_name"] = args.draft_model_name
                row["draft_scoring_weight_path"] = args.draft_weight_path

                writer.writerow(row)

                if (i + 1) % 50 == 0:
                    print(f"[INFO] scored {i+1} rows")


if __name__ == "__main__":
    main()
