import argparse
import json
import os
import sys
import math

import torch
import torch.nn.functional as F
import pandas as pd


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_REPO_DIR, "DNAGPT"))
from scoring_specdec_beam_search import get_model, load_model  # reuse helpers


def _looks_like_dnagpt_model_family(x: str) -> bool:
    v = (x or "").strip().lower()
    return v.startswith("dna") or ("dnagpt" in v)


def _guard_against_wrong_csv(df: pd.DataFrame, input_csv: str) -> None:
    if "draft_mode" in df.columns or "draft_model_effective" in df.columns:
        raise ValueError(
            "This looks like a protein wide CSV (has 'draft_mode' / draft metadata columns). "
            "Use 'python -m pipeline.score_likelihoods' for protein models, not likelihood_scoring.py. "
            f"input_csv={input_csv}"
        )

    if "model_family" in df.columns:
        vals = [str(v) for v in df["model_family"].dropna().unique().tolist()]
        if vals and any(not _looks_like_dnagpt_model_family(v) for v in vals):
            raise ValueError(
                "This CSV contains non-DNAGPT model_family values "
                f"({vals}). likelihood_scoring.py only supports DNAGPT. "
                "Use 'python -m pipeline.score_likelihoods' for protein sweeps. "
                f"input_csv={input_csv}"
            )


def _validate_ids_in_vocab(cleaned: list[tuple[int, list[int], list[int]]], vocab_size: int, input_csv: str) -> None:
    if vocab_size <= 0:
        raise ValueError(f"Invalid DNAGPT vocab_size={vocab_size}")

    min_id: int | None = None
    max_id: int | None = None
    first_bad: tuple[int, int] | None = None  # (row_idx, bad_id)

    for row_idx, p, n in cleaned:
        for tid in p:
            if min_id is None or tid < min_id:
                min_id = tid
            if max_id is None or tid > max_id:
                max_id = tid
            if tid < 0 or tid >= vocab_size:
                first_bad = (row_idx, tid)
                break
        if first_bad is not None:
            break
        for tid in n:
            if min_id is None or tid < min_id:
                min_id = tid
            if max_id is None or tid > max_id:
                max_id = tid
            if tid < 0 or tid >= vocab_size:
                first_bad = (row_idx, tid)
                break
        if first_bad is not None:
            break

    if first_bad is not None:
        row_idx, bad_id = first_bad
        raise ValueError(
            "Token ID out of range for DNAGPT vocabulary. "
            f"vocab_size={vocab_size}, bad_id={bad_id}, row_index={row_idx}, observed_min={min_id}, observed_max={max_id}. "
            "This usually means you're trying to score a protein CSV (ProtGPT2/ProGen2 token IDs) with the DNAGPT scorer. "
            "Use 'python -m pipeline.score_likelihoods' for protein wide CSVs. "
            f"input_csv={input_csv}"
        )


def _resolve_dnagpt_model_max_len(model) -> int:
    # DNAGPT's GPT base class exposes max_len, and also has transformer.wpe sized to max_len.
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
    raise ValueError("Could not resolve DNAGPT model max_len (positional embedding length).")


def _resolve_dnagpt_model_vocab_size(model) -> int:
    # Prefer the actual output logits dim (lm_head / mlm_head), not tokenizer length.
    try:
        v = getattr(model, "vocab_size", None)
        if isinstance(v, int) and v > 0:
            return v
    except Exception:
        pass
    try:
        wte = getattr(getattr(model, "transformer", None), "wte", None)
        num = getattr(getattr(wte, "weight", None), "shape", [None])[0]
        if isinstance(num, int) and num > 0:
            return num
    except Exception:
        pass
    # Fallback: run a tiny forward on CPU/device would be expensive here; callers can provide.
    raise ValueError("Could not resolve DNAGPT model vocab size.")


def _score_suffix_logprobs_sliding_window(
    full_ids: list[int],
    prefix_len: int,
    model,
    device: str,
    reduce: str,
) -> float:
    """Score suffix tokens using a sliding-window context.

    DNAGPT models have a fixed positional embedding length (`model.max_len`, 512 for dna_gpt3b_m).
    Passing a longer sequence directly causes an index-out-of-bounds in the positional embedding.
    This function computes per-token logprobs with a bounded context, matching generation-time
    cropping semantics (context limited to the last `max_len` tokens).
    """
    if prefix_len < 1:
        return float("nan")

    max_len = _resolve_dnagpt_model_max_len(model)
    T = len(full_ids)
    if T <= 1:
        return float("nan")

    # We need logprob for tokens at global positions j in [prefix_len .. T-1]
    # where logprob(full[j]) is predicted from context full[:j].
    suffix_start_pos = prefix_len
    suffix_end_pos = T - 1
    if suffix_end_pos < suffix_start_pos:
        return float("nan")

    # Sliding-window stride: smaller stride increases compute but gives the same result.
    # Use a reasonably large stride to keep scoring fast.
    stride = max(1, max_len // 2)

    collected: dict[int, float] = {}

    # Iterate end positions so that each window provides up to `max_len` tokens of context.
    # We score tokens in the tail of each window to avoid duplicate work.
    # Based on the standard HF perplexity sliding-window method.
    for i in range(0, T - 1, stride):
        end_loc = min(i + stride + 1, T)
        begin_loc = max(0, end_loc - max_len)
        seg = full_ids[begin_loc:end_loc]
        if len(seg) < 2:
            continue

        input_ids = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
        with torch.inference_mode():
            logits = model(input_ids)  # [1, L, V]

        # log p(token_t | context) from logits at position t-1
        logprobs_full = F.log_softmax(logits.float(), dim=-1)  # [1, L, V]
        logprobs_full = logprobs_full[:, :-1]  # [1, L-1, V]
        targets = input_ids[:, 1:]  # [1, L-1]
        token_logprobs = torch.gather(logprobs_full, 2, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

        # Decide which global positions this segment should contribute (avoid duplicates).
        # Segment predicts tokens at global positions [begin_loc+1 .. end_loc-1].
        seg_pred_start = begin_loc + 1
        seg_pred_end = end_loc - 1

        # Only take tokens at global positions >= i+1 (the new region for this step).
        take_start = max(i + 1, seg_pred_start)
        if take_start > seg_pred_end:
            continue
        local_start = take_start - seg_pred_start
        local_len = seg_pred_end - take_start + 1

        vals = token_logprobs[0, local_start:local_start + local_len].detach().cpu().tolist()
        for offset, lp in enumerate(vals):
            collected[take_start + offset] = float(lp)

    # Aggregate suffix logprobs.
    suffix_vals = [collected.get(pos) for pos in range(suffix_start_pos, suffix_end_pos + 1)]
    suffix_vals = [v for v in suffix_vals if v is not None]
    if not suffix_vals:
        return float("nan")

    if reduce == "mean":
        return float(sum(suffix_vals) / len(suffix_vals))
    if reduce == "sum":
        return float(sum(suffix_vals))
    raise ValueError(f"Unknown reduce={reduce}")


def score_dnagpt_sequences(
    seqs,
    model,
    tokenizer,
    device="cuda",
    reduce="mean",
):
    """
    seqs: list[str] (any text the tokenizer can handle)
    Returns list[float]: per-sequence mean (or sum) log-prob.
    """
    # Filter empty strings
    cleaned = [(i, s) for i, s in enumerate(seqs) if isinstance(s, str) and len(s) > 0]
    if not cleaned:
        return [float("nan")] * len(seqs)

    idxs, non_empty = zip(*cleaned)

    # Tokenize each sequence
    token_tensors = [tokenizer.encode(s, device=device) for s in non_empty]
    lengths = [t.shape[0] for t in token_tensors]
    max_len = max(lengths)

    pad_id = tokenizer.pad_id
    batch = []
    for t in token_tensors:
        pad = torch.full(
            (max_len - t.shape[0],),
            pad_id,
            device=device,
            dtype=torch.long
        )
        batch.append(torch.cat([t, pad], dim=0).unsqueeze(0))

    input_ids = torch.cat(batch, dim=0)  # [B, T]

    with torch.inference_mode():
        logits = model(input_ids)        # [B, T, V] for DNAGPT

    # log p(token_t | context) – predict position t from context < t
    logprobs_full = F.log_softmax(logits, dim=-1)  # [B, T, V]
    logprobs_full = logprobs_full[:, :-1]          # [B, T-1, V]
    targets = input_ids[:, 1:]                     # [B, T-1]

    logprobs = torch.gather(
        logprobs_full,
        2,
        targets.unsqueeze(-1)
    ).squeeze(-1)                                  # [B, T-1]

    logprobs = logprobs.cpu().numpy()

    if reduce == "mean":
        fn = lambda arr: float(arr.mean())
    elif reduce == "sum":
        fn = lambda arr: float(arr.sum())
    else:
        raise ValueError(f"Unknown reduce={reduce}")

    scores = [float("nan")] * len(seqs)
    for row_pos, (orig_idx, L) in enumerate(zip(idxs, lengths)):
        # We have T-1 logprobs; for a sequence of length L, use first L-1 positions
        if L <= 1:
            scores[orig_idx] = float("nan")
        else:
            scores[orig_idx] = fn(logprobs[row_pos, :L-1])

    return scores


def _maybe_parse_int_list(cell) -> list[int] | None:
    if cell is None:
        return None
    if isinstance(cell, float) and math.isnan(cell):
        return None
    if isinstance(cell, list):
        try:
            return [int(x) for x in cell]
        except Exception:
            return None
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if not isinstance(obj, list):
            return None
        try:
            return [int(x) for x in obj]
        except Exception:
            return None
    return None


def score_dnagpt_suffix_ids(
    prompt_ids_json,
    new_ids_json,
    model,
    tokenizer,
    device="cuda",
    reduce="mean",
):
    """Scores suffix log-prob conditioned on the prompt using token IDs.

    prompt_ids_json: list[str|nan] where each cell is a JSON-encoded list[int]
    new_ids_json: list[str|nan] where each cell is a JSON-encoded list[int]

    Returns list[float] of per-row logprob (mean/sum over suffix tokens).
    """
    assert len(prompt_ids_json) == len(new_ids_json)

    cleaned: list[tuple[int, list[int], list[int]]] = []
    for i, (p_cell, n_cell) in enumerate(zip(prompt_ids_json, new_ids_json)):
        p = _maybe_parse_int_list(p_cell)
        n = _maybe_parse_int_list(n_cell)
        if not p or not n:
            continue
        cleaned.append((i, p, n))

    scores = [float("nan")] * len(prompt_ids_json)
    if not cleaned:
        return scores

    pad_id = tokenizer.pad_id

    # Fail early (on CPU) if IDs don't match the DNAGPT vocab.
    # This prevents CUDA gather asserts and confusing downstream cublas failures.
    vocab_size = _resolve_dnagpt_model_vocab_size(model)
    _validate_ids_in_vocab(cleaned, vocab_size=vocab_size, input_csv=getattr(sys, "_dnagpt_input_csv", ""))

    # Important: DNAGPT has a fixed positional embedding length (e.g., 512 for dna_gpt3b_m).
    # Many of our prompt+suffix sequences can exceed this, which would crash if we batch-pad
    # to the maximum length. Score each row with a sliding window bounded by model.max_len.
    for orig_idx, p, n in cleaned:
        full = p + n
        try:
            scores[orig_idx] = _score_suffix_logprobs_sliding_window(
                full_ids=full,
                prefix_len=len(p),
                model=model,
                device=device,
                reduce=reduce,
            )
        except RuntimeError as e:
            # Improve the common failure mode message.
            raise RuntimeError(
                "DNAGPT scoring failed while computing suffix logprobs from token IDs. "
                "This can happen if the full prompt+suffix length exceeds the model's max_len "
                "and the scorer does not window properly (now fixed), or if token IDs are invalid. "
                f"row_index={orig_idx}, prompt_len={len(p)}, suffix_len={len(n)}, full_len={len(full)}"
            ) from e

    return scores


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline DNAGPT scoring for sequences produced by grid search. "
            "Reads a CSV with columns like 'sample_target_suffix', "
            "'sample_specdec_suffix', 'sample_draft_suffix', computes "
            "mean log-prob and perplexity, and writes an augmented CSV."
        )
    )

    parser.add_argument("input_csv", help="Grid-search CSV to score")
    parser.add_argument("--output_csv", default=None,
                        help="Output CSV (default: <input>_scored.csv)")
    parser.add_argument("--model_name", default="dna_gpt3b_m",
                        help="DNAGPT model name for scoring")
    parser.add_argument("--weight_path", default="DNAGPT/checkpoints/dna_gpt3b_m.pth",
                        help="Path to DNAGPT weights for scoring")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="float16",
                        help="float16, bfloat16, float32, etc.")
    parser.add_argument("--reduce", default="mean", choices=["mean", "sum"],
                        help="Reduce method over time dimension")

    args = parser.parse_args()

    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = base + "_scored.csv"

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print("Columns:", list(df.columns))

    _guard_against_wrong_csv(df, args.input_csv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    print(f"Scoring with model {args.model_name} on {device}, dtype={dtype}")

    # Make input path available to helper functions for better error messages.
    sys._dnagpt_input_csv = args.input_csv  # type: ignore[attr-defined]

    # Load tokenizer + model for scoring
    _, tokenizer = get_model(args.model_name)
    model, _tok_unused = get_model(args.model_name)
    model = load_model(model, args.weight_path, device=device, dtype=dtype)

    # Helper: score one column if present (prefers token IDs if available)
    def score_column(col_name: str, prefix: str, ids_col: str | None = None):
        if col_name not in df.columns:
            print(f"Column '{col_name}' not found, skipping {prefix} scoring.")
            return

        print(f"Scoring {col_name} -> {prefix}_* metrics ...")

        scores = None
        if ids_col and ("sample_prompt_ids" in df.columns) and (ids_col in df.columns):
            print(f"Using token IDs: sample_prompt_ids + {ids_col}")
            scores = score_dnagpt_suffix_ids(
                df["sample_prompt_ids"].tolist(),
                df[ids_col].tolist(),
                model,
                tokenizer,
                device=device,
                reduce=args.reduce,
            )
        else:
            seqs = df[col_name].tolist()
            scores = score_dnagpt_sequences(
                seqs,
                model,
                tokenizer,
                device=device,
                reduce=args.reduce,
            )

        df[f"{prefix}_logprob_{args.reduce}"] = scores
        df[f"{prefix}_nll_{args.reduce}"] = [
            -s if not math.isnan(s) else float("nan") for s in scores
        ]
        df[f"{prefix}_ppl"] = [
            math.exp(-s) if not math.isnan(s) else float("nan")
            for s in scores
        ]

    # Score target suffix (prefer ids if available)
    score_column("sample_target_suffix", "target_suffix", ids_col="sample_target_new_ids")

    # Score specdec suffix (prefer ids if available)
    score_column("sample_specdec_suffix", "specdec_suffix", ids_col="sample_specdec_new_ids")

    # Score draft suffix (prefer ids if available)
    score_column("sample_draft_suffix", "draft_suffix", ids_col="sample_draft_new_ids")

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote scored CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
