import argparse
import os
import sys
import math

import torch
import torch.nn.functional as F
import pandas as pd

# DNAGPT imports
sys.path.append(os.path.join(os.getcwd(), "DNAGPT"))
from specdec_beam_search import get_model, load_model  # reuse helpers


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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    print(f"Scoring with model {args.model_name} on {device}, dtype={dtype}")

    # Load tokenizer + model for scoring
    _, tokenizer = get_model(args.model_name)
    model, _tok_unused = get_model(args.model_name)
    model = load_model(model, args.weight_path, device=device, dtype=dtype)

    # Helper: score one column if present
    def score_column(col_name: str, prefix: str):
        if col_name not in df.columns:
            print(f"Column '{col_name}' not found, skipping {prefix} scoring.")
            return

        print(f"Scoring {col_name} -> {prefix}_* metrics ...")
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

    # Score target suffix
    score_column("sample_target_suffix", "target_suffix")

    # Score specdec suffix
    score_column("sample_specdec_suffix", "specdec_suffix")

    # Score draft suffix (if present)
    score_column("sample_draft_suffix", "draft_suffix")

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote scored CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
