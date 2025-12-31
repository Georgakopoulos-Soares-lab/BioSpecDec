#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


HYPERPARAM_GROUP_COLS_PRIORITY = [
    # Identifiers / model choices (protein)
    "model_family",
    "target_model_name",
    "draft_model_name",
    "draft_mode",
    "draft_model_effective",
    "draft_num_layers_effective",
    "draft_layer_indices",
    # Shared hyperparameters
    "target_context_len",
    "draft_context_len",
    "prefix_len_tokens",
    "L",
    "accept_mode",
    "temperature",
    "top_k",
    "top_p",
]

# Columns that are per-prompt / per-example; do NOT group by these.
PROMPT_LIKE_COLS = {
    "prompt_idx",
    "hg_row_index",
    "hg_id",
    "chrom",
    "start",
    "end",
    "prompt_len_bases",
    "prompt_len_tokens",
}

# Big sample columns we never want to aggregate.
SAMPLE_LIKE_PREFIXES = (
    "sample_",
)

# Core speed metrics (wide schema)
CORE_METRIC_COLS = [
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
]


@dataclass(frozen=True)
class FileStatsConfig:
    input_csv: str
    output_dir: str


def _is_sample_like(col: str) -> bool:
    return any(col.startswith(p) for p in SAMPLE_LIKE_PREFIXES)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _metric_candidates(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    for c in CORE_METRIC_COLS:
        if c in df.columns:
            cols.append(c)

    # If a scored CSV is provided, include common likelihood columns.
    for c in df.columns:
        if c.endswith("_ppl"):
            cols.append(c)
        if c.endswith("_logprob_mean") or c.endswith("_logprob_sum"):
            cols.append(c)
        if c.endswith("_nll_mean") or c.endswith("_nll_sum"):
            cols.append(c)

    # De-dup while preserving order.
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _group_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    for c in HYPERPARAM_GROUP_COLS_PRIORITY:
        if c in df.columns and c not in PROMPT_LIKE_COLS and not _is_sample_like(c):
            cols.append(c)

    # Fall back: if some hyperparams weren't present, still try a minimal grouping.
    if not cols:
        for c in ("model_family", "L", "temperature", "top_k", "top_p", "accept_mode"):
            if c in df.columns:
                cols.append(c)

    # DNAGPT sweeps often vary max prompt length; the wide CSV records this as
    # prompt_len_tokens. Although it's "prompt-like" in general, in these sweeps
    # it usually takes only a few discrete values (e.g., 32/64/128/256) and is
    # useful as a grouping axis for plots.
    if "prompt_len_tokens" in df.columns and "prompt_len_tokens" not in cols:
        try:
            n = int(df["prompt_len_tokens"].nunique(dropna=False))
        except Exception:
            n = 0
        if 1 < n <= 32:
            cols.append("prompt_len_tokens")

    return cols


def _nunique_prompt(df: pd.DataFrame) -> pd.Series | None:
    if "prompt_idx" in df.columns:
        return df["prompt_idx"].nunique(dropna=False)
    if "hg_id" in df.columns:
        return df["hg_id"].nunique(dropna=False)
    return None


def compute_grouped_stats(cfg: FileStatsConfig) -> str:
    df = pd.read_csv(cfg.input_csv)

    group_cols = _group_cols(df)
    if not group_cols:
        raise ValueError(f"No grouping columns found for {cfg.input_csv}")

    metrics = _metric_candidates(df)
    if not metrics:
        raise ValueError(f"No metric columns found for {cfg.input_csv}")

    # Coerce metric columns to numeric (strings/empties become NaN)
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop sample-like columns so we don't accidentally carry huge blobs.
    drop_cols = [c for c in df.columns if _is_sample_like(c)]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    agg_spec = {"n_rows": (metrics[0], "size")}

    # Count prompts per group if possible.
    if "prompt_idx" in df.columns:
        agg_spec["n_prompts"] = ("prompt_idx", pd.Series.nunique)
    elif "hg_id" in df.columns:
        agg_spec["n_prompts"] = ("hg_id", pd.Series.nunique)

    for m in metrics:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")

    grouped = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()

    _ensure_dir(cfg.output_dir)
    base = os.path.splitext(os.path.basename(cfg.input_csv))[0]
    out_csv = os.path.join(cfg.output_dir, f"{base}_grouped_stats.csv")
    grouped.to_csv(out_csv, index=False)

    meta = {
        "input_csv": cfg.input_csv,
        "output_csv": out_csv,
        "rows_in": int(len(df)),
        "groups_out": int(len(grouped)),
        "group_cols": group_cols,
        "metric_cols": metrics,
    }
    with open(os.path.join(cfg.output_dir, f"{base}_grouped_stats.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_csv


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compute grouped summary statistics for wide CSV sweep outputs (DNAGPT + protein). "
            "Groups by sweep hyperparameters (gamma/L, temperature, top_k/top_p, context lens, draft mode/layers if present) "
            "and emits aggregated mean/std metrics."
        )
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more wide CSV files (optionally scored).",
    )
    p.add_argument(
        "--output_dir",
        default="results/statistics",
        help="Directory to write grouped stats outputs.",
    )

    args = p.parse_args(argv)

    for path in args.inputs:
        out = compute_grouped_stats(FileStatsConfig(input_csv=path, output_dir=args.output_dir))
        print(f"Wrote grouped stats: {out}")


if __name__ == "__main__":
    main()
