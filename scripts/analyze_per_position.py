#!/usr/bin/env python3
"""
Reviewer 2.a.3 — Per-position acceptance analysis & plots.

Reads CSVs from collect_per_position.py and produces:
  1. Acceptance rate vs. position in generated sequence
  2. Acceptance rate by position within speculation block (pos_in_block)
  3. p_t vs p_d scatter colored by accept/reject
  4. Acceptance rate over generation (rolling window)
  5. Per-prompt position-acceptance heatmap

Usage:
  python scripts/analyze_per_position.py [--input results/per_position_*.csv]
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT, "analysis_final_plots_v2", "per_position")


def load_all(input_paths):
    dfs = []
    for p in input_paths:
        if os.path.isfile(p):
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  Loaded {p}: {len(df)} rows")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_acceptance_vs_seq_pos(df, out_dir):
    """Acceptance rate vs. position in generated sequence."""
    for model, mdf in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 4))
        by_pos = mdf.groupby("seq_pos")["accepted"].agg(["mean", "count"])
        # Only plot positions with enough data
        by_pos = by_pos[by_pos["count"] >= 3]
        ax.plot(by_pos.index, by_pos["mean"], marker=".", markersize=3,
                linewidth=0.8, alpha=0.7)
        # Rolling average
        if len(by_pos) > 10:
            window = max(5, len(by_pos) // 20)
            rolling = by_pos["mean"].rolling(window, center=True, min_periods=1).mean()
            ax.plot(by_pos.index, rolling, color="red", linewidth=2, label=f"rolling(w={window})")
            ax.legend()
        ax.set_xlabel("Position in generated sequence")
        ax.set_ylabel("Acceptance rate")
        ax.set_title(f"{model} — Acceptance vs. Generated Position")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"accept_vs_seq_pos_{model}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  -> accept_vs_seq_pos_{model}")


def plot_acceptance_by_block_pos(df, out_dir):
    """Acceptance rate by position within a speculation block."""
    for model, mdf in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(5, 4))
        by_bp = mdf.groupby("pos_in_block")["accepted"].agg(["mean", "sem", "count"]).reset_index()
        ax.bar(by_bp["pos_in_block"], by_bp["mean"], yerr=by_bp["sem"],
               capsize=3, color="steelblue", edgecolor="k", linewidth=0.4, alpha=0.8)
        for _, row in by_bp.iterrows():
            ax.text(row["pos_in_block"], row["mean"] + row["sem"] + 0.02,
                    f"n={int(row['count'])}", ha="center", fontsize=8)
        ax.set_xlabel("Position within speculation block (0-indexed)")
        ax.set_ylabel("Acceptance rate")
        ax.set_title(f"{model} — Acceptance by Block Position")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"accept_by_block_pos_{model}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  -> accept_by_block_pos_{model}")


def plot_pt_vs_pd(df, out_dir):
    """Scatter of p_t vs p_d colored by accept/reject."""
    for model, mdf in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(5, 5))
        accepted = mdf[mdf["accepted"] == True]
        rejected = mdf[mdf["accepted"] == False]
        ax.scatter(rejected["p_d"], rejected["p_t"], alpha=0.2, s=5,
                   color="red", label=f"Rejected ({len(rejected)})")
        ax.scatter(accepted["p_d"], accepted["p_t"], alpha=0.2, s=5,
                   color="blue", label=f"Accepted ({len(accepted)})")
        lims = [0, max(mdf["p_d"].quantile(0.99), mdf["p_t"].quantile(0.99))]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="p_t = p_d")
        ax.set_xlabel("Draft probability (p_d)")
        ax.set_ylabel("Target probability (p_t)")
        ax.set_title(f"{model} — Target vs. Draft Token Probability")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, lims[1])
        ax.set_ylim(0, lims[1])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pt_vs_pd_{model}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  -> pt_vs_pd_{model}")


def plot_rolling_acceptance(df, out_dir):
    """Rolling acceptance rate over the course of generation (per prompt)."""
    for model, mdf in df.groupby("model"):
        prompts = sorted(mdf["prompt_idx"].unique())
        fig, ax = plt.subplots(figsize=(8, 4))
        for pi in prompts:
            sub = mdf[mdf["prompt_idx"] == pi].sort_values("seq_pos")
            if len(sub) < 10:
                continue
            window = max(5, len(sub) // 20)
            rolling = sub["accepted"].astype(float).rolling(window, min_periods=1).mean()
            ax.plot(sub["seq_pos"].values, rolling.values, alpha=0.6,
                    linewidth=1.2, label=f"P{pi}")
        ax.set_xlabel("Position in generated sequence")
        ax.set_ylabel("Rolling acceptance rate")
        ax.set_title(f"{model} — Rolling Acceptance Over Generation")
        ax.set_ylim(0, 1)
        if len(prompts) <= 10:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rolling_accept_{model}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  -> rolling_accept_{model}")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("PER-POSITION ACCEPTANCE SUMMARY")
    print("=" * 60)
    for model, mdf in df.groupby("model"):
        print(f"\n--- {model} ---")
        print(f"  Total decisions: {len(mdf)}")
        print(f"  Overall acceptance: {mdf['accepted'].mean():.3f}")
        print(f"\n  By position in block:")
        for bp, g in mdf.groupby("pos_in_block"):
            print(f"    pos {bp}: {g['accepted'].mean():.3f} (n={len(g)})")
        print(f"\n  First 5 vs Last 5 generated positions:")
        early = mdf[mdf["seq_pos"] < 5]["accepted"].mean()
        late = mdf[mdf["seq_pos"] >= mdf["seq_pos"].quantile(0.9)]["accepted"].mean()
        print(f"    Early (pos<5): {early:.3f}")
        print(f"    Late (top 10%): {late:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", default=None,
                        help="Per-position CSV files (default: results/per_position_*.csv)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.input is None:
        args.input = glob.glob(os.path.join(ROOT, "results", "per_position_*.csv"))
    if args.output_dir is None:
        args.output_dir = OUT_DIR

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading per-position data...")
    df = load_all(args.input)
    if df.empty:
        print("No data found. Run collect_per_position.py first.")
        return

    print(f"\nTotal records: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")

    plot_acceptance_vs_seq_pos(df, args.output_dir)
    plot_acceptance_by_block_pos(df, args.output_dir)
    plot_pt_vs_pd(df, args.output_dir)
    plot_rolling_acceptance(df, args.output_dir)
    print_summary(df)

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
