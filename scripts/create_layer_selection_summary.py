#!/usr/bin/env python3
"""
Create a summary CSV and publication-quality plot comparing
First-N vs Last-N vs Mixed layer selection strategies for
ProtGPT2 and ProGen2-xlarge.


Outputs:
  results/layer_selection/layer_selection_summary.csv
  results/layer_selection_plots/layer_strategy_comparison.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "layer_selection")
PLOT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "layer_selection_plots")

# ── style ──────────────────────────────────────────────────────────────
AXIS_LABEL_FS = 16
TICK_FS       = 13
TITLE_FS      = 18
LEGEND_FS     = 12
DPI           = 300
STRATEGY_COLORS = {"First-N": "#1f77b4", "Last-N": "#ff7f0e", "Mixed": "#2ca02c"}
STRATEGY_ORDER  = ["First-N", "Last-N", "Mixed"]
LAYER_COUNTS    = [3, 4, 6]


def load_and_combine() -> pd.DataFrame:
    """Load raw measurement data for both models."""
    frames = []

    # ── ProtGPT2 (all clean) ──────────────────────────────────────────
    prot = pd.read_csv(os.path.join(RESULTS_DIR, "protgpt2_layer_selection.csv"))
    prot = prot[prot["n_draft_layers"].isin(LAYER_COUNTS)]
    prot["model_label"] = "ProtGPT2"
    prot["strategy_label"] = prot["strategy"].map(
        {"first": "First-N", "last": "Last-N", "mixed": "Mixed"}
    )
    frames.append(prot)

    # ── ProGen2 ───────────────────────────────────────────────────────
    prog = pd.read_csv(os.path.join(RESULTS_DIR, "progen2_layer_selection.csv"))
    prog = prog[prog["n_draft_layers"].isin(LAYER_COUNTS)]
    prog["model_label"] = "ProGen2-xlarge"
    prog["strategy_label"] = prog["strategy"].map(
        {"first": "First-N", "last": "Last-N", "mixed": "Mixed"}
    )
    frames.append(prog)

    combined = pd.concat(frames, ignore_index=True)
    for c in ["n_draft_layers", "acceptance_rate", "kl_div_mean"]:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce")
    return combined


def make_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (model, strategy, n_draft_layers)."""
    grp = (
        combined.groupby(["model_label", "strategy_label", "n_draft_layers"])
        .agg(
            acc_mean=("acceptance_rate", "mean"),
            acc_std=("acceptance_rate", "std"),
            kl_mean=("kl_div_mean", "mean"),
            kl_std=("kl_div_mean", "std"),
            n_samples=("acceptance_rate", "count"),
        )
        .reset_index()
    )
    grp = grp.sort_values(["model_label", "n_draft_layers", "strategy_label"])
    return grp


def plot_comparison(summary: pd.DataFrame, output_dir: str):
    """Grouped bar chart: one panel per model, bars = strategy, x = layer count."""
    models = ["ProtGPT2", "ProGen2-xlarge"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    bar_width = 0.22
    offsets = {s: (i - 1) * bar_width for i, s in enumerate(STRATEGY_ORDER)}

    for ax_idx, (model, ax) in enumerate(zip(models, axes)):
        sub = summary[summary["model_label"] == model]

        for strat in STRATEGY_ORDER:
            s = sub[sub["strategy_label"] == strat].sort_values("n_draft_layers")
            x = np.arange(len(LAYER_COUNTS))
            vals = np.full(len(LAYER_COUNTS), np.nan)
            errs = np.full(len(LAYER_COUNTS), np.nan)
            for i, nl in enumerate(LAYER_COUNTS):
                row = s[s["n_draft_layers"] == nl]
                if len(row) == 1:
                    vals[i] = row["acc_mean"].values[0]
                    errs[i] = row["acc_std"].values[0]
            # Only plot bars where data exists
            mask = ~np.isnan(vals)
            v = vals[mask]
            e = errs[mask]
            lower = np.minimum(e, v)

            ax.bar(
                x[mask] + offsets[strat], v, bar_width,
                yerr=[lower, e],
                label=strat, color=STRATEGY_COLORS[strat],
                capsize=4, edgecolor="none",
                error_kw={"elinewidth": 1.2, "capthick": 1.0, "ecolor": "#444"}
            )

        ax.set_xticks(np.arange(len(LAYER_COUNTS)))
        ax.set_xticklabels(LAYER_COUNTS, fontsize=TICK_FS)
        ax.set_xlabel("Number of Draft Layers", fontsize=AXIS_LABEL_FS)
        ax.set_title(model, fontsize=TITLE_FS - 2, pad=8)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.set_ylabel("Acceptance Rate (α)", fontsize=AXIS_LABEL_FS)
        if ax_idx == 1:
            ax.legend(fontsize=LEGEND_FS, framealpha=0.9)

    fig.suptitle("Layer Selection Strategy Comparison", fontsize=TITLE_FS, y=1.02)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "layer_strategy_comparison.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved plot: {out_path}")
    plt.close(fig)


def plot_lines(summary: pd.DataFrame, output_dir: str):
    """Line plot with ±1σ bands: one panel per model."""
    models = ["ProtGPT2", "ProGen2-xlarge"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_idx, (model, ax) in enumerate(zip(models, axes)):
        sub = summary[summary["model_label"] == model]

        for strat in STRATEGY_ORDER:
            s = sub[sub["strategy_label"] == strat].sort_values("n_draft_layers")
            xv = s["n_draft_layers"].values
            yv = s["acc_mean"].values
            yerr = s["acc_std"].fillna(0).values
            color = STRATEGY_COLORS[strat]
            ax.plot(xv, yv, marker="o", linewidth=1.8, color=color, label=strat)
            ax.fill_between(xv, np.maximum(yv - yerr, 0), yv + yerr,
                            alpha=0.15, color=color)

        ax.set_xlabel("Number of Draft Layers", fontsize=AXIS_LABEL_FS)
        ax.set_title(model, fontsize=TITLE_FS - 2, pad=8)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ax_idx == 0:
            ax.set_ylabel("Acceptance Rate (α)", fontsize=AXIS_LABEL_FS)
        if ax_idx == 1:
            ax.legend(fontsize=LEGEND_FS, framealpha=0.9)

    fig.suptitle("Layer Selection Strategy — By Number of Draft Layers",
                 fontsize=TITLE_FS, y=1.02)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "layer_strategy_lines.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved plot: {out_path}")
    plt.close(fig)


def main():
    combined = load_and_combine()
    summary = make_summary(combined)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "layer_selection_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved CSV: {csv_path}")
    print()
    print(summary.to_string(index=False))

    # Plots
    plot_comparison(summary, PLOT_DIR)
    plot_lines(summary, PLOT_DIR)


if __name__ == "__main__":
    main()
