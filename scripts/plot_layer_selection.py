#!/usr/bin/env python3
"""
Plot layer-selection ablation results for reviewer comment 2.a.1.

Generates two publication-ready combined figures (matching the style of
results/speedup_acceptance_plots):

  Figure 1 – "Average across layer counts":
      Two-panel bar chart (ProtGPT2 | ProGen2).  Each panel has three
      bars (First-N / Last-N / Mixed) showing acceptance rate averaged
      across all n_draft_layers.  Error bars = ±1 std.

  Figure 2 – "By number of layers":
      Two-panel line plot (ProtGPT2 | ProGen2).  x = n_draft_layers,
      hue = strategy, with coloured ±1σ bands identical to the
      speedup_acceptance line plots.

Usage:
  python scripts/plot_layer_selection.py
  python scripts/plot_layer_selection.py --input_dir results/layer_selection \
      --output_dir results/layer_selection_plots
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style constants (matching plot_speedup_acceptance.py exactly) ─────
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 11
DPI = 300

# Strategy colours — first 3 entries of tab10 (same as HUE_COLORS in
# plot_speedup_acceptance.py)
STRATEGY_COLORS = {
    "first": "#1f77b4",
    "last":  "#ff7f0e",
    "mixed": "#2ca02c",
}
STRATEGY_LABELS = {
    "first": "First-N Layers",
    "last":  "Last-N Layers",
    "mixed": "Mixed Layers",
}
STRATEGY_ORDER = ["first", "last", "mixed"]

MODEL_TITLES = {"protgpt2": "ProtGPT2", "progen2": "ProGen2"}

METRICS = {
    "acceptance_rate": "Acceptance Rate (α)",
    "kl_div_mean":     "KL Divergence",
    "top1_agreement":  "Top-1 Agreement",
    "mean_prob_ratio":  "Mean Probability Ratio",
}


# ── helpers ────────────────────────────────────────────────────────────
def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def load_data(input_dir: str) -> dict[str, pd.DataFrame]:
    dfs = {}
    for key, fname in [("protgpt2", "protgpt2_layer_selection.csv"),
                        ("progen2", "progen2_layer_selection.csv")]:
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            print(f"[WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        for c in ["n_draft_layers", "acceptance_rate", "kl_div_mean",
                   "top1_agreement", "mean_prob_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs[key] = df

    return dfs


# ── Figure 1: average across all layer counts (bar chart) ─────────────
def plot_avg_bars(dfs: dict, output_dir: str, metric: str = "acceptance_rate"):
    """Side-by-side bar chart (one panel per model), 3 bars per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    y_label = METRICS.get(metric, metric)

    for idx, (model_key, ax) in enumerate(zip(["protgpt2", "progen2"], axes)):
        if model_key not in dfs:
            ax.set_visible(False)
            continue
        df = dfs[model_key]

        # Per (n_draft_layers, strategy) mean → then across layers mean±std
        per_layer = (df.groupby(["n_draft_layers", "strategy"])[metric]
                       .mean().reset_index())
        strat_stats = (per_layer.groupby("strategy")[metric]
                         .agg(["mean", "std"]).reindex(STRATEGY_ORDER))

        x = np.arange(len(STRATEGY_ORDER))
        colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
        labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
        means  = strat_stats["mean"].values
        stds   = strat_stats["std"].fillna(0).values
        # Clip lower whisker so bars never go below 0
        lower = np.minimum(stds, means)
        upper = stds

        ax.bar(x, means, 0.55, yerr=[lower, upper], color=colors,
               capsize=7, edgecolor="none",
               error_kw={"elinewidth": 1.5, "capthick": 1.2,
                          "ecolor": "#444444"})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
        ax.set_title(MODEL_TITLES.get(model_key, model_key),
                     fontsize=TITLE_FONTSIZE - 2, pad=8)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle("Layer Selection Strategy",
                 fontsize=TITLE_FONTSIZE, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"layer_selection_avg_{metric}.png"))


# ── Figure 2: by number of layers (line plot with std bands) ──────────
def plot_by_layers(dfs: dict, output_dir: str, metric: str = "acceptance_rate"):
    """Side-by-side line plots.  x = n_draft_layers, hue = strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    y_label = METRICS.get(metric, metric)

    for idx, (model_key, ax) in enumerate(zip(["protgpt2", "progen2"], axes)):
        if model_key not in dfs:
            ax.set_visible(False)
            continue
        df = dfs[model_key]

        grp = (df.groupby(["n_draft_layers", "strategy"])[metric]
                 .agg(["mean", "std"]).reset_index())

        for strat in STRATEGY_ORDER:
            sub = grp[grp["strategy"] == strat].sort_values("n_draft_layers")
            color = STRATEGY_COLORS[strat]
            label = STRATEGY_LABELS[strat]
            xv = sub["n_draft_layers"].values
            yv = sub["mean"].values
            yerr = sub["std"].fillna(0).values

            ax.plot(xv, yv, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(xv, np.maximum(yv - yerr, 0), yv + yerr, alpha=0.15, color=color)

        ax.set_xlabel("Number of Draft Layers", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(MODEL_TITLES.get(model_key, model_key),
                     fontsize=TITLE_FONTSIZE - 2, pad=8)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 1:
            ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    fig.suptitle("Layer Selection Strategy — By Number of Draft Layers",
                 fontsize=TITLE_FONTSIZE, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"layer_selection_by_layers_{metric}.png"))


SWEEP_COLORS = {
    "first":      "#1f77b4",
    "last":       "#ff7f0e",
    "mixed":      "#2ca02c",
    "mixed_alt5": "#d62728",
    "step4":      "#9467bd",
    "step8":      "#8c564b",
}
SWEEP_LABELS = {
    "first":      "First-N",
    "last":       "Last-N",
    "mixed":      "Mixed (evenly-spaced)",
    "mixed_alt5": "Mixed-alt5 [0,6,12,19,25]",
    "step4":      "Step-4 multiples",
    "step8":      "Step-8 multiples",
}


# ── Figure 3: Mixed sweep comparison (ProGen2 only) ──────────────────
def plot_sweep(input_dir: str, output_dir: str, metric: str = "acceptance_rate"):
    """Bar chart comparing all mixed strategies + original first/last for ProGen2."""
    sweep_path = os.path.join(input_dir, "progen2_mixed_sweep.csv")
    orig_path = os.path.join(input_dir, "progen2_layer_selection.csv")
    if not os.path.isfile(sweep_path):
        print(f"[SKIP] sweep data not found: {sweep_path}")
        return
    if not os.path.isfile(orig_path):
        print(f"[SKIP] original data not found: {orig_path}")
        return

    sweep = pd.read_csv(sweep_path)
    orig = pd.read_csv(orig_path)
    for df in [sweep, orig]:
        for c in ["n_draft_layers", metric]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Combine: original first/last/mixed + sweep strategies
    combined = pd.concat([orig, sweep], ignore_index=True)
    y_label = METRICS.get(metric, metric)

    # Group by (strategy, n_draft_layers, layer_indices) → mean ± std
    grp = (combined.groupby(["strategy", "n_draft_layers", "layer_indices"])[metric]
                   .agg(["mean", "std"]).reset_index())
    grp["std"] = grp["std"].fillna(0)

    # Sort strategies in a nice order
    strat_order = ["first", "last", "mixed", "mixed_alt5", "step4", "step8"]
    grp["strat_rank"] = grp["strategy"].map({s: i for i, s in enumerate(strat_order)})
    grp = grp.sort_values(["n_draft_layers", "strat_rank"])

    # Plot: one group per n_draft_layers, bars = strategies
    layer_counts = sorted(grp["n_draft_layers"].unique())
    fig, ax = plt.subplots(figsize=(max(14, len(grp) * 0.7), 6))

    bar_width = 0.7
    x_pos = 0
    x_ticks = []
    x_labels = []
    group_starts = []

    for li, nl in enumerate(layer_counts):
        sub = grp[grp["n_draft_layers"] == nl].sort_values("strat_rank")
        if li > 0:
            x_pos += 0.8  # gap between groups
        group_starts.append(x_pos)
        for _, row in sub.iterrows():
            strat = row["strategy"]
            color = SWEEP_COLORS.get(strat, "#999999")
            mean_val = row["mean"]
            std_val = row["std"]
            lower = min(std_val, mean_val)

            ax.bar(x_pos, mean_val, bar_width, yerr=[[lower], [std_val]],
                   color=color, capsize=5, edgecolor="none",
                   error_kw={"elinewidth": 1.2, "capthick": 1.0, "ecolor": "#444444"})

            # Annotate with layer indices (compact)
            idx_str = row["layer_indices"]
            ax.text(x_pos, mean_val + std_val + 0.01, idx_str,
                    ha="center", va="bottom", fontsize=6, rotation=45)

            x_ticks.append(x_pos)
            x_labels.append(f"{SWEEP_LABELS.get(strat, strat)}\n({nl}L)")
            x_pos += 1.0

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("ProGen2: Mixed Strategy Sweep — All Configurations",
                 fontsize=TITLE_FONTSIZE - 2, pad=10)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    handles = []
    for s in strat_order:
        if s in grp["strategy"].values:
            handles.append(plt.Rectangle((0, 0), 1, 1,
                           fc=SWEEP_COLORS.get(s, "#999"), label=SWEEP_LABELS.get(s, s)))
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE - 1, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"progen2_mixed_sweep_{metric}.png"))


# ── Figure 4: Sweep as line plot by n_draft_layers ────────────────────
def plot_sweep_lines(input_dir: str, output_dir: str, metric: str = "acceptance_rate"):
    """Line plot: x = n_draft_layers, hue = strategy (all including sweep)."""
    sweep_path = os.path.join(input_dir, "progen2_mixed_sweep.csv")
    orig_path = os.path.join(input_dir, "progen2_layer_selection.csv")
    if not os.path.isfile(sweep_path) or not os.path.isfile(orig_path):
        return

    sweep = pd.read_csv(sweep_path)
    orig = pd.read_csv(orig_path)
    combined = pd.concat([orig, sweep], ignore_index=True)
    for c in ["n_draft_layers", metric]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    y_label = METRICS.get(metric, metric)

    grp = (combined.groupby(["strategy", "n_draft_layers"])[metric]
                   .agg(["mean", "std"]).reset_index())
    grp["std"] = grp["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    strat_order = ["first", "last", "mixed", "mixed_alt5", "step4", "step8"]

    for strat in strat_order:
        sub = grp[grp["strategy"] == strat].sort_values("n_draft_layers")
        if len(sub) == 0:
            continue
        color = SWEEP_COLORS.get(strat, "#999")
        label = SWEEP_LABELS.get(strat, strat)
        xv = sub["n_draft_layers"].values
        yv = sub["mean"].values
        yerr = sub["std"].values

        marker = "o" if strat in ("first", "last", "mixed") else "s"
        ls = "-" if strat in ("first", "last", "mixed") else "--"
        ax.plot(xv, yv, marker=marker, linewidth=1.8, color=color, label=label,
                linestyle=ls)
        ax.fill_between(xv, np.maximum(yv - yerr, 0), yv + yerr,
                        alpha=0.12, color=color)

    ax.set_xlabel("Number of Draft Layers", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("ProGen2: Strategy Comparison by Layer Count",
                 fontsize=TITLE_FONTSIZE - 2, pad=8)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, framealpha=0.9)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"progen2_sweep_lines_{metric}.png"))


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot layer-selection ablation results (reviewer 2.a.1)")
    parser.add_argument("--input_dir", default="results/layer_selection",
                        help="Directory containing experiment CSVs")
    parser.add_argument("--output_dir", default="results/layer_selection_plots",
                        help="Directory for output plots")
    args = parser.parse_args()

    dfs = load_data(args.input_dir)
    if not dfs:
        print("No data found. Exiting.")
        sys.exit(1)

    for metric in METRICS:
        plot_avg_bars(dfs, args.output_dir, metric)
        plot_by_layers(dfs, args.output_dir, metric)
        # Sweep plots (only generated if sweep CSV exists)
        plot_sweep(args.input_dir, args.output_dir, metric)
        plot_sweep_lines(args.input_dir, args.output_dir, metric)

    print(f"\nDone – plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
