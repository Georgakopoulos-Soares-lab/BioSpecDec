#!/usr/bin/env python3
"""
Reviewer comment 2.a.2: Comprehensive speedup + acceptance rate plots.

Generates per-model figures showing wall-clock speedup, acceptance rate,
and accepted prefix length (ideal speedup) vs draft length (L / gamma),
with hue on draft_num_layers_effective where applicable.

Also produces a combined summary table (LaTeX-friendly CSV) per model.

Usage:
  python scripts/plot_speedup_acceptance.py

Reads:
  results/dnagpt_final_scored_filtered.csv
  results/progen2_final_final_scored.csv
  results/protgpt2_wide_scored.csv

Writes to results/speedup_acceptance_plots/
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style constants (matching plot_grouped_statistics.py) ──
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 11

# Colors for hue levels (draft layer counts)
HUE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Colors for the 3-bar method comparison
METHOD_COLORS = {
    "target": "#c7c7c7",
    "specdec": "#a6cee3",
    "draft": "#b2df8a",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "L", "temperature", "target_tps", "draft_tps", "specdec_tps",
        "speedup_vs_target", "mean_accept_rate", "mean_accepted_prefix",
        "target_tokens_total", "draft_tokens_total", "specdec_tokens_total",
        "target_time_total", "draft_time_total", "specdec_time_total",
        "draft_num_layers_effective",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def infer_model_title(df: pd.DataFrame) -> str:
    label_map = {"protgpt2": "ProtGPT2", "dnagpt": "DNAGPT", "progen2": "ProGen2"}
    if "model_family" in df.columns:
        fam = str(df["model_family"].dropna().mode().iloc[0]).strip().lower()
        return label_map.get(fam, fam)
    return "Model"


# ═══════════════════════════════════════════════════════════════════════
# Grouping helpers
# ═══════════════════════════════════════════════════════════════════════

def _group_cols(df: pd.DataFrame, x_col: str, hue_col: Optional[str]) -> List[str]:
    """Build the list of columns to group by."""
    cols = [x_col]
    if hue_col and hue_col in df.columns:
        cols.append(hue_col)
    return cols


def aggregate_by(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Group by group_cols and compute mean ± std for each metric."""
    agg_dict = {}
    for m in metric_cols:
        if m in df.columns:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")
            agg_dict[f"{m}_count"] = (m, "count")
    return df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()


# ═══════════════════════════════════════════════════════════════════════
# Plotting: line plots with error bands
# ═══════════════════════════════════════════════════════════════════════

def _plot_metric_vs_L(
    agg: pd.DataFrame,
    x_col: str,
    y_mean_col: str,
    y_std_col: str,
    hue_col: Optional[str],
    ax: plt.Axes,
    y_label: str,
    title: str,
    legend: bool = True,
    integer_x: bool = True,
):
    """Line plot of y vs x, optionally split by hue, with error bands."""
    if hue_col and hue_col in agg.columns and agg[hue_col].nunique() > 1:
        hue_vals = sorted(agg[hue_col].dropna().unique())
        for ci, hv in enumerate(hue_vals):
            sub = agg[agg[hue_col] == hv].sort_values(x_col)
            color = HUE_COLORS[ci % len(HUE_COLORS)]
            lbl = _legend_label(hue_col, hv)
            ax.plot(sub[x_col], sub[y_mean_col], marker="o", linewidth=1.8,
                    color=color, label=lbl)
            if y_std_col in sub.columns:
                yerr = sub[y_std_col].fillna(0)
                ax.fill_between(sub[x_col],
                                sub[y_mean_col] - yerr,
                                sub[y_mean_col] + yerr,
                                alpha=0.15, color=color)
        if legend:
            ax.legend(fontsize=LEGEND_FONTSIZE - 1, framealpha=0.9)
    else:
        sub = agg.sort_values(x_col)
        ax.plot(sub[x_col], sub[y_mean_col], marker="o", linewidth=2.0,
                color=HUE_COLORS[0])
        if y_std_col in sub.columns:
            yerr = sub[y_std_col].fillna(0)
            ax.fill_between(sub[x_col],
                            sub[y_mean_col] - yerr,
                            sub[y_mean_col] + yerr,
                            alpha=0.15, color=HUE_COLORS[0])

    ax.set_xlabel(x_col if x_col != "L" else "Draft Length (γ)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE - 2, pad=8)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    if integer_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _legend_label(hue_col: str, val) -> str:
    if hue_col == "draft_num_layers_effective":
        try:
            return f"Draft layers: {int(float(val))}"
        except Exception:
            return f"Draft layers: {val}"
    return f"{hue_col}={val}"


# ═══════════════════════════════════════════════════════════════════════
# Main per-model plotting
# ═══════════════════════════════════════════════════════════════════════

def _modal_filter(df: pd.DataFrame, keep_cols: Iterable[str]) -> pd.DataFrame:
    """Fix nuisance variables to their modal value to avoid cluttered plots."""
    keep = set(keep_cols)
    nuisance = [
        "model_family", "target_model_name", "draft_model_name",
        "draft_mode", "accept_mode", "temperature", "top_k", "top_p",
        "target_context_len", "draft_context_len", "prefix_len_tokens",
    ]
    out = df
    for c in nuisance:
        if c in keep or c not in out.columns:
            continue
        if out[c].nunique(dropna=False) <= 1:
            continue
        mode = out[c].mode(dropna=False)
        if mode.empty:
            continue
        out = out[out[c] == mode.iloc[0]]
    return out


def plot_model(
    df: pd.DataFrame,
    model_title: str,
    output_dir: str,
    x_col: str = "L",
    hue_col: Optional[str] = None,
) -> List[str]:
    """Generate all reviewer-requested plots for one model."""

    os.makedirs(output_dir, exist_ok=True)
    written = []
    prefix = model_title.lower().replace("-", "").replace(" ", "_")

    # Determine hue
    if hue_col is None and "draft_num_layers_effective" in df.columns:
        if df["draft_num_layers_effective"].nunique(dropna=False) > 1:
            hue_col = "draft_num_layers_effective"

    # Filter for the truncated mode if progen2 has both modes
    if "draft_mode" in df.columns:
        if df["draft_mode"].nunique(dropna=False) > 1:
            trunc = df[df["draft_mode"] == "truncated"]
            if len(trunc) > 0:
                df = trunc

    # Fix nuisance variables
    keep_varying = {x_col}
    if hue_col:
        keep_varying.add(hue_col)
    df = _modal_filter(df, keep_varying)

    if len(df) == 0:
        print(f"  [SKIP] No data after filtering for {model_title}")
        return written

    # Compute accepted_prefix_len = mean_accept_rate * L
    # and ideal_speedup = accepted_prefix_len + 1 (theoretical upper bound)
    if "mean_accept_rate" in df.columns and "L" in df.columns:
        df = df.copy()
        df["accepted_prefix_len"] = df["mean_accept_rate"] * df["L"]
        df["ideal_speedup"] = df["accepted_prefix_len"] + 1

    # Aggregate
    group = _group_cols(df, x_col, hue_col)
    metrics = [
        "speedup_vs_target", "specdec_tps", "target_tps", "draft_tps",
        "mean_accept_rate", "mean_accepted_prefix", "accepted_prefix_len",
        "ideal_speedup",
    ]
    agg = aggregate_by(df, group, metrics)

    # ── 1) Five-panel figure: Speedup + Ideal Speedup + TPS + Accept Rate + Accepted Prefix ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    _plot_metric_vs_L(
        agg, x_col, "speedup_vs_target_mean", "speedup_vs_target_std",
        hue_col, axes[0, 0],
        y_label="Speedup vs Target", title="Wall-Clock Speedup",
    )
    axes[0, 0].axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    _plot_metric_vs_L(
        agg, x_col, "ideal_speedup_mean", "ideal_speedup_std",
        hue_col, axes[0, 1],
        y_label="Ideal Speedup (α×γ + 1)", title="Ideal Speedup (Upper Bound)",
        legend=False,
    )
    axes[0, 1].axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    _plot_metric_vs_L(
        agg, x_col, "specdec_tps_mean", "specdec_tps_std",
        hue_col, axes[0, 2],
        y_label="Tokens / Second", title="SpecDec Token Rate",
        legend=False,
    )

    _plot_metric_vs_L(
        agg, x_col, "mean_accept_rate_mean", "mean_accept_rate_std",
        hue_col, axes[1, 0],
        y_label="Acceptance Rate (α)", title="Draft Acceptance Rate",
        legend=False,
    )
    axes[1, 0].set_ylim(bottom=0, top=min(1.05, axes[1, 0].get_ylim()[1] * 1.1))

    _plot_metric_vs_L(
        agg, x_col, "accepted_prefix_len_mean", "accepted_prefix_len_std",
        hue_col, axes[1, 1],
        y_label="Accepted Tokens (α × γ)", title="Acceptance Length",
        legend=False,
    )

    # Hide the unused bottom-right subplot
    axes[1, 2].set_visible(False)

    fig.suptitle(f"{model_title}: Speedup and Acceptance Metrics vs Draft Length",
                 fontsize=TITLE_FONTSIZE + 2, y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_speedup_acceptance_panel.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(out)
    print(f"  Saved: {out}")

    # ── 2) Individual standalone plots ──
    individual_metrics = [
        ("speedup_vs_target", "Speedup vs Target", "Wall-Clock Speedup"),
        ("ideal_speedup", "Ideal Speedup (α×γ + 1)", "Ideal Speedup (Upper Bound)"),
        ("specdec_tps", "Tokens / Second", "SpecDec Token Rate"),
        ("mean_accept_rate", "Acceptance Rate (α)", "Draft Acceptance Rate"),
        ("accepted_prefix_len", "Accepted Tokens (α × γ)", "Acceptance Length"),
    ]
    for metric, ylabel, plot_title in individual_metrics:
        if f"{metric}_mean" not in agg.columns:
            continue
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        _plot_metric_vs_L(
            agg, x_col, f"{metric}_mean", f"{metric}_std",
            hue_col, ax2,
            y_label=ylabel, title=f"{model_title}: {plot_title}",
        )
        if metric in ("speedup_vs_target", "ideal_speedup"):
            ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        if metric == "mean_accept_rate":
            ax2.set_ylim(bottom=0, top=min(1.05, ax2.get_ylim()[1] * 1.1))
        fig2.tight_layout()
        out2 = os.path.join(output_dir, f"{prefix}_{metric}_vs_{x_col}.png")
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        written.append(out2)
        print(f"  Saved: {out2}")

    # ── 3) Summary table CSV (LaTeX-friendly) ──
    table_cols = [x_col]
    if hue_col:
        table_cols.append(hue_col)
    table_cols += [
        "speedup_vs_target_mean", "speedup_vs_target_std",
        "ideal_speedup_mean", "ideal_speedup_std",
        "specdec_tps_mean", "target_tps_mean",
        "mean_accept_rate_mean", "mean_accept_rate_std",
        "accepted_prefix_len_mean", "accepted_prefix_len_std",
    ]
    table = agg[[c for c in table_cols if c in agg.columns]].copy()

    # Rename for readability
    rename_map = {
        "speedup_vs_target_mean": "speedup_mean",
        "speedup_vs_target_std": "speedup_std",
        "ideal_speedup_mean": "ideal_speedup_mean",
        "ideal_speedup_std": "ideal_speedup_std",
        "specdec_tps_mean": "specdec_tps",
        "target_tps_mean": "target_tps",
        "mean_accept_rate_mean": "accept_rate_mean",
        "mean_accept_rate_std": "accept_rate_std",
        "accepted_prefix_len_mean": "accept_len_mean",
        "accepted_prefix_len_std": "accept_len_std",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})

    # Round for readability
    for c in table.select_dtypes(include="number").columns:
        table[c] = table[c].round(3)

    table_path = os.path.join(output_dir, f"{prefix}_summary_table.csv")
    table.to_csv(table_path, index=False)
    written.append(table_path)
    print(f"  Saved: {table_path}")

    return written


# ═══════════════════════════════════════════════════════════════════════
# Combined TPS bar chart across all 3 models
# ═══════════════════════════════════════════════════════════════════════

def plot_combined_tps_bars(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Optional[str]:
    """Horizontal bar chart: Target TPS vs SpecDec TPS for each model."""
    if not dfs:
        return None

    models = []
    for name, df in dfs.items():
        t_tps = df["target_tps"].mean() if "target_tps" in df.columns else 0
        s_tps = df["specdec_tps"].mean() if "specdec_tps" in df.columns else 0
        models.append((name, t_tps, s_tps))

    if not models:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(models))
    h = 0.35

    target_vals = [m[1] for m in models]
    specdec_vals = [m[2] for m in models]
    labels = [m[0] for m in models]

    ax.barh(y - h/2, target_vals, h, label="Target (baseline)",
            color=METHOD_COLORS["target"], edgecolor="black", linewidth=0.5)
    ax.barh(y + h/2, specdec_vals, h, label="SpecDec",
            color=METHOD_COLORS["specdec"], edgecolor="black", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("Tokens / Second", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Token Generation Rate: Target vs SpecDec", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = os.path.join(output_dir, "combined_tps_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate speedup + acceptance rate plots for reviewer 2.a.2"
    )
    parser.add_argument("--output_dir", default="results/speedup_acceptance_plots",
                        help="Output directory for plots")
    parser.add_argument("--dnagpt_csv",
                        default="results/dnagpt_final_scored_filtered.csv")
    parser.add_argument("--progen2_csv",
                        default="results/progen2_final_final_scored.csv")
    parser.add_argument("--protgpt2_csv",
                        default="results/protgpt2_wide_scored.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_dfs = {}

    # ── DNAGPT ──
    if os.path.exists(args.dnagpt_csv):
        print(f"\nProcessing DNAGPT: {args.dnagpt_csv}")
        df = load_csv(args.dnagpt_csv)
        all_dfs["DNAGPT"] = df
        plot_model(df, "DNAGPT", args.output_dir, x_col="L", hue_col="temperature")
    else:
        print(f"[SKIP] {args.dnagpt_csv} not found")

    # ── ProtGPT2 ──
    if os.path.exists(args.protgpt2_csv):
        print(f"\nProcessing ProtGPT2: {args.protgpt2_csv}")
        df = load_csv(args.protgpt2_csv)
        all_dfs["ProtGPT2"] = df
        plot_model(df, "ProtGPT2", args.output_dir, x_col="L",
                   hue_col="draft_num_layers_effective")
    else:
        print(f"[SKIP] {args.protgpt2_csv} not found")

    # ── ProGen2 ──
    if os.path.exists(args.progen2_csv):
        print(f"\nProcessing ProGen2: {args.progen2_csv}")
        df = load_csv(args.progen2_csv)
        all_dfs["ProGen2"] = df
        plot_model(df, "ProGen2", args.output_dir, x_col="L",
                   hue_col="draft_num_layers_effective")
    else:
        print(f"[SKIP] {args.progen2_csv} not found")

    # ── Combined TPS bar chart ──
    if all_dfs:
        print("\nGenerating combined TPS comparison...")
        plot_combined_tps_bars(all_dfs, args.output_dir)

    print("\nAll plots complete!")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
