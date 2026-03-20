#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 16


def _infer_model_title(df: pd.DataFrame, fallback: str) -> str:
    """Short model title for plots (avoid full HF repo names)."""

    def _norm_key(s: str) -> str:
        # normalize for matching: lowercase and keep only [a-z0-9]
        s = str(s).strip().lower()
        return "".join(ch for ch in s if ch.isalnum())

    family = None
    if "model_family" in df.columns:
        vals = df["model_family"].dropna().unique().tolist()
        if len(vals) == 1:
            family = str(vals[0]).strip()

    label_map = {
        "protgpt2": "ProtGPT2",
        "dnagpt": "DNAGPT",
        "progen2": "ProGen2",
    }

    key = _norm_key(family) if family else _norm_key(fallback)
    if key in label_map:
        return label_map[key]

    # If it's not one of the known families, keep it readable.
    return family if family else fallback


METRICS = [
    ("speedup_vs_target_mean", "speedup_vs_target_std", "Speedup vs target"),
    ("specdec_tps_mean", "specdec_tps_std", "SpecDec tokens/s"),
    ("target_tps_mean", "target_tps_std", "Target tokens/s"),
    ("draft_tps_mean", "draft_tps_std", "Draft tokens/s"),
    ("mean_accept_rate_mean", "mean_accept_rate_std", "Mean accept rate"),
]


FILTER_CANDIDATE_COLS = [
    # Keep plots legible by default: we will hold these constant to their modal value,
    # unless explicitly used as x/hue.
    "model_family",
    "target_model_name",
    "draft_model_name",
    "draft_mode",
    "accept_mode",
    "temperature",
    "top_k",
    "top_p",
    "target_context_len",
    "draft_context_len",
    "prefix_len_tokens",
    "prompt_len_tokens",
]


@dataclass(frozen=True)
class PlotSpec:
    x_col: str
    hue_col: Optional[str]


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna().head(50), errors="raise")
        return True
    except Exception:
        return False


def _choose_x_col(df: pd.DataFrame) -> str:
    # Prefer the sweep axes you care about.
    candidates = [
        "L",  # gamma
        "draft_num_layers_effective",
        "draft_context_len",
        "target_context_len",
        "temperature",
        "top_p",
        "top_k",
    ]
    for c in candidates:
        if c in df.columns and df[c].nunique(dropna=False) > 1:
            return c

    # Fallback: first column with >1 unique
    for c in df.columns:
        if df[c].nunique(dropna=False) > 1:
            return c

    raise ValueError("No varying columns to use as x")


def _choose_hue_col(df: pd.DataFrame, x_col: str, max_levels: int) -> Optional[str]:
    candidates = [
        "draft_num_layers_effective",
        "draft_mode",
        "accept_mode",
        "draft_context_len",
        "temperature",
        "top_p",
        "top_k",
    ]
    for c in candidates:
        if c in df.columns and c != x_col:
            n = int(df[c].nunique(dropna=False))
            if 2 <= n <= max_levels:
                return c
    return None


def _prep_for_plot(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and _is_numeric_series(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _modal_slice(df: pd.DataFrame, keep_varying: Iterable[str]) -> pd.DataFrame:
    """Return a subset of df where "nuisance" columns are fixed to their modal values.

    This prevents plotting lines that mix different temperatures/top_p/context lengths, etc.
    """

    keep = set(keep_varying)
    out = df
    for c in FILTER_CANDIDATE_COLS:
        if c in keep:
            continue
        if c not in out.columns:
            continue
        # Only filter if there is variation.
        if int(out[c].nunique(dropna=False)) <= 1:
            continue
        mode_vals = out[c].mode(dropna=False)
        if mode_vals.empty:
            continue
        mode_val = mode_vals.iloc[0]
        out = out[out[c] == mode_val]
    return out


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = s.notna() & w.notna() & (w > 0)
    if not m.any():
        return float("nan")
    return float((s[m] * w[m]).sum() / w[m].sum())


def _weighted_std(series: pd.Series, weights: pd.Series) -> float:
    """Weighted standard deviation (population, not sample) across rows.

    This is useful when each row represents an aggregated chunk of data
    (e.g., grouped stats) and we want spread across configurations.
    """

    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = s.notna() & w.notna() & (w > 0)
    if not m.any():
        return float("nan")
    mean = float((s[m] * w[m]).sum() / w[m].sum())
    var = float((w[m] * (s[m] - mean) ** 2).sum() / w[m].sum())
    return float(var**0.5)


def _plot_dnagpt_ppl_bars_scored_by_draft_model(
    *,
    repo_dir: str,
    output_dir: str,
) -> Optional[str]:
    """DNAGPT PPL bars using draft-model scoring for all three generations.

    Reads the wide CSV produced by scripts/score_dnagpt_draft_suffix_with_draft_model.py
    and plots mean ± std across rows for:
      - target_suffix_ppl__draft_model
      - specdec_suffix_ppl__draft_model
      - draft_suffix_ppl__draft_model
    """

    path = os.path.join(repo_dir, "results", "dnagpt_final_filtered__all3_scored_by_draft_model.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    cols = {
        "target": "target_suffix_ppl__draft_model",
        "specdec": "specdec_suffix_ppl__draft_model",
        "draft": "draft_suffix_ppl__draft_model",
    }
    if not all(c in df.columns for c in cols.values()):
        return None

    df = _prep_for_plot(df, list(cols.values()))

    vals = [float(df[cols[k]].mean()) for k in ("target", "specdec", "draft")]
    yerr = [float(df[cols[k]].std()) for k in ("target", "specdec", "draft")]

    out_ppl = os.path.join(output_dir, "dnagpt__ppl_mean__bars__target_specdec_draft__scored_by_draft_model.png")
    plt.figure(figsize=(7, 5))
    plt.title("DNAGPT", fontsize=TITLE_FONTSIZE)
    x = ["target", "specdec", "draft"]
    colors = ["#c7c7c7", "#a6cee3", "#b2df8a"]
    plt.bar(x, vals, color=colors, yerr=yerr, capsize=7, ecolor="#666666")
    plt.ylabel("Perplexity", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_ppl, dpi=200)
    plt.close()
    return out_ppl


def _ppl_summary_from_grouped_stats(path: str) -> Optional[dict]:
    """Return weighted mean/std for target/specdec/draft PPL from a grouped-stats CSV."""

    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    ppl_cols = {
        "target": "target_suffix_ppl_mean",
        "specdec": "specdec_suffix_ppl_mean",
        "draft": "draft_suffix_ppl_mean",
    }
    if "n_rows" not in df.columns or not all(c in df.columns for c in ppl_cols.values()):
        return None

    df = _prep_for_plot(df, ["n_rows", *list(ppl_cols.values())])
    out: dict[str, tuple[float, float]] = {}
    for k, col in ppl_cols.items():
        mean = _weighted_mean(df[col], df["n_rows"])
        std = _weighted_std(df[col], df["n_rows"])
        out[k] = (mean, std)
    return out


def _ppl_summary_from_del family names (one label covering the 3 bars)
    """

    prot_ppl = _first_existing(
        [
            os.path.join(stats_dir, "protgpt2_wide_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "protgpt2_wide_grouped_stats.csv"),
        ]
    )
    prog_ppl = _first_existing(
        [
            os.path.join(stats_dir, "progen2_wide_final_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_final_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_grouped_stats.csv"),
        ]
    )
    dna_ppl = _first_existing(
        [
            os.path.join(stats_dir, "dnagpt_final_scored_filtered_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_final_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_final_filtered_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_hg38_wide_2_grouped_stats.csv"),
        ]
    )

    sources = [
        ("ProtGPT2", prot_ppl),
        ("ProGen2", prog_ppl),
        ("DNAGPT", dna_ppl),
    ]

    summaries: list[tuple[str, dict]] = []
    for name, p in sources:
        if name == "DNAGPT" and dnagpt_override_summary is not None:
            s = dnagpt_override_summary
        else:
            s = _ppl_summary_from_grouped_stats(p) if p else None
        if s is not None:
            summaries.append((name, s))

    if not summaries:
        return None

    # Plot config
    order = ["target", "specdec", "draft"]
    colors = {
        "target": "#c7c7c7",
        "specdec": "#a6cee3",
        "draft": "#b2df8a",
    }
    labels = {
        "target": "Target",
        "specdec": "SpecDec",
        "draft": "Draft",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale("log")

    group_gap = 0.2
    bar_h = 0.3
    inner_gap = 0.08

    y_positions: list[float] = []
    x_vals: list[float] = []
    x_errs: list[float] = []
    bar_colors: list[str] = []
    bar_kinds: list[str] = []
    centers: list[float] = []
    center_labels: list[str] = []

    y = 0.0
    for model_name, summ in summaries:
        group_ys = []
        for kind in order:
            mean, std = summ.get(kind, (float("nan"), float("nan")))
            # For log-scale, ensure we don't pass non-positive values.
            if mean is None or not (mean > 0):
                continue
            if std is None or not (std == std) or std < 0:
                std = 0.0
            y_positions.append(y)
            x_vals.append(float(mean))
            x_errs.append(float(std))
            bar_colors.append(colors[kind])
            bar_kinds.append(kind)
            group_ys.append(y)
            y += bar_h + inner_gap

        if group_ys:
            centers.append(sum(group_ys) / len(group_ys))
            center_labels.append(model_name)

        y += group_gap

    # Draw bars (one call per kind so the legend is clean)
    for kind in order:
        idxs = [i for i, k in enumerate(bar_kinds) if k == kind]
        if not idxs:
            continue
        ax.barh(
            [y_positions[i] for i in idxs],
            [x_vals[i] for i in idxs],
            xerr=[x_errs[i] for i in idxs],
            height=bar_h,
            color=colors[kind],
            edgecolor="none",
            alpha=0.95,
            capsize=7,
            error_kw={"elinewidth": 1.5, "capthick": 1.2},
            label=labels[kind],
        )

    ax.set_yticks(centers)
    ax.set_yticklabels(center_labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.invert_yaxis()

    ax.set_xlabel("Perplexity", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()

    out = os.path.join(output_dir, out_name)
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_requested(
    stats_dir: str,
    output_dir: str,
    max_hue_levels: int,
) -> List[str]:
    """Generate the specific plots the user asked for."""

    written: List[str] = []

    # --- ProtGPT2: speedup vs L, hue=draft_num_layers_effective ---
    prot = os.path.join(stats_dir, "protgpt2_wide_grouped_stats.csv")
    if os.path.exists(prot):
        df = pd.read_csv(prot)
        df = _prep_for_plot(df, ["L", "draft_num_layers_effective", "speedup_vs_target_mean", "speedup_vs_target_std"])
        df = _modal_slice(df, keep_varying=["L", "draft_num_layers_effective"])
        out_path = os.path.join(
            output_dir,
            "protgpt2__speedup_vs_target_mean__by__L__hue__draft_num_layers_effective.png",
        )
        if "speedup_vs_target_mean" in df.columns and "L" in df.columns:
            _plot_lines(
                df,
                out_path=out_path,
                title=None,
                x_col="L",
                y_col="speedup_vs_target_mean",
                y_std_col="speedup_vs_target_std" if "speedup_vs_target_std" in df.columns else None,
                hue_col="draft_num_layers_effective" if "draft_num_layers_effective" in df.columns else None,
                max_hue_levels=max_hue_levels,
                x_label="L",
                y_label="Speedup",
                integer_x=True,
            )
            written.append(out_path)

    # --- ProtGPT2: perplexity bar plot (target/specdec/draft) ---
    prot_ppl = _first_existing(
        [
            os.path.join(stats_dir, "protgpt2_wide_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "protgpt2_wide_grouped_stats.csv"),
        ]
    )
    if prot_ppl:
        dfp = pd.read_csv(prot_ppl)
        ppl_cols = {
            "target": "target_suffix_ppl_mean",
            "specdec": "specdec_suffix_ppl_mean",
            "draft": "draft_suffix_ppl_mean",
        }
        if "n_rows" in dfp.columns and all(c in dfp.columns for c in ppl_cols.values()):
            # For this summary, we want variability across the sweep configurations.
            # So we do NOT modal-filter sweep axes (including prompt_len_tokens).
            dfb = dfp
            dfb = _prep_for_plot(
                dfb,
                [
                    "n_rows",
                    *list(ppl_cols.values()),
                    "target_suffix_ppl_min",
                    "target_suffix_ppl_max",
                    "specdec_suffix_ppl_min",
                    "specdec_suffix_ppl_max",
                    "draft_suffix_ppl_min",
                    "draft_suffix_ppl_max",
                ],
            )

            vals = [
                _weighted_mean(dfb[ppl_cols["target"]], dfb["n_rows"]),
                _weighted_mean(dfb[ppl_cols["specdec"]], dfb["n_rows"]),
                _weighted_mean(dfb[ppl_cols["draft"]], dfb["n_rows"]),
            ]

            yerr = [
                _weighted_std(dfb[ppl_cols["target"]], dfb["n_rows"]),
                _weighted_std(dfb[ppl_cols["specdec"]], dfb["n_rows"]),
                _weighted_std(dfb[ppl_cols["draft"]], dfb["n_rows"]),
            ]

            out_ppl = os.path.join(output_dir, "protgpt2__ppl_mean__bars__target_specdec_draft.png")
            plt.figure(figsize=(7, 5))
            plt.title(_infer_model_title(dfp, fallback="protgpt2"), fontsize=TITLE_FONTSIZE)
            x = ["target", "specdec", "draft"]
            colors = ["#c7c7c7", "#a6cee3", "#b2df8a"]
            plt.bar(x, vals, color=colors, yerr=yerr, capsize=6, ecolor="#666666")
            plt.ylabel("Perplexity", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_ppl, dpi=200)
            plt.close()
            written.append(out_ppl)

    # --- ProGen2: requested plots ---
    prog = _first_existing(
        [
            os.path.join(stats_dir, "progen2_wide_final_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_final_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "progen2_wide_grouped_stats.csv"),
        ]
    )
    if prog:
        dfg = pd.read_csv(prog)
        dfg = _prep_for_plot(
            dfg,
            [
                "L",
                "draft_mode",
                "draft_num_layers_effective",
                "speedup_vs_target_mean",
                "speedup_vs_target_std",
                "mean_accept_rate_mean",
                "mean_accept_rate_std",
            ],
        )

        # (1) Comparison: truncated vs pretrained (speedup vs L, hue=draft_mode)
        if "draft_mode" in dfg.columns and "L" in dfg.columns and "speedup_vs_target_mean" in dfg.columns:
            d1 = _modal_slice(dfg, keep_varying=["L", "draft_mode"])
            out_cmp = os.path.join(output_dir, "progen2__speedup_vs_target_mean__by__L__hue__draft_mode.png")
            _plot_lines(
                d1,
                out_path=out_cmp,
                title=None,
                x_col="L",
                y_col="speedup_vs_target_mean",
                y_std_col="speedup_vs_target_std" if "speedup_vs_target_std" in d1.columns else None,
                hue_col="draft_mode",
                max_hue_levels=max_hue_levels,
                x_label="L",
                y_label="Speedup",
                integer_x=True,
            )
            written.append(out_cmp)

        # (2) Truncated only: mean acceptance vs draft layers (average across L)
        if "draft_mode" in dfg.columns and "draft_num_layers_effective" in dfg.columns and "mean_accept_rate_mean" in dfg.columns:
            d2 = dfg[dfg["draft_mode"] == "truncated"].copy()
            d2 = d2.dropna(subset=["draft_num_layers_effective", "mean_accept_rate_mean"])
            if not d2.empty and "n_rows" in d2.columns:
                rows = []
                for layers, sub in d2.groupby("draft_num_layers_effective", dropna=False):
                    rows.append(
                        {
                            "draft_num_layers_effective": layers,
                            "mean_accept_rate_mean": _weighted_mean(sub["mean_accept_rate_mean"], sub["n_rows"]),
                        }
                    )
                d2a = pd.DataFrame(rows).sort_values("draft_num_layers_effective")
                out_acc = os.path.join(output_dir, "progen2__mean_accept_rate_mean__by__draft_num_layers_effective__truncated.png")
                _plot_lines(
                    d2a,
                    out_path=out_acc,
                    title=None,
                    x_col="draft_num_layers_effective",
                    y_col="mean_accept_rate_mean",
                    y_std_col=None,
                    hue_col=None,
                    max_hue_levels=max_hue_levels,
                )
                written.append(out_acc)

        # (3) Truncated only: speedup vs layers with L sizes (hue=L)
        if (
            "draft_mode" in dfg.columns
            and "draft_num_layers_effective" in dfg.columns
            and "L" in dfg.columns
            and "speedup_vs_target_mean" in dfg.columns
        ):
            d3 = dfg[dfg["draft_mode"] == "truncated"].copy()
            d3 = _modal_slice(d3, keep_varying=["draft_num_layers_effective", "L"])
            out_sp = os.path.join(
                output_dir,
                "progen2__speedup_vs_target_mean__by__draft_num_layers_effective__hue__L__truncated.png",
            )
            _plot_lines(
                d3,
                out_path=out_sp,
                title=None,
                x_col="draft_num_layers_effective",
                y_col="speedup_vs_target_mean",
                y_std_col="speedup_vs_target_std" if "speedup_vs_target_std" in d3.columns else None,
                hue_col="L",
                max_hue_levels=max_hue_levels,
                y_label="Speedup",
            )
            written.append(out_sp)

        # (4) ProGen2: perplexity bar plot (target/specdec/draft)
        ppl_cols = {
            "target": "target_suffix_ppl_mean",
            "specdec": "specdec_suffix_ppl_mean",
            "draft": "draft_suffix_ppl_mean",
        }
        if "n_rows" in dfg.columns and all(c in dfg.columns for c in ppl_cols.values()):
            # For this summary, we want variability across the sweep configurations.
            dgb = dfg
            dgb = _prep_for_plot(
                dgb,
                [
                    "n_rows",
                    *list(ppl_cols.values()),
                    "target_suffix_ppl_min",
                    "target_suffix_ppl_max",
                    "specdec_suffix_ppl_min",
                    "specdec_suffix_ppl_max",
                    "draft_suffix_ppl_min",
                    "draft_suffix_ppl_max",
                ],
            )

            vals = [
                _weighted_mean(dgb[ppl_cols["target"]], dgb["n_rows"]),
                _weighted_mean(dgb[ppl_cols["specdec"]], dgb["n_rows"]),
                _weighted_mean(dgb[ppl_cols["draft"]], dgb["n_rows"]),
            ]

            yerr = [
                _weighted_std(dgb[ppl_cols["target"]], dgb["n_rows"]),
                _weighted_std(dgb[ppl_cols["specdec"]], dgb["n_rows"]),
                _weighted_std(dgb[ppl_cols["draft"]], dgb["n_rows"]),
            ]

            out_ppl = os.path.join(output_dir, "progen2__ppl_mean__bars__target_specdec_draft.png")
            plt.figure(figsize=(7, 5))
            plt.title(_infer_model_title(dfg, fallback="progen2"), fontsize=TITLE_FONTSIZE)
            x = ["target", "specdec", "draft"]
            colors = ["#c7c7c7", "#a6cee3", "#b2df8a"]
            plt.bar(x, vals, color=colors, yerr=yerr, capsize=6, ecolor="#666666")
            plt.ylabel("Perplexity", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_ppl, dpi=200)
            plt.close()
            written.append(out_ppl)

    # --- DNAGPT: requested plots ---
    # Prefer a grouped-stats file computed from a *scored* CSV (so *_ppl is available).
    dna = _first_existing(
        [
            os.path.join(stats_dir, "dnagpt_final_scored_filtered_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_final_scored_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_final_filtered_grouped_stats.csv"),
            os.path.join(stats_dir, "dnagpt_hg38_wide_2_grouped_stats.csv"),
        ]
    )
    if dna:
        df = pd.read_csv(dna)
        df = _prep_for_plot(df, ["L", "temperature", "speedup_vs_target_mean", "speedup_vs_target_std"])
        df = _modal_slice(df, keep_varying=["L", "temperature"])
        out_path = os.path.join(output_dir, "dnagpt__speedup_vs_target_mean__by__L__hue__temperature.png")
        if "speedup_vs_target_mean" in df.columns and "L" in df.columns:
            _plot_lines(
                df,
                out_path=out_path,
                title=None,
                x_col="L",
                y_col="speedup_vs_target_mean",
                y_std_col="speedup_vs_target_std" if "speedup_vs_target_std" in df.columns else None,
                hue_col="temperature" if "temperature" in df.columns else None,
                max_hue_levels=max_hue_levels,
                x_label="L",
                y_label="Speedup",
                integer_x=True,
            )
            written.append(out_path)

        # --- DNAGPT: perplexity bar plot (target/specdec/draft) ---
        # Uses weighted means over groups (weights=n_rows) so it reflects the underlying dataset.
        ppl_cols = {
            "target": "target_suffix_ppl_mean",
            "specdec": "specdec_suffix_ppl_mean",
            "draft": "draft_suffix_ppl_mean",
        }
        if "n_rows" in df.columns and all(c in df.columns for c in ppl_cols.values()):
            # For this bar plot, hold nuisance hyperparams constant to the modal value.
            # We don't keep any axes varying.
            # For this summary, we want variability across the sweep configurations.
            dfb = df
            dfb = _prep_for_plot(
                dfb,
                [
                    "n_rows",
                    *list(ppl_cols.values()),
                    "target_suffix_ppl_min",
                    "target_suffix_ppl_max",
                    "specdec_suffix_ppl_min",
                    "specdec_suffix_ppl_max",
                    "draft_suffix_ppl_min",
                    "draft_suffix_ppl_max",
                ],
            )

            vals = [
                _weighted_mean(dfb[ppl_cols["target"]], dfb["n_rows"]),
                _weighted_mean(dfb[ppl_cols["specdec"]], dfb["n_rows"]),
                _weighted_mean(dfb[ppl_cols["draft"]], dfb["n_rows"]),
            ]

            yerr = [
                _weighted_std(dfb[ppl_cols["target"]], dfb["n_rows"]),
                _weighted_std(dfb[ppl_cols["specdec"]], dfb["n_rows"]),
                _weighted_std(dfb[ppl_cols["draft"]], dfb["n_rows"]),
            ]

            out_ppl = os.path.join(output_dir, "dnagpt__ppl_mean__bars__target_specdec_draft.png")
            plt.figure(figsize=(7, 5))
            plt.title(_infer_model_title(df, fallback="dnagpt"), fontsize=TITLE_FONTSIZE)
            x = ["target", "specdec", "draft"]
            colors = ["#c7c7c7", "#a6cee3", "#b2df8a"]
            plt.bar(x, vals, color=colors, yerr=yerr, capsize=6, ecolor="#666666")
            plt.ylabel("Perplexity", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_ppl, dpi=200)
            plt.close()
            written.append(out_ppl)

        # --- DNAGPT: speedup vs prompt_len_tokens, hue=L ---
        # NOTE: this requires compute_grouped_statistics.py to group by prompt_len_tokens.
        if "prompt_len_tokens" in df.columns:
            df2 = pd.read_csv(dna)
            df2 = _prep_for_plot(df2, ["prompt_len_tokens", "L", "speedup_vs_target_mean", "speedup_vs_target_std"])
            df2 = _modal_slice(df2, keep_varying=["prompt_len_tokens", "L"])
            out_path2 = os.path.join(output_dir, "dnagpt__speedup_vs_target_mean__by__prompt_len_tokens__hue__L.png")
            if "speedup_vs_target_mean" in df2.columns:
                _plot_lines(
                    df2,
                    out_path=out_path2,
                    title=None,
                    x_col="prompt_len_tokens",
                    y_col="speedup_vs_target_mean",
                    y_std_col="speedup_vs_target_std" if "speedup_vs_target_std" in df2.columns else None,
                    hue_col="L" if "L" in df2.columns else None,
                    max_hue_levels=max_hue_levels,
                    x_label="Length of prompt (tokens)",
                    y_label="Speedup",
                    integer_x=True,
                )
                written.append(out_path2)

    # --- DNAGPT: PPL bars using draft-model scoring (all three generations) ---
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dnagpt_draft_scored = _plot_dnagpt_ppl_bars_scored_by_draft_model(repo_dir=repo_dir, output_dir=output_dir)
    if dnagpt_draft_scored:
        written.append(dnagpt_draft_scored)

    # --- Combined PPL plot across all model families (single figure) ---
    combined = _plot_ppl_combined(stats_dir=stats_dir, output_dir=output_dir)
    if combined:
        written.append(combined)

    # --- Combined PPL plot variant: DNAGPT bars scored by draft model ---
    # Uses the raw rescoring output (one row per example), so error bars reflect
    # spread across the dataset rather than spread across sweep configurations.
    dnagpt_draft_scored = os.path.join(
        os.path.dirname(stats_dir),
        "dnagpt_final_filtered__all3_scored_by_draft_model.csv",
    )
    dnagpt_override = _ppl_summary_from_raw_csv(
        dnagpt_draft_scored,
        {
            "target": "target_suffix_ppl__draft_model",
            "specdec": "specdec_suffix_ppl__draft_model",
            "draft": "draft_suffix_ppl__draft_model",
        },
    )
    combined_draft_scored = _plot_ppl_combined(
        stats_dir=stats_dir,
        output_dir=output_dir,
        dnagpt_override_summary=dnagpt_override,
        out_name="ppl__combined__horizontal__logx__dnagpt_scored_by_draft_model.png",
    )fective":
            # More readable for ProtGPT2: show just the draft layer count.
            try:
                ival = int(float(val))
                return f"Draft layers: {ival}"
            except Exception:
                return f"Draft layers: {val}"
        return f"{hue}={val}"

    if hue_col and hue_col in df.columns:
        # Make sure stable ordering.
        hue_vals = list(df[hue_col].dropna().unique())
        if len(hue_vals) > max_hue_levels:
            hue_col = None

    if hue_col:
        for hv, sub in df.groupby(hue_col, dropna=False):
            sub = sub.sort_values(x_col)
            x = sub[x_col]
            y = sub[y_col]
            plt.plot(x, y, marker="o", linewidth=1.5, label=_legend_label(hue_col, hv))
            if y_std_col and y_std_col in sub.columns:
                yerr = sub[y_std_col]
                if yerr.notna().any():
                    plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)
        plt.legend(fontsize=8)
    else:
        sub = df.sort_values(x_col)
        x = sub[x_col]
        y = sub[y_col]
        plt.plot(x, y, marker="o", linewidth=1.8)
        if y_std_col and y_std_col in sub.columns:
            yerr = sub[y_std_col]
            if yerr.notna().any():
                plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    if title:
        plt.title(title)
    plt.xlabel(x_label or x_col)
    plt.ylabel(y_label or y_col)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_scatter_speedup(df: pd.DataFrame, out_path: str, title: str) -> None:
    if "target_tps_mean" not in df.columns or "specdec_tps_mean" not in df.columns:
        return

    d = _prep_for_plot(df, ["target_tps_mean", "specdec_tps_mean", "speedup_vs_target_mean"])
    x = d["target_tps_mean"]
    y = d["specdec_tps_mean"]
    c = d["speedup_vs_target_mean"] if "speedup_vs_target_mean" in d.columns else None

    plt.figure(figsize=(7, 7))
    if c is not None:
        plt.scatter(x, y, c=c, cmap="viridis", s=30, alpha=0.9)
        cb = plt.colorbar()
        cb.set_label("speedup_vs_target_mean")
    else:
        plt.scatter(x, y, s=30, alpha=0.9)

    # y=x reference
    finite = pd.concat([x, y], axis=1).dropna()
    if not finite.empty:
        mn = float(min(finite["target_tps_mean"].min(), finite["specdec_tps_mean"].min()))
        mx = float(max(finite["target_tps_mean"].max(), finite["specdec_tps_mean"].max()))
        plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, color="gray")

    plt.title(title)
    plt.xlabel("target_tps_mean")
    plt.ylabel("specdec_tps_mean")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_one_grouped_csv(input_csv: str, output_dir: str, max_hue_levels: int) -> List[str]:
    df = pd.read_csv(input_csv)
    base = os.path.splitext(os.path.basename(input_csv))[0]

    x_col = _choose_x_col(df)
    hue_col = _choose_hue_col(df, x_col=x_col, max_levels=max_hue_levels)

    dfp = _prep_for_plot(df, [x_col])

    written: List[str] = []

    # Main metric lines
    for y_col, y_std_col, label in METRICS:
        if y_col not in dfp.columns:
            continue
        dfm = _prep_for_plot(dfp, [y_col] + ([y_std_col] if y_std_col else []))
        out_path = os.path.join(
            output_dir,
            f"{base}__{y_col}__by__{x_col}" + (f"__hue__{hue_col}" if hue_col else "") + ".png",
        )
        title = f"{base}: {label} vs {x_col}" + (f" (hue={hue_col})" if hue_col else "")
        _plot_lines(
            dfm,
            out_path=out_path,
            title=title,
            x_col=x_col,
            y_col=y_col,
            y_std_col=y_std_col if (y_std_col and y_std_col in dfm.columns) else None,
            hue_col=hue_col,
            max_hue_levels=max_hue_levels,
        )
        written.append(out_path)

    # Scatter: specdec vs target colored by speedup
    scatter_out = os.path.join(output_dir, f"{base}__scatter_specdec_vs_target.png")
    _plot_scatter_speedup(df, out_path=scatter_out, title=f"{base}: specdec vs target")
    if os.path.exists(scatter_out):
        written.append(scatter_out)

    return written


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot grouped sweep statistics from results/statistics/*_grouped_stats.csv. "
            "Generates PNG plots (speedup, TPS, accept rate) grouped by sweep hyperparameters."
        )
    )
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="One or more grouped_stats.csv files. If empty, uses --input_glob.",
    )
    p.add_argument(
        "--input_glob",
        default="results/statistics/*_grouped_stats.csv",
        help="Glob used when --inputs is empty.",
    )
    p.add_argument(
        "--output_dir",
        default="results/statistics/plots",
        help="Directory to write PNG files.",
    )
    p.add_argument(
        "--max_hue_levels",
        type=int,
        default=12,
        help="Max unique values to allow for hue (legend size control).",
    )
    p.add_argument(
        "--also_requested",
        action="store_true",
        help="Also generate a small set of specifically requested plots (paper-style).",
    )
    p.add_argument(
        "--requested_only",
        action="store_true",
        help="Only generate the requested paper plots (skip generic per-file plots).",
    )

    args = p.parse_args(argv)

    inputs = list(args.inputs)
    if not inputs:
        inputs = sorted(glob.glob(args.input_glob))

    if not inputs:
        raise SystemExit(f"No inputs matched (glob={args.input_glob})")

    _safe_mkdir(args.output_dir)

    if not args.requested_only:
        for path in inputs:
            written = plot_one_grouped_csv(path, output_dir=args.output_dir, max_hue_levels=args.max_hue_levels)
            print(f"[OK] {path} -> {len(written)} plots")

    if args.also_requested or args.requested_only:
        stats_dir = os.path.dirname(os.path.abspath(inputs[0]))
        more = _plot_requested(stats_dir=stats_dir, output_dir=args.output_dir, max_hue_levels=args.max_hue_levels)
        print(f"[OK] requested -> {len(more)} plots")


if __name__ == "__main__":
    main()
