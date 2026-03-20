#!/usr/bin/env python3
"""
Reviewer 2.a.3 — Biological feature ↔ speculative decoding performance analysis.

Extracts biological features from prompt and generated sequences, then correlates
them with speculative decoding performance metrics (acceptance rate, speedup, PPL).

Protein features (ProtGPT2, ProGen2):
  - Amino acid composition, sequence length
  - Shannon entropy, hydrophobicity (Kyte-Doolittle)
  - Net charge, aromatic fraction, disorder propensity
  - Dipeptide entropy

DNA features (DNAGPT):
  - GC content, CpG observed/expected ratio
  - Dinucleotide entropy, homopolymer run length
  - Linguistic complexity (distinct k-mers / possible k-mers)

Produces:
  - Correlation heatmaps (bio features vs. performance metrics) per model
  - Scatter plots for top-correlated bio features
  - Per-prompt bar charts of biological feature profiles with acceptance overlay
  - Summary CSV of all correlations
"""

import argparse
import os
from collections import Counter
from math import erfc, log2, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── Utilities ────────────────────────────────────────────────────────────────

def pearsonr(x, y):
    """Numpy-only Pearson r + two-sided p-value (avoids broken scipy)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    xm, ym = x - x.mean(), y - y.mean()
    r_num = np.dot(xm, ym)
    r_den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    r = r_num / r_den if r_den > 0 else 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-30))
    p = erfc(abs(t_stat) / sqrt(2))
    return float(r), float(p)


def _save(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {stem}")


# ─── Biological feature extractors — PROTEIN ─────────────────────────────────

# Kyte-Doolittle hydrophobicity scale
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
_AROMATIC = set("FWY")
_POSITIVE = set("RKH")
_NEGATIVE = set("DE")
# Disorder propensity (TOP-IDP scale, simplified)
_DISORDER = set("AGQSPENDRK")
_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def protein_features(seq):
    """Compute biological features for an amino acid sequence."""
    seq = seq.upper().strip()
    L = len(seq)
    if L == 0:
        return {}
    feats = {"seq_len": L}
    # Composition counts
    counts = Counter(seq)
    total = float(L)
    # Shannon entropy
    freqs = np.array([counts.get(aa, 0) / total for aa in _STD_AA])
    freqs = freqs[freqs > 0]
    feats["shannon_entropy"] = float(-np.sum(freqs * np.log2(freqs))) if len(freqs) > 1 else 0.0
    # Hydrophobicity (mean Kyte-Doolittle)
    kd_vals = [_KD.get(c, 0.0) for c in seq if c in _KD]
    feats["hydrophobicity"] = float(np.mean(kd_vals)) if kd_vals else 0.0
    # Net charge at pH 7 (approximate: K,R,H positive; D,E negative)
    pos = sum(1 for c in seq if c in _POSITIVE)
    neg = sum(1 for c in seq if c in _NEGATIVE)
    feats["net_charge"] = (pos - neg) / total
    feats["charge_density"] = (pos + neg) / total
    # Aromatic fraction
    feats["aromatic_frac"] = sum(1 for c in seq if c in _AROMATIC) / total
    # Disorder propensity
    feats["disorder_frac"] = sum(1 for c in seq if c in _DISORDER) / total
    # Dipeptide entropy
    if L >= 2:
        di = [seq[i:i+2] for i in range(L - 1)]
        di_counts = Counter(di)
        di_total = float(len(di))
        di_freqs = np.array([v / di_total for v in di_counts.values()])
        feats["dipeptide_entropy"] = float(-np.sum(di_freqs * np.log2(di_freqs)))
    else:
        feats["dipeptide_entropy"] = 0.0
    # Small / tiny fraction
    feats["small_aa_frac"] = sum(1 for c in seq if c in set("AGSCTDN")) / total
    return feats


# ─── Biological feature extractors — DNA ─────────────────────────────────────

def dna_features(seq):
    """Compute biological features for a DNA sequence."""
    seq = seq.upper().strip()
    L = len(seq)
    if L == 0:
        return {}
    feats = {"seq_len": L}
    counts = Counter(seq)
    gc = (counts.get("G", 0) + counts.get("C", 0))
    feats["gc_content"] = gc / L
    # CpG observed / expected
    cpg_obs = seq.count("CG")
    c_count = counts.get("C", 0)
    g_count = counts.get("G", 0)
    cpg_exp = (c_count * g_count) / L if L > 0 else 1
    feats["cpg_obs_exp"] = cpg_obs / cpg_exp if cpg_exp > 0 else 0.0
    # Dinucleotide entropy
    if L >= 2:
        di = [seq[i:i+2] for i in range(L - 1)]
        di_counts = Counter(di)
        di_total = float(len(di))
        di_freqs = np.array([v / di_total for v in di_counts.values()])
        feats["dinuc_entropy"] = float(-np.sum(di_freqs * np.log2(di_freqs)))
    else:
        feats["dinuc_entropy"] = 0.0
    # Homopolymer max run length
    max_run = 1
    cur_run = 1
    for i in range(1, L):
        if seq[i] == seq[i - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 1
    feats["max_homopolymer"] = max_run
    # Trinucleotide (3-mer) linguistic complexity
    k = 3
    if L >= k:
        possible = min(4**k, L - k + 1)
        distinct = len(set(seq[i:i+k] for i in range(L - k + 1)))
        feats["trinuc_complexity"] = distinct / possible
    else:
        feats["trinuc_complexity"] = 0.0
    # Mono-nucleotide entropy
    nuc_freqs = np.array([counts.get(b, 0) / L for b in "ACGT"])
    nuc_freqs = nuc_freqs[nuc_freqs > 0]
    feats["mono_entropy"] = float(-np.sum(nuc_freqs * np.log2(nuc_freqs))) if len(nuc_freqs) > 1 else 0.0
    # AT skew = (A-T)/(A+T)
    a_c, t_c = counts.get("A", 0), counts.get("T", 0)
    feats["at_skew"] = (a_c - t_c) / (a_c + t_c) if (a_c + t_c) > 0 else 0.0
    # GC skew = (G-C)/(G+C)
    feats["gc_skew"] = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0.0
    return feats


# ─── Sequence extraction helpers ─────────────────────────────────────────────

def _extract_protein_prompt(hg_id):
    """Extract the amino acid prompt from hg_id column."""
    s = str(hg_id)
    # ProtGPT2: "<|endoftext|>MKWV..."
    if s.startswith("<|endoftext|>"):
        return s[len("<|endoftext|>"):]
    # ProGen2: "1MKWV..." or "2MKWV..."
    if len(s) > 1 and s[0] in "12" and s[1:2].isalpha():
        return s[1:]
    return s


def _extract_dna_prompts(df, results_dir):
    """Join DNAGPT data with hg38_sequences.csv for raw DNA sequences."""
    seq_path = os.path.join(results_dir, "hg38_sequences.csv")
    if not os.path.exists(seq_path):
        # Try parent dir
        seq_path = os.path.join(os.path.dirname(results_dir), "hg38_sequences.csv")
    if not os.path.exists(seq_path):
        print("  WARNING: hg38_sequences.csv not found, DNA prompt features will be limited")
        return df
    hg = pd.read_csv(seq_path)
    # Merge on hg_id == id
    if "id" in hg.columns and "seq" in hg.columns:
        hg = hg.rename(columns={"seq": "raw_dna_seq"})
        df = df.merge(hg[["id", "raw_dna_seq"]], left_on="hg_id", right_on="id", how="left")
        matched = df["raw_dna_seq"].notna().sum()
        print(f"  Joined {matched}/{len(df)} DNAGPT rows with raw DNA sequences")
    return df


# ─── Feature computation per row ─────────────────────────────────────────────

def compute_protein_bio_features(df, prefix="prompt"):
    """Add biological feature columns for protein sequences."""
    seqs = df[f"__{prefix}_seq"]
    all_feats = [protein_features(s) for s in seqs]
    feat_df = pd.DataFrame(all_feats, index=df.index)
    feat_df.columns = [f"{prefix}_{c}" for c in feat_df.columns]
    return pd.concat([df, feat_df], axis=1)


def compute_dna_bio_features(df, seq_col, prefix="prompt"):
    """Add biological feature columns for DNA sequences."""
    seqs = df[seq_col].fillna("")
    all_feats = [dna_features(s) for s in seqs]
    feat_df = pd.DataFrame(all_feats, index=df.index)
    feat_df.columns = [f"{prefix}_{c}" for c in feat_df.columns]
    return pd.concat([df, feat_df], axis=1)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(results_dir):
    datasets = {}
    files = {
        "ProtGPT2": "protgpt2_wide_scored.csv",
        "ProGen2":  "progen2_final_final_scored.csv",
        "DNAGPT":   "dnagpt_final_scored_filtered.csv",
    }
    for name, fname in files.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["mean_accept_rate"].notna() & (df["mean_accept_rate"] > 0)]
            datasets[name] = df
            print(f"  {name}: {len(df)} specdec rows from {fname}")
        else:
            print(f"  WARNING: {path} not found, skipping {name}")
    return datasets


def enrich_with_bio_features(datasets, results_dir):
    """Add biological feature columns to each dataset."""
    for name, df in datasets.items():
        if name in ("ProtGPT2", "ProGen2"):
            # Extract prompt sequence
            df["__prompt_seq"] = df["hg_id"].apply(_extract_protein_prompt)
            df = compute_protein_bio_features(df, prefix="prompt")
            # Extract generated (specdec suffix) sequence
            if "sample_specdec_suffix" in df.columns:
                df["__output_seq"] = df["sample_specdec_suffix"].fillna("")
                df = compute_protein_bio_features(df, prefix="output")
            print(f"  {name}: extracted {len([c for c in df.columns if c.startswith('prompt_')])} prompt + "
                  f"{len([c for c in df.columns if c.startswith('output_')])} output bio features")
        elif name == "DNAGPT":
            # Join raw DNA sequences
            df = _extract_dna_prompts(df, results_dir)
            if "raw_dna_seq" in df.columns:
                df = compute_dna_bio_features(df, seq_col="raw_dna_seq", prefix="prompt")
            # Generated DNA suffix
            if "sample_specdec_suffix" in df.columns:
                df = compute_dna_bio_features(df, seq_col="sample_specdec_suffix", prefix="output")
            print(f"  DNAGPT: extracted {len([c for c in df.columns if c.startswith('prompt_')])} prompt + "
                  f"{len([c for c in df.columns if c.startswith('output_')])} output bio features")
        datasets[name] = df
    return datasets


# ─── Plotting: correlation heatmap ───────────────────────────────────────────

PERF_COLS = ["mean_accept_rate", "speedup_vs_target", "target_suffix_ppl"]
PERF_SHORT = {"mean_accept_rate": "Acceptance", "speedup_vs_target": "Speedup",
              "target_suffix_ppl": "PPL"}


def _bio_feature_cols(df, prefixes=("prompt_", "output_")):
    """Return list of biological feature columns present in df."""
    skip = {"prompt_len_tokens", "prompt_idx"}
    return [c for c in df.columns
            if any(c.startswith(p) for p in prefixes)
            and c not in skip
            and not c.startswith("__")]


def _pretty_feat(col):
    """Shorten column name for display."""
    return col.replace("prompt_", "P:").replace("output_", "O:")


def plot_bio_correlation_heatmap(datasets, output_dir):
    """Correlation heatmap: bio features (rows) vs performance metrics (cols)."""
    for name, df in datasets.items():
        bio_cols = _bio_feature_cols(df)
        perf_avail = [c for c in PERF_COLS if c in df.columns]
        if not bio_cols or not perf_avail:
            continue

        # Compute correlation matrix: bio_features × perf_metrics
        n_bio = len(bio_cols)
        n_perf = len(perf_avail)
        corr = np.full((n_bio, n_perf), np.nan)
        pvals = np.full((n_bio, n_perf), np.nan)
        for i, bc in enumerate(bio_cols):
            for j, pc in enumerate(perf_avail):
                r, p = pearsonr(df[bc].values, df[pc].values)
                corr[i, j] = r
                pvals[i, j] = p

        fig, ax = plt.subplots(figsize=(max(4, n_perf * 2.2), max(5, n_bio * 0.45 + 1.5)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_perf))
        ax.set_yticks(range(n_bio))
        ax.set_xticklabels([PERF_SHORT.get(c, c) for c in perf_avail], rotation=30, ha="right")
        ax.set_yticklabels([_pretty_feat(c) for c in bio_cols])
        for i in range(n_bio):
            for j in range(n_perf):
                if np.isnan(corr[i, j]):
                    continue
                color = "white" if abs(corr[i, j]) > 0.4 else "black"
                star = "*" if pvals[i, j] < 0.05 else ""
                ax.text(j, i, f"{corr[i, j]:.2f}{star}", ha="center", va="center",
                        fontsize=8, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
        ax.set_title(f"{name} — Bio Features vs. SpeDec Performance (* p<0.05)")
        _save(fig, output_dir, f"bio_corr_heatmap_{name.lower()}")


# ─── Plotting: scatter of top-correlated features ───────────────────────────

def plot_top_bio_scatter(datasets, output_dir, top_n=4):
    """Scatter plots of the top-N most correlated bio features vs acceptance."""
    for name, df in datasets.items():
        bio_cols = _bio_feature_cols(df)
        if not bio_cols or "mean_accept_rate" not in df.columns:
            continue
        # Rank by |r|
        ranking = []
        for bc in bio_cols:
            r, p = pearsonr(df[bc].values, df["mean_accept_rate"].values)
            if not np.isnan(r):
                ranking.append((bc, r, p))
        ranking.sort(key=lambda x: abs(x[1]), reverse=True)
        top = ranking[:top_n]
        if not top:
            continue

        ncols = min(len(top), 2)
        nrows = (len(top) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for idx, (bc, r, p) in enumerate(top):
            ax = axes[idx // ncols, idx % ncols]
            valid = np.isfinite(df[bc]) & np.isfinite(df["mean_accept_rate"])
            ax.scatter(df.loc[valid, bc], df.loc[valid, "mean_accept_rate"],
                       alpha=0.4, s=15, edgecolors="none")
            # Trend line
            if valid.sum() > 2:
                z = np.polyfit(df.loc[valid, bc], df.loc[valid, "mean_accept_rate"], 1)
                xs = np.linspace(df.loc[valid, bc].min(), df.loc[valid, bc].max(), 50)
                ax.plot(xs, np.polyval(z, xs), "r--", lw=1.2)
            ax.set_xlabel(_pretty_feat(bc))
            ax.set_ylabel("Acceptance rate")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"r={r:.3f} p={p:.2e} {sig}")
        # Hide unused panels
        for idx in range(len(top), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.suptitle(f"{name} — Top Bio Features vs. Acceptance Rate", fontsize=13, y=1.02)
        _save(fig, output_dir, f"bio_scatter_top_{name.lower()}")


# ─── Plotting: per-prompt bio feature profile with acceptance ────────────────

def plot_prompt_bio_profile(datasets, output_dir):
    """Bar chart: per-prompt bio features with acceptance rate overlay."""
    for name, df in datasets.items():
        if name in ("ProtGPT2", "ProGen2"):
            feat_subset = ["prompt_hydrophobicity", "prompt_shannon_entropy",
                           "prompt_net_charge", "prompt_disorder_frac",
                           "prompt_aromatic_frac"]
        elif name == "DNAGPT":
            feat_subset = ["prompt_gc_content", "prompt_cpg_obs_exp",
                           "prompt_dinuc_entropy", "prompt_trinuc_complexity",
                           "prompt_mono_entropy"]
        else:
            continue

        feat_subset = [f for f in feat_subset if f in df.columns]
        if not feat_subset:
            continue

        # Aggregate per prompt
        group_col = "hg_id"
        if group_col not in df.columns:
            continue
        agg = df.groupby(group_col).agg(
            **{f: (f, "first") for f in feat_subset},
            acc=("mean_accept_rate", "mean"),
        ).reset_index()
        agg = agg.sort_values("acc")

        n_prompts = len(agg)
        if n_prompts == 0:
            continue
        n_feats = len(feat_subset)

        # Shorten prompt labels
        if name in ("ProtGPT2", "ProGen2"):
            labels = [_extract_protein_prompt(hid)[:15] + "..." if len(_extract_protein_prompt(hid)) > 15
                      else _extract_protein_prompt(hid) for hid in agg[group_col]]
        else:
            labels = [str(hid)[:18] for hid in agg[group_col]]

        fig, ax1 = plt.subplots(figsize=(max(8, n_prompts * 0.8), 5))
        x = np.arange(n_prompts)
        width = 0.8 / n_feats
        colors_bar = plt.cm.Set2(np.linspace(0, 1, n_feats))

        for fi, feat in enumerate(feat_subset):
            vals = agg[feat].values
            # Normalize to [0,1] for display
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                vals_norm = (vals - vmin) / (vmax - vmin)
            else:
                vals_norm = np.zeros_like(vals)
            offset = (fi - n_feats / 2 + 0.5) * width
            ax1.bar(x + offset, vals_norm, width, color=colors_bar[fi],
                    label=_pretty_feat(feat), alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax1.set_ylabel("Feature value (normalized)")
        ax1.legend(loc="upper left", fontsize=7, ncol=2)

        ax2 = ax1.twinx()
        ax2.plot(x, agg["acc"].values, "ko-", markersize=6, linewidth=2, label="Acceptance rate")
        ax2.set_ylabel("Mean acceptance rate")
        ax2.legend(loc="upper right", fontsize=8)

        ax1.set_title(f"{name} — Prompt Bio Features vs. Acceptance Rate")
        _save(fig, output_dir, f"bio_prompt_profile_{name.lower()}")


# ─── Plotting: output bio features vs acceptance ────────────────────────────

def plot_output_bio_profile(datasets, output_dir):
    """Scatter: bio features of generated sequences vs performance."""
    for name, df in datasets.items():
        if name in ("ProtGPT2", "ProGen2"):
            feat_subset = ["output_hydrophobicity", "output_shannon_entropy",
                           "output_net_charge", "output_disorder_frac",
                           "output_dipeptide_entropy"]
        elif name == "DNAGPT":
            feat_subset = ["output_gc_content", "output_cpg_obs_exp",
                           "output_dinuc_entropy", "output_trinuc_complexity",
                           "output_mono_entropy"]
        else:
            continue

        feat_subset = [f for f in feat_subset if f in df.columns]
        if not feat_subset:
            continue

        ncols = min(len(feat_subset), 3)
        nrows = (len(feat_subset) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
        for idx, feat in enumerate(feat_subset):
            ax = axes[idx // ncols, idx % ncols]
            valid = np.isfinite(df[feat]) & np.isfinite(df["mean_accept_rate"])
            if valid.sum() < 3:
                ax.set_visible(False)
                continue
            ax.scatter(df.loc[valid, feat], df.loc[valid, "mean_accept_rate"],
                       alpha=0.35, s=12, edgecolors="none")
            r, p = pearsonr(df.loc[valid, feat].values, df.loc[valid, "mean_accept_rate"].values)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"r={r:.3f} {sig}", fontsize=10)
            ax.set_xlabel(_pretty_feat(feat))
            ax.set_ylabel("Acceptance rate")
        for idx in range(len(feat_subset), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.suptitle(f"{name} — Output Bio Features vs. Acceptance", fontsize=12, y=1.02)
        _save(fig, output_dir, f"bio_output_scatter_{name.lower()}")


# ─── Plotting: prompt vs output feature delta ───────────────────────────────

def plot_feature_delta(datasets, output_dir):
    """Scatter: difference in bio features (output - prompt) vs acceptance."""
    for name, df in datasets.items():
        if name in ("ProtGPT2", "ProGen2"):
            pairs = [("prompt_hydrophobicity", "output_hydrophobicity"),
                     ("prompt_shannon_entropy", "output_shannon_entropy"),
                     ("prompt_net_charge", "output_net_charge"),
                     ("prompt_disorder_frac", "output_disorder_frac")]
        elif name == "DNAGPT":
            pairs = [("prompt_gc_content", "output_gc_content"),
                     ("prompt_dinuc_entropy", "output_dinuc_entropy"),
                     ("prompt_trinuc_complexity", "output_trinuc_complexity")]
        else:
            continue

        pairs = [(p, o) for p, o in pairs if p in df.columns and o in df.columns]
        if not pairs:
            continue

        ncols = min(len(pairs), 2)
        nrows = (len(pairs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for idx, (pcol, ocol) in enumerate(pairs):
            ax = axes[idx // ncols, idx % ncols]
            delta = df[ocol] - df[pcol]
            valid = np.isfinite(delta) & np.isfinite(df["mean_accept_rate"])
            if valid.sum() < 3:
                ax.set_visible(False)
                continue
            ax.scatter(delta[valid], df.loc[valid, "mean_accept_rate"],
                       alpha=0.35, s=12, edgecolors="none")
            r, p = pearsonr(delta[valid].values, df.loc[valid, "mean_accept_rate"].values)
            feat_name = pcol.replace("prompt_", "")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"Δ{feat_name}  r={r:.3f} {sig}", fontsize=10)
            ax.set_xlabel(f"Output − Prompt ({feat_name})")
            ax.set_ylabel("Acceptance rate")
            ax.axvline(0, color="gray", ls="--", lw=0.7)
        for idx in range(len(pairs), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.suptitle(f"{name} — Feature Shift (Output−Prompt) vs. Acceptance", fontsize=12, y=1.02)
        _save(fig, output_dir, f"bio_feature_delta_{name.lower()}")


# ─── Combined heatmap from CSV ───────────────────────────────────────────────

_METRIC_DISPLAY = {
    "mean_accept_rate": "Acceptance",
    "speedup_vs_target": "Speedup",
    "target_suffix_ppl": "PPL",
}
_MODEL_ORDER = ["DNAGPT", "ProGen2", "ProtGPT2"]


def _display_feature(raw):
    """Convert raw feature name to a readable display label."""
    if raw.startswith("prompt_"):
        base = raw[len("prompt_"):]
        tag = "(prompt)"
    elif raw.startswith("output_"):
        base = raw[len("output_"):]
        tag = "(output)"
    else:
        return raw
    pretty = base.replace("_", " ").replace("frac", "fraction").title()
    return f"{pretty} {tag}"


def plot_combined_heatmap_top(output_dir, top_n=7):
    """Combined 3-panel heatmap of top bio features from the correlations CSV."""
    csv_path = os.path.join(output_dir, "bio_feature_correlations.csv")
    if not os.path.exists(csv_path):
        print("  WARNING: bio_feature_correlations.csv not found, skipping combined heatmap")
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    metrics = ["mean_accept_rate", "speedup_vs_target", "target_suffix_ppl"]
    models = [m for m in _MODEL_ORDER if m in df["model"].unique()]

    # --- Feature selection ---
    # Per model: importance = max |r| across metrics where p<0.05, fallback max |r|
    importance_records = []
    for model in models:
        mdf = df[df["model"] == model]
        for feat in mdf["bio_feature"].unique():
            fdf = mdf[mdf["bio_feature"] == feat]
            sig = fdf[fdf["significant_005"] == True]
            if len(sig) > 0:
                score = sig["pearson_r"].abs().max()
            else:
                score = fdf["pearson_r"].abs().max()
            importance_records.append({"model": model, "bio_feature": feat, "score": score})
    imp = pd.DataFrame(importance_records)

    # Top N per model
    selected_feats = set()
    selected_by = {}  # feat -> list of models
    for model in models:
        sub = imp[imp["model"] == model].nlargest(top_n, "score")
        for _, row in sub.iterrows():
            selected_feats.add(row["bio_feature"])
            selected_by.setdefault(row["bio_feature"], []).append(model)

    # Global importance for ordering
    global_imp = imp.groupby("bio_feature")["score"].max()
    ordered_feats = sorted(selected_feats, key=lambda f: global_imp.get(f, 0), reverse=True)

    # Save selected feature list
    sel_rows = []
    for feat in ordered_feats:
        sel_rows.append({
            "feature_raw": feat,
            "feature_display": _display_feature(feat),
            "selected_by_model": "|".join(selected_by.get(feat, [])),
            "importance_score": float(global_imp.get(feat, 0)),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(output_dir, "bio_corr_heatmap_selected_features.csv"),
                  index=False, float_format="%.4f")
    print(f"  -> bio_corr_heatmap_selected_features.csv ({len(sel_df)} features)")

    # --- Build matrices ---
    n_feats = len(ordered_feats)
    n_metrics = len(metrics)
    n_models = len(models)

    corr_mats = {}   # model -> (n_feats, n_metrics)
    pval_mats = {}
    for model in models:
        mdf = df[df["model"] == model]
        corr = np.full((n_feats, n_metrics), np.nan)
        pval = np.full((n_feats, n_metrics), np.nan)
        for i, feat in enumerate(ordered_feats):
            for j, met in enumerate(metrics):
                row = mdf[(mdf["bio_feature"] == feat) & (mdf["perf_metric"] == met)]
                if len(row) == 1:
                    corr[i, j] = row["pearson_r"].values[0]
                    pval[i, j] = row["p_value"].values[0]
        corr_mats[model] = corr
        pval_mats[model] = pval

    # --- Plot ---
    feat_labels = [_display_feature(f) for f in ordered_feats]
    met_labels = [_METRIC_DISPLAY.get(m, m) for m in metrics]

    fig, axes = plt.subplots(1, n_models, figsize=(3.6 * n_models + 1.2, n_feats * 0.42 + 1.8),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        corr = corr_mats[model]
        pval = pval_mats[model]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(met_labels, rotation=30, ha="right", fontsize=9)
        if idx == 0:
            ax.set_yticks(range(n_feats))
            ax.set_yticklabels(feat_labels, fontsize=8)
        ax.set_title(model, fontsize=11, fontweight="bold")
        # Annotate cells
        for i in range(n_feats):
            for j in range(n_metrics):
                val = corr[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
                    continue
                star = "*" if (not np.isnan(pval[i, j]) and pval[i, j] < 0.05) else ""
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                        fontsize=7.5, color=color)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson r")

    fig.suptitle("Biological Features vs. Speculative Decoding Performance (* p<0.05)",
                 fontsize=12, y=0.98)
    stem = "bio_corr_heatmap_combined_top_features"
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {stem}")


def plot_combined_heatmap_full(output_dir):
    """Supplementary: combined 3-panel heatmap with ALL features (compressed)."""
    csv_path = os.path.join(output_dir, "bio_feature_correlations.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    metrics = ["mean_accept_rate", "speedup_vs_target", "target_suffix_ppl"]
    models = [m for m in _MODEL_ORDER if m in df["model"].unique()]
    all_feats = sorted(df["bio_feature"].unique())
    n_feats = len(all_feats)
    n_metrics = len(metrics)
    n_models = len(models)

    corr_mats, pval_mats = {}, {}
    for model in models:
        mdf = df[df["model"] == model]
        corr = np.full((n_feats, n_metrics), np.nan)
        pval = np.full((n_feats, n_metrics), np.nan)
        for i, feat in enumerate(all_feats):
            for j, met in enumerate(metrics):
                row = mdf[(mdf["bio_feature"] == feat) & (mdf["perf_metric"] == met)]
                if len(row) == 1:
                    corr[i, j] = row["pearson_r"].values[0]
                    pval[i, j] = row["p_value"].values[0]
        corr_mats[model] = corr
        pval_mats[model] = pval

    feat_labels = [_display_feature(f) for f in all_feats]
    met_labels = [_METRIC_DISPLAY.get(m, m) for m in metrics]

    fig, axes = plt.subplots(1, n_models, figsize=(3.4 * n_models + 1.2, n_feats * 0.32 + 1.8),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        corr = corr_mats[model]
        pval = pval_mats[model]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(met_labels, rotation=30, ha="right", fontsize=8)
        if idx == 0:
            ax.set_yticks(range(n_feats))
            ax.set_yticklabels(feat_labels, fontsize=6)
        ax.set_title(model, fontsize=10, fontweight="bold")
        for i in range(n_feats):
            for j in range(n_metrics):
                val = corr[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=5.5, color="gray")
                    continue
                star = "*" if (not np.isnan(pval[i, j]) and pval[i, j] < 0.05) else ""
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                        fontsize=5.5, color=color)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson r")
    fig.suptitle("All Bio Features vs. SpeDec Performance (Supplementary, * p<0.05)",
                 fontsize=11, y=0.98)
    stem = "bio_corr_heatmap_combined_full"
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {stem}")


# ─── Summary CSV ─────────────────────────────────────────────────────────────

def save_correlation_summary(datasets, output_dir):
    """Save CSV of all bio-feature × performance correlations."""
    rows = []
    for name, df in datasets.items():
        bio_cols = _bio_feature_cols(df)
        perf_avail = [c for c in PERF_COLS if c in df.columns]
        for bc in bio_cols:
            for pc in perf_avail:
                r, p = pearsonr(df[bc].values, df[pc].values)
                rows.append({
                    "model": name,
                    "bio_feature": bc,
                    "perf_metric": pc,
                    "pearson_r": r,
                    "p_value": p,
                    "n": int(np.isfinite(df[bc]).sum()),
                    "significant_005": p < 0.05 if not np.isnan(p) else False,
                })
    tbl = pd.DataFrame(rows)
    path = os.path.join(output_dir, "bio_feature_correlations.csv")
    tbl.to_csv(path, index=False, float_format="%.6f")
    print(f"  -> bio_feature_correlations.csv ({len(tbl)} rows)")

    # Print top correlations
    if len(tbl) > 0:
        sig = tbl[tbl["significant_005"] == True].copy()
        if len(sig) > 0:
            sig["abs_r"] = sig["pearson_r"].abs()
            sig = sig.sort_values("abs_r", ascending=False)
            print("\n  Top significant correlations (p<0.05):")
            for _, row in sig.head(15).iterrows():
                print(f"    {row['model']:10s} | {row['bio_feature']:30s} vs {row['perf_metric']:20s} | "
                      f"r={row['pearson_r']:+.3f}  p={row['p_value']:.2e}")
    return tbl


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Correlate biological features of sequences with speculative decoding performance")
    parser.add_argument("--results_dir", default="results",
                        help="Directory with scored CSV files")
    parser.add_argument("--output_dir", default="results/acceptance_characteristics",
                        help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading data from {args.results_dir}...")
    datasets = load_data(args.results_dir)

    if not datasets:
        print("No data found. Exiting.")
        return

    print("\nExtracting biological features...")
    datasets = enrich_with_bio_features(datasets, args.results_dir)

    print(f"\nGenerating bio-feature correlation plots in {args.output_dir}...")
    plot_bio_correlation_heatmap(datasets, args.output_dir)
    plot_top_bio_scatter(datasets, args.output_dir)
    plot_prompt_bio_profile(datasets, args.output_dir)
    plot_output_bio_profile(datasets, args.output_dir)
    plot_feature_delta(datasets, args.output_dir)
    save_correlation_summary(datasets, args.output_dir)

    print("\nGenerating combined heatmaps from correlations CSV...")
    plot_combined_heatmap_top(args.output_dir)
    plot_combined_heatmap_full(args.output_dir)

    print(f"\nDone. All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
