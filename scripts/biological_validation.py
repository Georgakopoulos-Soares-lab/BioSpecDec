#!/usr/bin/env python3
"""
Reviewer 2.a — Biological validation of speculative decoding.

Compares biologically relevant sequence statistics between baseline (target-only)
and speculative decoding outputs across DNAGPT (DNA), ProGen2 (protein), and
ProtGPT2 (protein).

Metrics:
  DNA  — GC content, Shannon entropy, 4-mer frequency spectrum
  Prot — AA frequency distribution, Shannon entropy, hydrophobic fraction

Tests:
  KS test (per-sequence distributions), JSD (global frequency vectors),
  Pearson r (frequency vector correlation)

Produces:
  - bio_validation_figure.{png,pdf}        (2×2 main-text figure)
  - dna_validation_summary.csv
  - protein_validation_summary.csv
  - kmer/AA frequency CSVs per model/method
"""

import argparse
import os
import re
from collections import Counter
from itertools import product
from math import erfc, log2, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

# ══════════════════════════════════════════════════════════════════════════════
# Config — paths to scored CSVs (edit if needed)
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FILES = {
    "DNAGPT":   "results/dnagpt_final_scored_filtered.csv",
    "ProGen2":  "results/progen2_final_final_scored.csv",
    "ProtGPT2": "results/protgpt2_wide_scored.csv",
}

# Column mapping
COL_BASELINE = "sample_target_suffix"
COL_SPECDEC  = "sample_specdec_suffix"

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_AA = set("AVILMFWYC")
DNA_BASES = set("ACGT")


def pearsonr_np(x, y):
    """Numpy-only Pearson r + two-sided p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
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


def shannon_entropy(counts_dict, alphabet):
    """Shannon entropy in bits from a Counter over an alphabet."""
    total = sum(counts_dict.get(c, 0) for c in alphabet)
    if total == 0:
        return 0.0
    freqs = np.array([counts_dict.get(c, 0) / total for c in alphabet])
    freqs = freqs[freqs > 0]
    return float(-np.sum(freqs * np.log2(freqs)))


def clean_protein_seq(s):
    """Clean protein sequence: uppercase, strip special tokens, keep std AAs."""
    s = str(s).upper()
    s = s.replace("<|ENDOFTEXT|>", "").replace("\n", "").strip()
    return "".join(c for c in s if c in STD_AA)


def clean_dna_seq(s):
    """Clean DNA sequence: uppercase, keep only ACGT."""
    s = str(s).upper()
    return "".join(c for c in s if c in DNA_BASES)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_sequences(results_dir):
    """Load and clean sequences from scored CSVs.

    Returns dict: model -> {"baseline": [str], "specdec": [str]}
    """
    data = {}
    for model, relpath in INPUT_FILES.items():
        path = os.path.join(results_dir, relpath) if not os.path.isabs(relpath) else relpath
        if not os.path.exists(path):
            # Try without results/ prefix if results_dir already includes it
            alt = relpath.replace("results/", "")
            path2 = os.path.join(results_dir, alt)
            if os.path.exists(path2):
                path = path2
            else:
                print(f"  WARNING: {path} not found, skipping {model}")
                continue

        df = pd.read_csv(path)
        cleaner = clean_dna_seq if model == "DNAGPT" else clean_protein_seq

        baseline_raw = df[COL_BASELINE].dropna().astype(str).tolist()
        specdec_raw = df[COL_SPECDEC].dropna().astype(str).tolist()

        baseline = [cleaner(s) for s in baseline_raw]
        specdec = [cleaner(s) for s in specdec_raw]

        # Filter out very short sequences (< 10 residues/bases)
        baseline = [s for s in baseline if len(s) >= 10]
        specdec = [s for s in specdec if len(s) >= 10]

        data[model] = {"baseline": baseline, "specdec": specdec}
        print(f"  {model}: baseline={len(baseline)}, specdec={len(specdec)} sequences "
              f"(median len {np.median([len(s) for s in baseline]):.0f} / "
              f"{np.median([len(s) for s in specdec]):.0f})")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# DNA statistics
# ══════════════════════════════════════════════════════════════════════════════

def dna_gc_content(seq):
    n = len(seq)
    if n == 0:
        return np.nan
    gc = sum(1 for c in seq if c in "GC")
    return gc / n


def dna_entropy(seq):
    counts = Counter(seq)
    return shannon_entropy(counts, "ACGT")


def kmer_frequencies(seqs, k=4):
    """Compute normalized k-mer frequency vector across all sequences."""
    all_kmers = ["".join(x) for x in product("ACGT", repeat=k)]
    counts = Counter()
    total = 0
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(c in DNA_BASES for c in kmer):
                counts[kmer] += 1
                total += 1
    freq = np.array([counts.get(km, 0) for km in all_kmers], dtype=float)
    if total > 0:
        freq /= total
    return dict(zip(all_kmers, freq)), freq, all_kmers


def compute_dna_stats(seqs_dict, output_dir, model="DNAGPT"):
    """Compute DNA statistics and comparisons. Returns summary dict."""
    baseline = seqs_dict["baseline"]
    specdec = seqs_dict["specdec"]

    # Per-sequence metrics
    gc_bl = np.array([dna_gc_content(s) for s in baseline])
    gc_sp = np.array([dna_gc_content(s) for s in specdec])
    ent_bl = np.array([dna_entropy(s) for s in baseline])
    ent_sp = np.array([dna_entropy(s) for s in specdec])

    # KS tests
    ks_gc, p_gc = ks_2samp(gc_bl, gc_sp)
    ks_ent, p_ent = ks_2samp(ent_bl, ent_sp)

    # K-mer spectra
    freq_bl_dict, freq_bl, kmers = kmer_frequencies(baseline, k=4)
    freq_sp_dict, freq_sp, _ = kmer_frequencies(specdec, k=4)

    # JSD between k-mer vectors
    jsd = jensenshannon(freq_bl, freq_sp)

    # Pearson correlation of k-mer frequencies
    r_kmer, p_kmer = pearsonr_np(freq_bl, freq_sp)

    # Save k-mer frequency CSVs
    kmer_df_bl = pd.DataFrame({"kmer": kmers, "frequency": freq_bl})
    kmer_df_sp = pd.DataFrame({"kmer": kmers, "frequency": freq_sp})
    kmer_df_bl.to_csv(os.path.join(output_dir, "kmer_freq_dnagpt_baseline.csv"),
                      index=False, float_format="%.8f")
    kmer_df_sp.to_csv(os.path.join(output_dir, "kmer_freq_dnagpt_spec.csv"),
                      index=False, float_format="%.8f")

    summary = {
        "metric": ["GC_content", "Nucleotide_entropy", "4mer_spectrum"],
        "baseline_mean": [gc_bl.mean(), ent_bl.mean(), np.nan],
        "baseline_std": [gc_bl.std(), ent_bl.std(), np.nan],
        "specdec_mean": [gc_sp.mean(), ent_sp.mean(), np.nan],
        "specdec_std": [gc_sp.std(), ent_sp.std(), np.nan],
        "KS_statistic": [ks_gc, ks_ent, np.nan],
        "KS_pvalue": [p_gc, p_ent, np.nan],
        "JSD": [np.nan, np.nan, jsd],
        "Pearson_r": [np.nan, np.nan, r_kmer],
        "Pearson_p": [np.nan, np.nan, p_kmer],
        "n_baseline": [len(baseline)] * 3,
        "n_specdec": [len(specdec)] * 3,
    }
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "dna_validation_summary.csv"),
                                 index=False, float_format="%.6f")

    return {
        "gc_bl": gc_bl, "gc_sp": gc_sp,
        "ent_bl": ent_bl, "ent_sp": ent_sp,
        "freq_bl": freq_bl, "freq_sp": freq_sp,
        "kmers": kmers,
        "ks_gc": ks_gc, "p_gc": p_gc,
        "ks_ent": ks_ent, "p_ent": p_ent,
        "jsd": jsd, "r_kmer": r_kmer, "p_kmer": p_kmer,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Protein statistics
# ══════════════════════════════════════════════════════════════════════════════

AA_ALPHABET = sorted(STD_AA)


def protein_entropy(seq):
    counts = Counter(seq)
    return shannon_entropy(counts, AA_ALPHABET)


def hydrophobic_fraction(seq):
    n = len(seq)
    if n == 0:
        return np.nan
    return sum(1 for c in seq if c in HYDROPHOBIC_AA) / n


def aa_frequencies(seqs):
    """Compute normalized AA frequency vector across all sequences."""
    counts = Counter()
    total = 0
    for seq in seqs:
        for c in seq:
            if c in STD_AA:
                counts[c] += 1
                total += 1
    freq = np.array([counts.get(aa, 0) for aa in AA_ALPHABET], dtype=float)
    if total > 0:
        freq /= total
    return dict(zip(AA_ALPHABET, freq)), freq


def compute_protein_stats(seqs_dict, output_dir, model="ProGen2"):
    """Compute protein statistics and comparisons. Returns summary dict."""
    baseline = seqs_dict["baseline"]
    specdec = seqs_dict["specdec"]

    # Per-sequence
    ent_bl = np.array([protein_entropy(s) for s in baseline])
    ent_sp = np.array([protein_entropy(s) for s in specdec])
    hyd_bl = np.array([hydrophobic_fraction(s) for s in baseline])
    hyd_sp = np.array([hydrophobic_fraction(s) for s in specdec])

    # KS tests
    ks_ent, p_ent = ks_2samp(ent_bl, ent_sp)
    ks_hyd, p_hyd = ks_2samp(hyd_bl, hyd_sp)

    # AA freq spectra
    freq_bl_dict, freq_bl = aa_frequencies(baseline)
    freq_sp_dict, freq_sp = aa_frequencies(specdec)

    # JSD
    jsd = jensenshannon(freq_bl, freq_sp)
    # Pearson
    r_aa, p_aa = pearsonr_np(freq_bl, freq_sp)

    # Save AA freq CSVs
    model_lower = model.lower().replace(" ", "")
    aa_df_bl = pd.DataFrame({"amino_acid": AA_ALPHABET, "frequency": freq_bl})
    aa_df_sp = pd.DataFrame({"amino_acid": AA_ALPHABET, "frequency": freq_sp})
    aa_df_bl.to_csv(os.path.join(output_dir, f"aa_freq_{model_lower}_baseline.csv"),
                    index=False, float_format="%.8f")
    aa_df_sp.to_csv(os.path.join(output_dir, f"aa_freq_{model_lower}_spec.csv"),
                    index=False, float_format="%.8f")

    return {
        "ent_bl": ent_bl, "ent_sp": ent_sp,
        "hyd_bl": hyd_bl, "hyd_sp": hyd_sp,
        "freq_bl": freq_bl, "freq_sp": freq_sp,
        "ks_ent": ks_ent, "p_ent": p_ent,
        "ks_hyd": ks_hyd, "p_hyd": p_hyd,
        "jsd": jsd, "r_aa": r_aa, "p_aa": p_aa,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — 2×2 main figure
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {"baseline": "#4C72B0", "specdec": "#DD8452"}


def plot_main_figure(dna_stats, prot_stats, output_dir):
    """Create the 2×2 biological validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

    # ── Panel A: DNAGPT GC-content distributions ──
    ax = axes[0, 0]
    bins = np.linspace(0.15, 0.65, 40)
    ax.hist(dna_stats["gc_bl"], bins=bins, alpha=0.6, color=COLORS["baseline"],
            label="Baseline", density=True, edgecolor="white", linewidth=0.3)
    ax.hist(dna_stats["gc_sp"], bins=bins, alpha=0.6, color=COLORS["specdec"],
            label="SpecDec", density=True, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("GC Content")
    ax.set_ylabel("Density")
    ax.set_title("A) DNAGPT — GC Content Distribution", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    # Annotate KS
    ax.text(0.97, 0.95,
            f"KS = {dna_stats['ks_gc']:.3f}\np = {dna_stats['p_gc']:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel B: DNAGPT 4-mer frequency scatter ──
    ax = axes[0, 1]
    ax.scatter(dna_stats["freq_bl"], dna_stats["freq_sp"],
               s=8, alpha=0.5, color="#555555", edgecolors="none")
    lim_max = max(dna_stats["freq_bl"].max(), dna_stats["freq_sp"].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "r--", lw=0.8, label="y = x")
    ax.set_xlabel("Baseline 4-mer Frequency")
    ax.set_ylabel("SpecDec 4-mer Frequency")
    ax.set_title("B) DNAGPT — 4-mer Spectrum", fontsize=10, fontweight="bold", loc="left")
    ax.text(0.03, 0.95,
            f"r = {dna_stats['r_kmer']:.4f}\nJSD = {dna_stats['jsd']:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")

    # ── Panel C: Protein AA frequency scatter (both models) ──
    ax = axes[1, 0]
    markers = {"ProGen2": "o", "ProtGPT2": "s"}
    model_colors = {"ProGen2": "#4C72B0", "ProtGPT2": "#C44E52"}
    for model, stats in prot_stats.items():
        ax.scatter(stats["freq_bl"], stats["freq_sp"],
                   s=30, alpha=0.7, marker=markers.get(model, "o"),
                   color=model_colors.get(model, "gray"), edgecolors="white",
                   linewidth=0.3, label=f"{model} (r={stats['r_aa']:.4f})")
    lim_max_p = max(
        max(s["freq_bl"].max() for s in prot_stats.values()),
        max(s["freq_sp"].max() for s in prot_stats.values()),
    ) * 1.15
    ax.plot([0, lim_max_p], [0, lim_max_p], "r--", lw=0.8)
    ax.set_xlabel("Baseline AA Frequency")
    ax.set_ylabel("SpecDec AA Frequency")
    ax.set_title("C) Protein — AA Frequency Comparison", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    # Add AA labels to the points for the larger model
    main_model = list(prot_stats.keys())[0]
    for i, aa in enumerate(AA_ALPHABET):
        ax.annotate(aa, (prot_stats[main_model]["freq_bl"][i],
                         prot_stats[main_model]["freq_sp"][i]),
                    fontsize=5.5, ha="center", va="bottom", alpha=0.6)
    ax.set_xlim(0, lim_max_p)
    ax.set_ylim(0, lim_max_p)
    ax.set_aspect("equal")

    # ── Panel D: Sequence entropy box/violin comparison ──
    ax = axes[1, 1]
    # Collect all entropy data
    all_models = []
    if dna_stats is not None:
        all_models.append(("DNAGPT", dna_stats["ent_bl"], dna_stats["ent_sp"]))
    for model, stats in prot_stats.items():
        all_models.append((model, stats["ent_bl"], stats["ent_sp"]))

    positions = []
    labels = []
    pos = 0
    for model, ent_bl, ent_sp in all_models:
        bp_bl = ax.boxplot([ent_bl], positions=[pos], widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor=COLORS["baseline"], alpha=0.7),
                           medianprops=dict(color="black"), flierprops=dict(markersize=2),
                           whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        bp_sp = ax.boxplot([ent_sp], positions=[pos + 0.4], widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor=COLORS["specdec"], alpha=0.7),
                           medianprops=dict(color="black"), flierprops=dict(markersize=2),
                           whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        labels.append(model)
        positions.append(pos + 0.2)
        pos += 1.2

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("D) Sequence Entropy — Baseline vs. SpecDec", fontsize=10, fontweight="bold", loc="left")

    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["baseline"], alpha=0.7, label="Baseline"),
                       Patch(facecolor=COLORS["specdec"], alpha=0.7, label="SpecDec")]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(output_dir, "bio_validation_figure.pdf"),
                bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "bio_validation_figure.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> bio_validation_figure.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════════════════════

def save_protein_summary(prot_stats, output_dir):
    """Save combined protein validation summary CSV."""
    rows = []
    for model, s in prot_stats.items():
        rows.append({
            "model": model,
            "metric": "AA_entropy",
            "baseline_mean": s["ent_bl"].mean(),
            "baseline_std": s["ent_bl"].std(),
            "specdec_mean": s["ent_sp"].mean(),
            "specdec_std": s["ent_sp"].std(),
            "KS_statistic": s["ks_ent"],
            "KS_pvalue": s["p_ent"],
            "JSD": np.nan,
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
            "n_baseline": len(s["ent_bl"]),
            "n_specdec": len(s["ent_sp"]),
        })
        rows.append({
            "model": model,
            "metric": "Hydrophobic_fraction",
            "baseline_mean": s["hyd_bl"].mean(),
            "baseline_std": s["hyd_bl"].std(),
            "specdec_mean": s["hyd_sp"].mean(),
            "specdec_std": s["hyd_sp"].std(),
            "KS_statistic": s["ks_hyd"],
            "KS_pvalue": s["p_hyd"],
            "JSD": np.nan,
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
            "n_baseline": len(s["hyd_bl"]),
            "n_specdec": len(s["hyd_sp"]),
        })
        rows.append({
            "model": model,
            "metric": "AA_frequency_spectrum",
            "baseline_mean": np.nan,
            "baseline_std": np.nan,
            "specdec_mean": np.nan,
            "specdec_std": np.nan,
            "KS_statistic": np.nan,
            "KS_pvalue": np.nan,
            "JSD": s["jsd"],
            "Pearson_r": s["r_aa"],
            "Pearson_p": s["p_aa"],
            "n_baseline": len(s["ent_bl"]),
            "n_specdec": len(s["ent_sp"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "protein_validation_summary.csv"),
              index=False, float_format="%.6f")
    print(f"  -> protein_validation_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(data, dna_stats, prot_stats):
    print("\n" + "=" * 78)
    print("BIOLOGICAL VALIDATION SUMMARY")
    print("=" * 78)

    # Sample sizes
    print("\nSample sizes:")
    for model, seqs in data.items():
        print(f"  {model:10s}: baseline={len(seqs['baseline']):5d}, "
              f"specdec={len(seqs['specdec']):5d}")

    # DNA
    if dna_stats:
        print("\n--- DNAGPT (DNA) ---")
        print(f"  GC content:  baseline={dna_stats['gc_bl'].mean():.4f}±{dna_stats['gc_bl'].std():.4f}  "
              f"specdec={dna_stats['gc_sp'].mean():.4f}±{dna_stats['gc_sp'].std():.4f}  "
              f"KS={dna_stats['ks_gc']:.4f} p={dna_stats['p_gc']:.2e}")
        print(f"  Entropy:     baseline={dna_stats['ent_bl'].mean():.4f}±{dna_stats['ent_bl'].std():.4f}  "
              f"specdec={dna_stats['ent_sp'].mean():.4f}±{dna_stats['ent_sp'].std():.4f}  "
              f"KS={dna_stats['ks_ent']:.4f} p={dna_stats['p_ent']:.2e}")
        print(f"  4-mer JSD:   {dna_stats['jsd']:.6f}")
        print(f"  4-mer r:     {dna_stats['r_kmer']:.6f} (p={dna_stats['p_kmer']:.2e})")

    # Proteins
    for model, s in prot_stats.items():
        print(f"\n--- {model} (Protein) ---")
        print(f"  AA entropy:  baseline={s['ent_bl'].mean():.4f}±{s['ent_bl'].std():.4f}  "
              f"specdec={s['ent_sp'].mean():.4f}±{s['ent_sp'].std():.4f}  "
              f"KS={s['ks_ent']:.4f} p={s['p_ent']:.2e}")
        print(f"  Hydro frac:  baseline={s['hyd_bl'].mean():.4f}±{s['hyd_bl'].std():.4f}  "
              f"specdec={s['hyd_sp'].mean():.4f}±{s['hyd_sp'].std():.4f}  "
              f"KS={s['ks_hyd']:.4f} p={s['p_hyd']:.2e}")
        print(f"  AA JSD:      {s['jsd']:.6f}")
        print(f"  AA freq r:   {s['r_aa']:.6f} (p={s['p_aa']:.2e})")

    print("\n" + "=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Biological validation: baseline vs speculative decoding sequences")
    parser.add_argument("--results_dir", default=".",
                        help="Root directory (parent of results/)")
    parser.add_argument("--output_dir",
                        default="analysis_final_plots_v2/biological_validation",
                        help="Output directory for plots and CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading sequences...")
    data = load_sequences(args.results_dir)
    if not data:
        print("No data loaded. Exiting.")
        return

    # --- DNA analysis ---
    dna_stats = None
    if "DNAGPT" in data:
        print("\nComputing DNA statistics (DNAGPT)...")
        dna_stats = compute_dna_stats(data["DNAGPT"], args.output_dir)

    # --- Protein analysis ---
    prot_stats = {}
    for model in ["ProGen2", "ProtGPT2"]:
        if model in data:
            print(f"\nComputing protein statistics ({model})...")
            prot_stats[model] = compute_protein_stats(data[model], args.output_dir, model)

    # --- Save protein summary ---
    if prot_stats:
        save_protein_summary(prot_stats, args.output_dir)

    # --- Plot ---
    if dna_stats and prot_stats:
        print("\nGenerating validation figure...")
        plot_main_figure(dna_stats, prot_stats, args.output_dir)
    else:
        print("  Insufficient data for full figure (need both DNA and protein)")

    # --- Console summary ---
    print_summary(data, dna_stats, prot_stats)
    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
