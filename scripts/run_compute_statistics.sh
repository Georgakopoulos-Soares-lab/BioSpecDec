#!/usr/bin/env bash
#SBATCH -J stats_wide
#SBATCH -o stats_wide.%j.out
#SBATCH -e stats_wide.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -t 0:30:00

# Computes grouped statistics for multiple wide CSV outputs.
#
# Usage:
#   bash scripts/run_compute_statistics.sh [csv1 csv2 ...]
#
# Defaults (if no args):
#   results/dnagpt_hg38_wide_2.csv
#   results/protgpt2_wide.csv
#   results/progen2_wide.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
	echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
	exit 2
fi

mkdir -p results/statistics

if [[ "$#" -gt 0 ]]; then
	INPUTS=("$@")
else
	INPUTS=(
		# Prefer scored+filtered DNAGPT results (contains *_ppl columns).
		# Falls back to the older unscored file if the scored file isn't present.
		"results/dnagpt_final_scored_filtered.csv"
		"results/dnagpt_hg38_wide_2.csv"
		# Prefer scored ProtGPT2 results (contains *_ppl columns).
		"results/protgpt2_wide_scored.csv"
		"results/protgpt2_wide.csv"
		"results/progen2_final_final.csv",
		"results/progen2_final_final_scored.csv"
	)
fi

# Remove any defaults that don't exist to avoid failing on new workspaces.
EXISTING_INPUTS=()
for p in "${INPUTS[@]}"; do
	if [[ -f "$p" ]]; then
		EXISTING_INPUTS+=("$p")
	fi
done
INPUTS=("${EXISTING_INPUTS[@]}")

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
	echo "[ERROR] No default input CSVs found. Pass files as args." >&2
	exit 3
fi

echo "[INFO] Writing grouped stats into results/statistics"
python scripts/compute_grouped_statistics.py --output_dir results/statistics --inputs "${INPUTS[@]}"
