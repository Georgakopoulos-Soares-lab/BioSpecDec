#!/usr/bin/env bash
#SBATCH -J plot_stats
#SBATCH -o plot_stats.%j.out
#SBATCH -e plot_stats.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH -t 0:20:00

# Plot grouped sweep statistics into results/statistics/plots.
#
# Usage:
#   bash scripts/run_plot_statistics.sh [grouped_stats.csv ...]
#
# If no args are provided, it plots all results/statistics/*_grouped_stats.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
	echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
	exit 2
fi

mkdir -p results/statistics/plots

if [[ "$#" -gt 0 ]]; then
	python scripts/plot_grouped_statistics.py --requested_only --output_dir results/statistics/plots --inputs "$@"
else
	python scripts/plot_grouped_statistics.py --requested_only --output_dir results/statistics/plots
fi

echo "[INFO] Wrote plots under results/statistics/plots"
