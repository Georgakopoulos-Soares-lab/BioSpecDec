#!/usr/bin/env bash
#SBATCH -J score_dna_wide
#SBATCH -o score_dna_wide.%j.out
#SBATCH -e score_dna_wide.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH -t 24:00:00

# DNAGPT wide-CSV scoring wrapper.
#
# Usage:
#   bash scripts/run_dna_score_wide.sh <input_wide.csv> [output_scored.csv] [model_name] [weight_path]
#
# Example:
#   bash scripts/run_dna_score_wide.sh \
#     results/dnagpt_hg38_wide.csv results/dnagpt_hg38_wide_scored.csv \
#     dna_gpt3b_m DNAGPT/checkpoints/dna_gpt3b_m.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
  exit 2
fi

IN_CSV="${1:-results/dnagpt_final.csv}"
OUT_CSV="${2:-}"
MODEL_NAME="${3:-dna_gpt3b_m}"
WEIGHT_PATH="${4:-DNAGPT/checkpoints/dna_gpt3b_m.pth}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
REDUCE="${REDUCE:-mean}"

if [[ -z "$OUT_CSV" ]]; then
	OUT_CSV="${IN_CSV%.csv}_scored.csv"
fi

echo "[INFO] IN_CSV=$IN_CSV"
echo "[INFO] OUT_CSV=$OUT_CSV"
echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] WEIGHT_PATH=$WEIGHT_PATH"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] DTYPE=$DTYPE"
echo "[INFO] REDUCE=$REDUCE"

python -m pipeline.score_likelihoods \
  --input_csv "$IN_CSV" \
  --output_csv "$OUT_CSV" \
  --score_model_family dnagpt \
  --model_name "$MODEL_NAME" \
  --weight_path "$WEIGHT_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --reduce "$REDUCE"
