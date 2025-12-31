#!/usr/bin/env bash
#SBATCH -J specdec_dna
#SBATCH -o specdec_dna.%j.out
#SBATCH -e specdec_dna.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
  exit 2
fi

# mkdir -p logs results

# DNAGPT: multi-GPU sweep (sharded) + merge + likelihood scoring.
#
# Usage:
#   bash scripts/run_dna_specdec_multigpu_and_score.sh <config.json> <out.csv> [model_name] [weight_path]
#
# Example:
#   bash scripts/run_dna_specdec_multigpu_and_score.sh \
#     sweeps/dnagpt/dnagpt_hg38_sweep.json results/dnagpt_hg38_wide.csv \
#     dna_gpt3b_m DNAGPT/checkpoints/dna_gpt3b_m.pth

CFG="${1:-sweeps/dnagpt/dnagpt_hg38_sweep.json}"
OUT_CSV="${2:-results/dnagpt_hg38_wide.csv}"
MODEL_NAME="${3:-dna_gpt3b_m}"
WEIGHT_PATH="${4:-DNAGPT/checkpoints/dna_gpt3b_m.pth}"

bash scripts/run_dna_specdec_multigpu.sh "$CFG" "$OUT_CSV"

python -m pipeline.score_likelihoods \
  --input_csv "$OUT_CSV" \
  --output_csv "${OUT_CSV%.csv}_scored.csv" \
  --score_model_family dnagpt \
  --model_name "$MODEL_NAME" \
  --weight_path "$WEIGHT_PATH" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float16}" \
  --reduce mean
