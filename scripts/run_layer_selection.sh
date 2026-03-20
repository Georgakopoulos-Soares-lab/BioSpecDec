#!/usr/bin/env bash
#SBATCH -J layer_select_ablation
#SBATCH -o logs/layer_select_%j.out
#SBATCH -e logs/layer_select_%j.err
#SBATCH -p gpu-a100-small
#SBATCH -A BCS25073
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH -t 48:00:00

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs results/layer_selection

# ── Activate environment ──
# Adjust this to match your setup. Examples:
#   source .venv/bin/activate
#   conda activate biospecdec
#   module load python3/3.10
if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found. Activate your environment and try again." >&2
  exit 2
fi

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"
echo "GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# ── Which models to run ──
# Options: protgpt2, progen2, both
MODEL="${1:-both}"
NUM_SAMPLES="${2:-3}"

echo "Running layer-selection ablation: model=$MODEL, num_samples=$NUM_SAMPLES"
echo "Start: $(date)"
echo ""

python scripts/layer_selection_experiment.py \
  --model "$MODEL" \
  --device cuda \
  --dtype float16 \
  --num_samples "$NUM_SAMPLES" \
  --output_dir results/layer_selection

echo ""
echo "Experiment done. Generating plots..."

python scripts/plot_layer_selection.py \
  --input_dir results/layer_selection \
  --output_dir results/layer_selection/plots

echo ""
echo "All done: $(date)"
echo "Results: results/layer_selection/"
echo "Plots:   results/layer_selection/plots/"
