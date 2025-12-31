#!/usr/bin/env bash
#SBATCH -J specdec_protein_one
#SBATCH -o specdec_protein_one.%j.out
#SBATCH -e specdec_protein_one.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH -t 2:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
  exit 2
fi

mkdir -p results

# One-off (non-sweep) runner for protein models.
#
# Usage:
#   bash scripts/run_protein_one.sh <progen2|protgpt2> <prompt> [out.jsonl] [out.csv] [extra pipeline.run_generate args...]
#
# Examples:
#   bash scripts/run_protein_one.sh progen2 "1M" results/progen2_one.jsonl results/progen2_one.csv
#
#   bash scripts/run_protein_one.sh protgpt2 "<|endoftext|>M" \
#     results/protgpt2_one.jsonl results/protgpt2_one.csv --draft_layers 4 --gamma 4

FAMILY="${1:-progen2}"
PROMPT="${2:-1M}"
OUT_JSONL="${3:-results/${FAMILY}_one.jsonl}"
OUT_CSV="${4:-results/${FAMILY}_one_summary.csv}"

EXTRA_ARGS=("${@:5}")

METHOD="${METHOD:-specdec}"
NUM_TOKENS="${NUM_TOKENS:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-0.0}"
GAMMA="${GAMMA:-5}"
ACCEPT_MODE="${ACCEPT_MODE:-prob}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"

TARGET_MODEL_NAME=""
DRAFT_MODEL_NAME=""
PROGEN2_DRAFT_MODE="${PROGEN2_DRAFT_MODE:-pretrained}"
DRAFT_LAYERS_DEFAULT="6"

case "$FAMILY" in
  progen2)
    TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-hugohrban/progen2-base}"
    DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-hugohrban/progen2-small}"
    ;;
  protgpt2)
    TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-nferruz/ProtGPT2}"
    # The pipeline uses this sentinel to indicate "truncate the target model".
    DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-(truncated)}"
    ;;
  *)
    echo "[ERROR] Unknown family '$FAMILY' (expected progen2 or protgpt2)" >&2
    exit 2
    ;;
esac

python -m pipeline.run_generate \
  --model_family "$FAMILY" \
  --method "$METHOD" \
  --prompt "$PROMPT" \
  --num_tokens "$NUM_TOKENS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --gamma "$GAMMA" \
  --accept_mode "$ACCEPT_MODE" \
  --target_model_name "$TARGET_MODEL_NAME" \
  --draft_model_name "$DRAFT_MODEL_NAME" \
  --progen2_draft_mode "$PROGEN2_DRAFT_MODE" \
  --draft_layers "${DRAFT_LAYERS:-$DRAFT_LAYERS_DEFAULT}" \
  --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output_jsonl "$OUT_JSONL" \
  --output_csv "$OUT_CSV" \
  "${EXTRA_ARGS[@]}"
