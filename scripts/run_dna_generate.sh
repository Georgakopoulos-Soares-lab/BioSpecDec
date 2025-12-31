#!/usr/bin/env bash
#SBATCH -J dnagpt_one
#SBATCH -o dnagpt_one.%j.out
#SBATCH -e dnagpt_one.%j.err
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

# One-off (non-sweep) runner for DNAGPT using the unified pipeline runner.
#
# Usage:
#   bash scripts/run_dna_generate.sh <prompt> [out.jsonl] [out.csv] [extra pipeline.run_generate args...]
#
# Examples:
#   bash scripts/run_dna_generate.sh "<R>ACGTACGT" results/dnagpt_one.jsonl results/dnagpt_one_summary.csv
#
#   METHOD=target_baseline TEMPERATURE=1.0 TOPP=0.95 NUM_TOKENS=256 bash scripts/run_dna_generate.sh \
#     "<R>ACGTACGT" results/dnagpt_one.jsonl results/dnagpt_one_summary.csv
#
# Notes:
# - If you want to generate from an hg38 row, use scripts/run_dna_one.sh (hg38_sequences.csv).

PROMPT="${1:-<R>ACGT}"
OUT_JSONL="${2:-results/dnagpt_one.jsonl}"
OUT_CSV="${3:-results/dnagpt_one_summary.csv}"
EXTRA_ARGS=("${@:4}")

METHOD="${METHOD:-specdec}"
NUM_TOKENS="${NUM_TOKENS:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK="${TOPK:-0}"
TOPP="${TOPP:-0.95}"
GAMMA="${GAMMA:-5}"
ACCEPT_MODE="${ACCEPT_MODE:-prob}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
TARGET_CONTEXT_LEN="${TARGET_CONTEXT_LEN:-}"
DRAFT_CONTEXT_LEN="${DRAFT_CONTEXT_LEN:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"

DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-dna_gpt0.1b_m}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-dna_gpt3b_m}"
DRAFT_WEIGHT="${DRAFT_WEIGHT:-DNAGPT/checkpoints/dna_gpt0.1b_m.pth}"
TARGET_WEIGHT="${TARGET_WEIGHT:-DNAGPT/checkpoints/dna_gpt3b_m.pth}"

ARGS=(
  -m pipeline.run_generate
  --model_family dnagpt
  --method "$METHOD"
  --prompt "$PROMPT"
  --num_tokens "$NUM_TOKENS"
  --num_samples "$NUM_SAMPLES"
  --temperature "$TEMPERATURE"
  --top_k "$TOPK"
  --top_p "$TOPP"
  --gamma "$GAMMA"
  --accept_mode "$ACCEPT_MODE"
  --target_model_name "$TARGET_MODEL_NAME"
  --draft_model_name "$DRAFT_MODEL_NAME"
  --target_weight "$TARGET_WEIGHT"
  --draft_weight "$DRAFT_WEIGHT"
  --max_prompt_tokens "$MAX_PROMPT_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --output_jsonl "$OUT_JSONL"
  --output_csv "$OUT_CSV"
)

if [[ -n "$TARGET_CONTEXT_LEN" ]]; then
  ARGS+=(--target_context_len "$TARGET_CONTEXT_LEN")
fi
if [[ -n "$DRAFT_CONTEXT_LEN" ]]; then
  ARGS+=(--draft_context_len "$DRAFT_CONTEXT_LEN")
fi

python "${ARGS[@]}" "${EXTRA_ARGS[@]}"
