#!/usr/bin/env bash
#SBATCH -J specdec_protein
#SBATCH -o specdec_protein.%j.out
#SBATCH -e specdec_protein.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH -t 48:00:00

set -euo pipefail

# Helps reduce allocator fragmentation for long-running GPU jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v python >/dev/null 2>&1; then
	echo "[ERROR] python not found on PATH. Activate your environment (conda/venv) and try again." >&2
	exit 2
fi

mkdir -p results logs

# Protein: multi-GPU sweep (sharded) + merge + likelihood scoring.
#
# Usage:
#   bash scripts/run_protein_specdec_multigpu_and_score.sh <config.json> <out.csv> <model_family> <model_name>
#
# Examples:
#   bash scripts/run_protein_specdec_multigpu_and_score.sh \
#     sweeps/protein/protgpt2_speed_sweep.json results/protgpt2_wide.csv protgpt2 nferruz/ProtGPT2
#
#   bash scripts/run_protein_specdec_multigpu_and_score.sh \
#     sweeps/protein/progen2_speed_sweep.json results/progen2_wide.csv progen2 Salesforce/progen2-medium

CFG="${1:-sweeps/protein/progen2_speed_sweep.json}"
OUT_CSV="${2:-results/progen2_wide_final.csv}"

# If MODEL_FAMILY / MODEL_NAME are not provided, infer from the config.
# This avoids a common ProGen2 failure mode: scoring with a different vocab than the IDs in the CSV.
MODEL_FAMILY="${3:-}"
MODEL_NAME="${4:-}"

if [[ -z "$MODEL_FAMILY" || -z "$MODEL_NAME" ]]; then
	if [[ ! -f "$CFG" ]]; then
		echo "[WARN] Config not found at '$CFG'; cannot infer MODEL_FAMILY/MODEL_NAME." >&2
		INFERRED=""
	else
		if INFERRED="$(CFG_PATH="$CFG" python - <<'PY'
import json
import os
import sys

cfg_path = os.environ.get("CFG_PATH", "")
if not cfg_path:
    # Fail soft: caller will use defaults/CLI args
    print("\t")
    sys.exit(0)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

base = cfg.get("base", {})
mf = base.get("model_family", "")
tm = base.get("target_model_name", "")
print(f"{mf}\t{tm}")
PY
		)"; then
			:
		else
			echo "[WARN] Failed to infer MODEL_FAMILY/MODEL_NAME from '$CFG'; using defaults/CLI args." >&2
			INFERRED=""
		fi
	fi
	INFERRED_FAMILY="${INFERRED%%$'\t'*}"
	INFERRED_MODEL="${INFERRED#*$'\t'}"
	if [[ -z "$MODEL_FAMILY" && -n "$INFERRED_FAMILY" ]]; then
		MODEL_FAMILY="$INFERRED_FAMILY"
	fi
	if [[ -z "$MODEL_NAME" && -n "$INFERRED_MODEL" && "$INFERRED_MODEL" != "$INFERRED" ]]; then
		MODEL_NAME="$INFERRED_MODEL"
	fi
fi

# Fallback defaults (kept for backwards compatibility)
MODEL_FAMILY="${MODEL_FAMILY:-progen2}"
MODEL_NAME="${MODEL_NAME:-hugohrban/progen2-base}"

echo "[INFO] CFG=$CFG"
echo "[INFO] OUT_CSV=$OUT_CSV"
echo "[INFO] MODEL_FAMILY=$MODEL_FAMILY"
echo "[INFO] MODEL_NAME=$MODEL_NAME"

if [[ "$MODEL_FAMILY" == "progen2" ]]; then
	echo "[INFO] ProGen2 note: scoring MODEL_NAME should match the tokenizer/vocab used to generate IDs (usually base.target_model_name in the config)."
fi

# Run sharded sweep and merge shards into OUT_CSV
bash scripts/run_protein_specdec_multigpu.sh "$CFG" "" "$OUT_CSV"

python -m pipeline.score_likelihoods \
	--input_csv "$OUT_CSV" \
	--output_csv "${OUT_CSV%.csv}_scored.csv" \
	--score_model_family "$MODEL_FAMILY" \
	--model_name "$MODEL_NAME" \
	--device "${DEVICE:-cuda}" \
	--dtype "${DTYPE:-float16}" \
	--reduce mean
