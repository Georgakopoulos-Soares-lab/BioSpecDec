#!/usr/bin/env bash
#SBATCH -J specdec_protein
#SBATCH -o specdec_protein.%j.out
#SBATCH -e specdec_protein.%j.err
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

mkdir -p results logs

# Multi-GPU launcher for protein sweeps (pipeline.sweep_wide).
# Spawns one process per visible GPU and shards the sweep across them.
#
# Usage:
#   bash scripts/run_protein_specdec_multigpu.sh <config.json> [out.jsonl] [out.csv]
#
# Example:
#   bash scripts/run_protein_specdec_multigpu.sh sweeps/protein/progen2_speed_sweep.json \
#     "" results/progen2_wide.csv

CFG="${1:-sweeps/protein/progen2_minisweep.json}"
OUT_JSONL="${2:-}"
OUT_CSV="${3:-results/protein_wide.csv}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NGPU="${#GPUS[@]}"
else
  NGPU="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

if [[ "$NGPU" -lt 1 ]]; then
  echo "[ERROR] No GPUs visible." >&2
  exit 2
fi

echo "[INFO] Launching $NGPU shards"

pids=()
for ((i=0; i<NGPU; i++)); do
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_ID="${GPUS[$i]}"
  else
    GPU_ID="$i"
  fi

  SHARD_JSONL=""
  if [[ -n "$OUT_JSONL" ]]; then
    SHARD_JSONL="${OUT_JSONL%.jsonl}.shard${i}.jsonl"
  fi
  SHARD_CSV="${OUT_CSV%.csv}.shard${i}.csv"

  echo "[INFO] Shard $i/$NGPU -> GPU $GPU_ID"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python -m pipeline.sweep_wide \
      --config "$CFG" \
      --output_csv "$SHARD_CSV" \
      ${SHARD_JSONL:+--output_jsonl "$SHARD_JSONL"} \
      --num_shards "$NGPU" \
      --shard_idx "$i"
  ) &
  pids+=("$!")
done

# Wait (and fail-fast if any shard fails)
failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    echo "[ERROR] Shard $i exited non-zero (pid=$pid)." >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] One or more shards failed; refusing to merge partial CSVs." >&2
  exit 3
fi

# Merge CSV shards (same header)
python - <<PY
import glob, os
out = "$OUT_CSV"
parts = sorted(glob.glob("${OUT_CSV%.csv}.shard*.csv"))
if not parts:
    raise SystemExit("No shard CSVs found")

header = None
rows = []
for p in parts:
    with open(p, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines:
        continue
    if header is None:
        header = lines[0]
    elif lines[0] != header:
        raise SystemExit(f"Header mismatch in {p}")
    rows.extend(lines[1:])

os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(header + "\n")
    for r in rows:
        f.write(r + "\n")

print(f"Wrote merged CSV: {out} ({len(rows)} rows)")
PY

if [[ -n "$OUT_JSONL" ]]; then
  echo "[INFO] Per-sample JSONL was written as shards: ${OUT_JSONL%.jsonl}.shard*.jsonl"
fi
