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

mkdir -p logs results

# Multi-GPU launcher for DNAGPT hg38 sweeps.
# Spawns one process per visible GPU and shards the sweep across them.
#
# Usage:
#   bash scripts/run_dna_specdec_multigpu.sh <config.json> <out.csv>
#
# Example:
#   bash scripts/run_dna_specdec_multigpu.sh sweeps/dnagpt/dnagpt_hg38_sweep.json results/dnagpt_wide.csv

CFG="${1:-sweeps/dnagpt/dnagpt_hg38_sweep.json}"
OUT_CSV="${2:-results/dnagpt_hg38_wide.csv}"

# Decide which GPUs to use.
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
  echo "[ERROR] No GPUs visible (CUDA_VISIBLE_DEVICES not set and torch.cuda.device_count()==0)." >&2
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

  # Use separate intermediate files to avoid write conflicts.
  SHARD_CSV="${OUT_CSV%.csv}.shard${i}.csv"

  echo "[INFO] Shard $i/$NGPU -> GPU $GPU_ID"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python dnagpt_hg38_sweep.py \
      --config "$CFG" \
      --output_csv "$SHARD_CSV" \
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

# Merge shard CSVs (same header)
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


