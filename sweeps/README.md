# Sweep configs

Protein sweep JSON files are inputs to `pipeline/sweep_wide`.

DNAGPT sweep JSON files are inputs to `dnagpt_hg38_sweep.py` (via `scripts/run_dna_specdec_multigpu.sh`).

- `dnagpt/`: DNAGPT sweep configs (hg38 prompt sampling via CSV)
- `protein/`: ProGen2 + ProtGPT2 sweep configs

Example:

```bash
bash scripts/run_protein_specdec_multigpu.sh sweeps/protein/progen2_minisweep.json \
  "" results/progen2_wide.csv

# (Optional) score likelihood/perplexity for the wide CSV
bash scripts/run_protein_specdec_multigpu_and_score.sh sweeps/protein/progen2_minisweep.json \
  results/progen2_wide.csv progen2 hugohrban/progen2-base
```
