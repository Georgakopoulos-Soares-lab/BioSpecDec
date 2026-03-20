# BioSpecDec

BioSpecDec is a benchmarking and analysis toolkit for speculative decoding on genomic and protein language models (DNAGPT, ProGen2, ProtGPT2). It provides multi-GPU sweep runners for large hyperparameter grids, offline likelihood/perplexity scoring to compare speculative vs. baseline decoding, and utilities to compute grouped statistics and generate paper-ready figures. The code is organized so you can easily reproduce the experiments from our manuscript, run new sweeps on your own prompts, or perform quick one-prompt sanity checks for any supported model.

This repository contains:
- Multi-GPU sweep runners (sharded across GPUs) that write **wide CSVs**.
- Offline likelihood/perplexity scoring that augments those CSVs.
- Grouped statistics + plotting scripts used to generate paper-ready figures.

The “current implementation” is:
- All wrappers in scripts/
- dnagpt_hg38_sweep.py (DNAGPT hg38 sweep runner)
- likelihood_scoring.py (DNAGPT CSV scoring)
- scoring_specdec_beam_search.py (core benchmark runner used by the sweep)
- pipeline/ (protein/progen2/protgpt2 utilities)

## 0) Environment (venv)

All shell scripts call `python` directly, so you must activate an environment first.

### Create venv

```bash
cd /path/to/BioSpecDec

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install PyTorch (pick ONE option)

PyTorch is not pinned in requirements.txt because the correct build depends on your machine.

CPU-only:

```bash
pip install torch
```

CUDA (example for CUDA 12.1 wheels):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Notes:
- If you want ProGen2 model-sharding with `--device_map auto`, you may need `pip install accelerate` (already included in requirements.txt).

### Optional: conda

If you prefer conda, an example environment file is provided in environment.yml.

## 1) DNAGPT dependency (code + weights)

You need the DNAGPT repository and checkpoints available relative to this folder. Clone the upstream repo.

```bash
git clone https://github.com/TencentAILabHealthcare/DNAGPT.git
```

Place it at `./DNAGPT` (same level as the scripts here) and ensure checkpoints exist at:
- `DNAGPT/checkpoints/dna_gpt0.1b_h.pth`
- `DNAGPT/checkpoints/dna_gpt0.1b_m.pth`
- `DNAGPT/checkpoints/dna_gpt3b_m.pth`

To download these checkpoints with gdown (from Google Drive):

```bash
cd DNAGPT
mkdir -p checkpoints
cd checkpoints

pip install gdown
gdown --id 15m6CH3zaMSqflOaf6ec5VPfiulg-Gh0u -O dna_gpt0.1b_h.pth
gdown --id 1C0BRXfz7RNtCSjSY1dKQeR1yP7I3wTyx -O dna_gpt0.1b_m.pth
gdown --id 1pQ3Ai7C-ObzKkKTRwuf6eshVneKHzYEg -O dna_gpt3b_m.pth
```

The DNAGPT runner imports DNAGPT via the repo-local path (see the code in scoring_specdec_beam_search.py).

## 2) hg38 input file (hg38_sequences.csv)

The DNAGPT hg38 sweep reads prompts from:

hg38_sequences.csv

Expected CSV columns:
- seq (required): raw DNA sequence (bases)
- id (optional): unique identifier used as prompt_id/hg_id in outputs
- chrom, start, end (optional): genomic coordinates; included in output for provenance

How it is used:
- dnagpt_hg38_sweep.py reads rows and constructs a prompt string: prompt_text = hg_prefix + seq
- hg_prefix defaults to "<R>" and is configurable via the sweep JSON.
- The prompt is tokenized and optionally truncated to max_prompt_tokens.

## 3) DNAGPT hg38 sweep (multi-GPU)

Primary script you used:

```bash
bash scripts/run_dna_specdec_multigpu_and_score.sh \
  sweeps/dnagpt/dnagpt_hg38_sweep.json \
  results/dnagpt_hg38_wide.csv
```

What this does:
1) Runs a sharded sweep across GPUs via scripts/run_dna_specdec_multigpu.sh.
2) Merges the shard CSVs into a single wide CSV.
3) Runs DNAGPT likelihood scoring on the merged wide CSV and writes results/dnagpt_hg38_wide_scored.csv.

GPU selection:
- If CUDA_VISIBLE_DEVICES is set, the wrapper uses those GPUs.
- Otherwise it uses torch.cuda.device_count().

### Sweep configs (JSON)

The DNAGPT sweep config is a JSON file with three main sections:
- (Some people informally call these “YAML sweep files”; in this repo they are JSON.)
- base: fixed parameters (models, weights, device/dtype, etc.)
- grid: lists of hyperparameters to sweep (cartesian product)
- prompt_source: how prompts are sampled from hg38_sequences.csv

Relevant prompt_source keys (see dnagpt_hg38_sweep.py):
- hg_csv (default: hg38_sequences.csv)
- hg_prefix (default: <R>)
- num_prompts
- hg_row_indices (explicit row indices) or hg_ids (explicit ids)
- seed

## 4) Protein sweeps (ProGen2 / ProtGPT2) (multi-GPU)

Protein sweeps run through pipeline/ and are launched with:

```bash
bash scripts/run_protein_specdec_multigpu.sh \
  sweeps/protein/protgpt2_speed_sweep.json \
  "" \
  results/protgpt2_wide.csv
```

To run sweep + likelihood scoring:

```bash
bash scripts/run_protein_specdec_multigpu_and_score.sh \
  sweeps/protein/protgpt2_speed_sweep.json \
  results/protgpt2_wide.csv \
  protgpt2 nferruz/ProtGPT2
```

## 5) Grouped statistics + plots

After you have wide CSVs (optionally scored), compute grouped stats:

```bash
bash scripts/run_compute_statistics.sh \
  results/dnagpt_hg38_wide_scored.csv \
  results/protgpt2_wide_scored.csv
```

Then generate plots:

```bash
bash scripts/run_plot_statistics.sh
```

## 6) One-prompt runs (instant sanity checks)

One-prompt runs use the unified runner in pipeline/run_generate.py.

It supports three model families:
- protgpt2 (HuggingFace)
- progen2 (HuggingFace, trust_remote_code)
- dnagpt (local DNAGPT repo + checkpoint .pth files)

The unified CLI always writes:
- output_jsonl: per-sample records (prompt ids, generated ids, speed, acceptance rate)
- output_csv: one-row summary of averages

### Protein one prompt (ProtGPT2 / ProGen2)

Convenience wrapper:

```bash
bash scripts/run_protein_one.sh protgpt2 "<|endoftext|>M" \
  results/protgpt2_one.jsonl results/protgpt2_one_summary.csv
```

Direct CLI (same underlying runner):

```bash
python -m pipeline.run_generate \
  --model_family protgpt2 \
  --method specdec \
  --prompt "<|endoftext|>M" \
  --num_tokens 256 \
  --num_samples 1 \
  --temperature 1.0 \
  --top_k 0 \
  --top_p 0.0 \
  --gamma 5 \
  --accept_mode prob \
  --target_model_name nferruz/ProtGPT2 \
  --draft_model_name "(truncated)" \
  --draft_layers 6 \
  --device cuda \
  --dtype float16 \
  --output_jsonl results/protgpt2_one.jsonl \
  --output_csv results/protgpt2_one_summary.csv
```

For ProGen2, set model_family/progen2 names, and optionally `--device_map auto` if you want HF sharding.

### DNAGPT one prompt (custom prompt + temperature)

Convenience wrapper:

```bash
TEMPERATURE=1.0 TOPP=0.95 NUM_TOKENS=256 METHOD=specdec bash scripts/run_dna_generate.sh \
  "<R>ACGTACGTACGT" \
  results/dnagpt_one.jsonl results/dnagpt_one_summary.csv
```

Direct CLI:

```bash
python -m pipeline.run_generate \
  --model_family dnagpt \
  --method specdec \
  --prompt "<R>ACGTACGTACGT" \
  --num_tokens 256 \
  --num_samples 1 \
  --temperature 1.0 \
  --top_k 0 \
  --top_p 0.95 \
  --gamma 5 \
  --accept_mode prob \
  --target_model_name dna_gpt3b_m \
  --draft_model_name dna_gpt0.1b_m \
  --target_weight DNAGPT/checkpoints/dna_gpt3b_m.pth \
  --draft_weight DNAGPT/checkpoints/dna_gpt0.1b_m.pth \
  --device cuda \
  --dtype float16 \
  --output_jsonl results/dnagpt_one.jsonl \
  --output_csv results/dnagpt_one_summary.csv
```

## 7) Layer selection ablation (reviewer 2.a.1)

Compares three layer selection strategies for constructing truncated draft models: **First-N** (bottom layers), **Last-N** (top layers), and **Mixed** (evenly-spaced). Evaluated on ProtGPT2 and ProGen2-xlarge across 3–6 draft layers.

### Run the experiment

```bash
python scripts/layer_selection_experiment.py --model both --device cuda
```

This produces per-sample CSVs in `results/layer_selection/`.

### Generate summary + plots

```bash
python scripts/create_layer_selection_summary.py
python scripts/plot_layer_selection.py
```

Outputs:
- `results/layer_selection/layer_selection_summary.csv` — mean ± std per (model, strategy, n_draft_layers)
- `results/layer_selection_plots/layer_strategy_comparison.png` — bar chart (average across layer counts)
- `results/layer_selection_plots/layer_strategy_lines.png` — line plot (by number of layers)
