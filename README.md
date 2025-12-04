# DNASpecDec: Speculative Decoding for DNAGPT

This folder contains scripts to run speculative decoding experiments on DNAGPT models, perform grid sweeps for hyperparameter tuning, and compute likelihood/perplexity metrics on generated suffixes.

## Prerequisites: DNAGPT weights and code

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

All scripts assume `DNAGPT` is importable via `sys.path.append(os.path.join(os.getcwd(), 'DNAGPT'))`.

## Environment setup

We build and run primarily on ARM (linux-aarch64) with CUDA 12 available. A reproducible environment is described in `pixi.toml` and can be recreated with Pixi.

### Using Pixi (recommended)

`pixi.toml`:
- channels: conda-forge via Prefix.dev
- platforms: linux-aarch64
- system-requirements: `cuda = "12"`
- dependencies:
  - `python = 3.10.*`
  - `pytorch-gpu`
  - `transformers`
  - `matplotlib >=3.10.8,<4`

To set up and run:

```bash
# From this folder
pixi install          # create the environment for linux-aarch64 (ARM)
pixi run python --version
```

If you have multiple platforms, Pixi will resolve for your active one. Ensure CUDA 12 drivers exist if you plan to use GPU.

### Alternative: conda or pip (x86_64 or ARM)

If you prefer conda or pip, install packages equivalent to those in `pixi.toml`:

Conda (CUDA-enabled PyTorch):
```bash
conda create -n dnaspecdec python=3.10 -y
conda activate dnaspecdec
# Use conda-forge channel for consistency
conda install -c conda-forge pytorch pytorch-cuda=12.1 cuda=12 transformers matplotlib -y
```

Pip (make sure a matching CUDA toolkit / torch build is installed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Choose a torch build that matches your CUDA (e.g., cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers matplotlib
```

Notes:
- On CPU-only nodes, you can install CPU builds of PyTorch (`pip install torch` without the CUDA index URL) and run with `--device cpu`.
- Transformers is used for utilities; generation is from DNAGPT.
- Matplotlib is only needed for plotting; the core scripts run without it.

## Data inputs

Several scripts expect an hg38 CSV with at least a `seq` column and often `id`, `chrom`, `start`, `end`. In examples below, we refer to a file like `hg38_sequences.csv` placed in this folder.

## 1) Simple speculative decoding

Script: `specdec_dnagpt.py`

Run a single speculative decoding benchmark on one prompt (randomly chosen from CSV or by index). Key arguments:
- `--hg_csv` Path to CSV with genome sequences (default `hg38_sequences.csv`).
- `--hg_row_index` Row index (0-based); if negative, a random row is used.
- `--hg_prefix` Prompt prefix to prepend to the raw sequence (default `<R>`).
- `--max_prompt_tokens` Truncate encoded prompt to this many tokens (default 32).
- `--num_tokens` Number of tokens to generate after the prompt (default 256).
- `--num_samples` Repetitions for timing (default 10).
- `--temperature`, `--topk`, `--topp` Sampling controls.
- `--draft_model_name`, `--target_model_name` Model IDs (defaults: `dna_gpt0.1b_m`, `dna_gpt3b_m`).
- `--draft_weight`, `--target_weight` Paths to checkpoints (defaults: `DNAGPT/checkpoints/...`).
- `--L` Speculation window size (gamma).
- `--accept_mode` One of `prob`, `pt_gt_pd`, `match`.
- `--target_context_len`, `--draft_context_len` Optional context windows.
- `--device` `cuda` or `cpu`; `--dtype` e.g. `float16`.

Example (relative paths inside this folder):

```bash
python specdec_dnagpt.py \
  --hg_csv hg38_sequences.csv \
  --hg_row_index 42 \
  --hg_prefix "<R>" \
  --max_prompt_tokens 32 \
  --num_tokens 256 \
  --num_samples 5 \
  --draft_model_name dna_gpt0.1b_m \
  --target_model_name dna_gpt3b_m \
  --draft_weight DNAGPT/checkpoints/dna_gpt0.1b_m.pth \
  --target_weight DNAGPT/checkpoints/dna_gpt3b_m.pth \
  --L 4
```

All defaults point to sensible locations assuming `DNAGPT/` exists here and `hg38_sequences.csv` is present.

## 2) Grid sweep for hyperparameter tuning and statistics

Scripts:
- `run_scoring_specdec.sh` (SLURM job wrapper)
- `scoring_specdec_grid_search_all.py` (multi-prompt, multi-hyperparam sweep)
- `scoring_specdec_beam_search.py` (core benchmarking utilities with optional lookahead)

The shell script demonstrates running the full sweep with Pixi:

```bash
bash run_scoring_specdec.sh
```

Inside it, the main call is equivalent to:

```bash
pixi run python scoring_specdec_grid_search_all.py \
  --hg_csv hg38_sequences.csv \
  --hg_prefix "<R>" \
  --hg_row_indices 5 \
  --num_tokens 256 \
  --num_samples 3 \
  --draft_model_name dna_gpt0.1b_m \
  --target_model_name dna_gpt3b_m \
  --draft_weight DNAGPT/checkpoints/dna_gpt0.1b_m.pth \
  --target_weight DNAGPT/checkpoints/dna_gpt3b_m.pth \
  --device cuda \
  --dtype float16 \
  --L_values 4 \
  --sweep_mode core \
  --prefix_token_lengths 32,64 \
  --target_context_values 256,512 \
  --draft_context_values 256 \
  --output_csv results/scoring_specdec_grid_results_testing_8.csv
```

What you can sweep:
- Prompts: `--hg_row_indices` or `--hg_ids`, otherwise random sampling via `--num_prompts`.
- SpecDec hyperparams: `--L_values`, `--accept_mode` (currently `prob` by default).
- Sampling: `temperature`, `top_k`, `top_p` depending on `--sweep_mode` (`core`, `lookahead`, or `all`).
- Prefix length (tokens): `--prefix_token_lengths` truncates the prompt per-rows.
- Context windows: `--target_context_values`, `--draft_context_values`.
- Optional lookahead knobs are wired through `scoring_specdec_beam_search.py` (width/depth kept simple by design).

Outputs:
- A CSV at `--output_csv` with per-combination metrics and example generated suffixes.
- Columns include: prompt metadata, hyperparams, TPS for target/draft/specdec, speedup, acceptance rate, totals, and `sample_*_suffix` text fields for the first sample.

## 3) Likelihood scoring (perplexity etc.)

Script: `likelihood_scoring.py`

Use after the sweep to augment the results CSV with log-prob, NLL, and perplexity metrics computed under a chosen DNAGPT model.

Required input columns (if present, they will be scored):
- `sample_target_suffix`
- `sample_specdec_suffix`
- `sample_draft_suffix` (optional)

Key arguments:
- `input_csv` Path to the sweep results.
- `--output_csv` Output path (default: `<input>_scored.csv`).
- `--model_name` Model used for scoring (default: `dna_gpt3b_m`).
- `--weight_path` Checkpoint path (default: `DNAGPT/checkpoints/dna_gpt3b_m.pth`).
- `--device`, `--dtype` Torch device and dtype.
- `--reduce` `mean` or `sum` over token log-probs.

Example:

```bash
python likelihood_scoring.py \
  results/scoring_specdec_grid_results.csv \
  --model_name dna_gpt3b_m \
  --weight_path DNAGPT/checkpoints/dna_gpt3b_m.pth \
  --device cuda \
  --dtype float16 \
  --reduce mean
```

This will add columns:
- `<prefix>_logprob_mean` and `<prefix>_nll_mean` for target/specdec/draft suffixes.
- `<prefix>_ppl` (perplexity via exp(-mean_logprob)).

## Tips and troubleshooting

- If imports fail, verify `DNAGPT/` is cloned next to these scripts and that checkpoints exist.
- For ARM machines, confirm the CUDA 12 runtime and drivers match the PyTorch build.
- CSV format: ensure the input CSV has a header and at least a `seq` column. Optional columns (`id`, `chrom`, `start`, `end`) enable nicer metadata in outputs.

## File index

- `specdec_dnagpt.py` — single-prompt speculative decoding benchmark with configurable acceptance.
- `scoring_specdec_beam_search.py` — shared helpers and the core benchmark runner (includes optional lookahead logic).
- `scoring_specdec_grid_search_all.py` — grid sweep over prompts and hyperparameters; writes results CSV.
- `run_scoring_specdec.sh` — SLURM job script to launch the sweep using Pixi.
- `likelihood_scoring.py` — offline scoring of generated suffixes; outputs logprob/NLL/perplexity.
- `pixi.toml` — environment specification for ARM (linux-aarch64) with CUDA 12.
