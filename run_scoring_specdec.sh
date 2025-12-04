#!/bin/bash

mkdir -p logs

cd ~/DNASpecDec/

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
  --prefix_token_lengths 32 \
  --target_context_values 256 \
  --draft_context_values 256 \
  --output_csv results/scoring_specdec_grid_results.csv