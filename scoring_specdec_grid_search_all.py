import argparse
import os
import sys
import csv
import random
import time

import torch

# Make DNAGPT importable (same as in your main script)
sys.path.append(os.path.join(os.getcwd(), 'DNAGPT'))

from dna_gpt.utils import seed_all_rng

# Import the core benchmark helpers from your main script
from scoring_specdec_beam_search import (
    get_model,
    load_model,
    run_benchmarks_for_prompt,
)


def parse_int_list(s: str):
    """Parse comma-separated integer values like '3,4,5' into [3, 4, 5]."""
    vals = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid search for DNAGPT speculative decoding hyperparameters "
            "(including L, temperature/top-k/top-p, lookahead width/depth, "
            "prefix length, and context size) over multiple hg38 prompts."
        )
    )

    # --- Data / prompts ---
    parser.add_argument('--hg_csv', required=True,
                        help='Path to hg38 CSV with a "seq" column')
    parser.add_argument('--hg_prefix', default='<R>',
                        help='Prefix to prepend to genome sequence when building the prompt')
    parser.add_argument('--num_prompts', type=int, default=20,
                        help='Number of random prompts to sample from hg_csv (ignored if hg_row_indices/hg_ids provided)')
    parser.add_argument('--hg_row_indices', type=str, default=None,
                        help='Comma-separated list of 0-based row indices to use from hg_csv; '
                             'if set, overrides num_prompts and random sampling.')
    parser.add_argument('--hg_ids', type=str, default=None,
                        help='Comma-separated list of hg38 "id" strings to use '
                             '(matches the CSV "id" column, e.g. "chr1:3495000-3500000"); '
                             'if set, overrides --hg_row_indices and random sampling.')

    parser.add_argument('--num_tokens', type=int, default=256,
                        help='Number of new tokens to generate (after the prompt)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of repetitions per (prompt, hyperparam combo)')

    # --- Models / device ---
    parser.add_argument('--draft_model_name', default='dna_gpt0.1b_m',
                        help='Draft model name')
    parser.add_argument('--target_model_name', default='dna_gpt3b_m',
                        help='Target model name')
    parser.add_argument('--draft_weight', default='DNAGPT/checkpoints/dna_gpt0.1b_m.pth',
                        help='Draft model weights')
    parser.add_argument('--target_weight', default='DNAGPT/checkpoints/dna_gpt3b_m.pth',
                        help='Target model weights')
    parser.add_argument('--device', default=None,
                        help='Device, e.g. "cuda" or "cpu" (default: auto)')
    parser.add_argument('--dtype', default='float16',
                        help='Torch dtype for model weights, e.g. float16, bfloat16, float32')

    # Base context windows (can be overridden)
    parser.add_argument('--target_context_len', type=int, default=None,
                        help='Base max context (tokens) for target (default: model.max_len)')
    parser.add_argument('--draft_context_len', type=int, default=None,
                        help='Base max context (tokens) for draft (default: model.max_len)')

    # --- Extra sweeps: prefix & context sizes ---
    parser.add_argument('--prefix_token_lengths', type=str, default=None,
                        help='Comma-separated list of prompt prefix lengths in tokens to sweep, '
                             'e.g. "256,512,768". Each prompt will be truncated to these lengths; '
                             'if omitted, use the full prompt.')
    parser.add_argument('--target_context_values', type=str, default=None,
                        help='Comma-separated list of target context lengths (tokens) to sweep. '
                             'If omitted, use a single base target_context_len.')
    parser.add_argument('--draft_context_values', type=str, default=None,
                        help='Comma-separated list of draft context lengths (tokens) to sweep. '
                             'If omitted, use a single base draft_context_len.')

    # --- L grid ---
    parser.add_argument('--L_values', type=str, default='5',
                        help='Comma-separated list of speculation window sizes (gamma), '
                             'e.g. "2,3,4,5"')

    # --- Sweep mode for sampling vs lookahead ---
    parser.add_argument(
        '--sweep_mode',
        choices=['core', 'lookahead', 'all'],
        default='core',
        help=(
            "Which hyperparameters to sweep:\n"
            "  core      - sweep L, accept_mode, temperature, top_k, top_p;\n"
            "              keep lookahead_width/depth fixed (1).\n"
            "  lookahead - sweep L, accept_mode, lookahead_width/depth;\n"
            "              keep temperature/top_k/top_p fixed.\n"
            "  all       - sweep everything (large grid).\n"
            "Note: prefix/context sweeps are controlled separately by\n"
            "      --prefix_token_lengths / --target_context_values /\n"
            "      --draft_context_values."
        )
    )

    # --- Output ---
    parser.add_argument('--output_csv', type=str, default='specdec_grid_results.csv',
                        help='Where to store aggregated grid results')

    # --- RNG / misc ---
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging in speculative sampling (slows things down)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Set up RNG and device
    # ------------------------------------------------------------------
    torch.set_grad_enabled(False)
    seed_all_rng(args.seed)
    random.seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, args.dtype)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Sweep mode: {args.sweep_mode}")
    print(f"Output CSV: {args.output_csv}")

    # ------------------------------------------------------------------
    # Load tokenizer + models once
    # ------------------------------------------------------------------
    print(f"Loading tokenizer / target model: {args.target_model_name}")
    _, tokenizer = get_model(args.target_model_name)

    target_model, _ = get_model(args.target_model_name)
    target_model = load_model(target_model,
                              args.target_weight,
                              device=device,
                              dtype=dtype)

    print(f"Loading draft model: {args.draft_model_name}")
    draft_model, _ = get_model(args.draft_model_name)
    draft_model = load_model(draft_model,
                             args.draft_weight,
                             device=device,
                             dtype=dtype)

    base_target_context_len = args.target_context_len or getattr(target_model, "max_len", None)
    base_draft_context_len = args.draft_context_len or getattr(draft_model, "max_len", None)

    print(f"Base target context (tokens): {base_target_context_len}")
    print(f"Base draft  context (tokens): {base_draft_context_len}")

    # Build context-length sweep lists
    if args.target_context_values:
        target_context_values = parse_int_list(args.target_context_values)
    else:
        target_context_values = [base_target_context_len]

    if args.draft_context_values:
        draft_context_values = parse_int_list(args.draft_context_values)
    else:
        draft_context_values = [base_draft_context_len]

    print(f"Target context sweep values: {target_context_values}")
    print(f"Draft  context sweep values: {draft_context_values}")

    # ------------------------------------------------------------------
    # Load hg38 rows
    # ------------------------------------------------------------------
    if not os.path.exists(args.hg_csv):
        raise FileNotFoundError(f"hg38 CSV not found: {args.hg_csv}")

    with open(args.hg_csv, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in CSV: {args.hg_csv}")

    total_rows = len(rows)

    # Determine which rows to use: by id, by explicit indices, or random sample
    if args.hg_ids:
        requested_ids = [s.strip() for s in args.hg_ids.split(',') if s.strip()]
        id_to_indices = {}
        for idx, row in enumerate(rows):
            row_id = row.get('id')
            if row_id in requested_ids:
                id_to_indices.setdefault(row_id, []).append(idx)

        selected_indices = []
        for rid in requested_ids:
            if rid in id_to_indices:
                selected_indices.append(id_to_indices[rid][0])
            else:
                print(f"WARNING: hg_id '{rid}' not found in CSV; skipping.")
        num_prompts = len(selected_indices)
        if num_prompts == 0:
            raise RuntimeError("No valid hg_ids matched any rows in the CSV.")
        print(f"Using {num_prompts} prompts specified by --hg_ids.")
    elif args.hg_row_indices:
        raw_indices = parse_int_list(args.hg_row_indices)
        selected_indices = [i for i in raw_indices if 0 <= i < total_rows]
        if len(selected_indices) < len(raw_indices):
            print("WARNING: Some hg_row_indices were out of range and have been skipped.")
        num_prompts = len(selected_indices)
        if num_prompts == 0:
            raise RuntimeError("No valid hg_row_indices after filtering.")
        print(f"Using {num_prompts} prompts specified by --hg_row_indices.")
    else:
        num_prompts = min(args.num_prompts, total_rows)
        selected_indices = random.sample(range(total_rows), num_prompts)
        print(f"Total hg38 rows: {total_rows}, using {num_prompts} prompts (random sample).")

    print(f"Selected row indices: {selected_indices}")

    # ------------------------------------------------------------------
    # Define hyperparameter grids
    # ------------------------------------------------------------------
    L_values = parse_int_list(args.L_values)
    accept_modes = ['prob']  # you can add 'pt_gt_pd' etc. if you want

    # Base (default) values used when a dimension is NOT being swept
    base_temperature = 1.0
    base_top_k = 0
    base_top_p = 0.95
    base_lookahead_width = 1
    base_lookahead_depth = 1

    sweep_mode = args.sweep_mode

    if sweep_mode == 'core':
        # Sweep sampling hyperparameters + L / accept_mode
        temperature_values = [0.8, 0.9, 1.0, 1.1, 1.2]
        top_k_values = [0]
        top_p_values = [0.95]
        lookahead_width_values = [base_lookahead_width]
        lookahead_depth_values = [base_lookahead_depth]
    elif sweep_mode == 'lookahead':
        # Sweep lookahead hyperparameters + L / accept_mode
        temperature_values = [base_temperature]
        top_k_values = [base_top_k]
        top_p_values = [base_top_p]
        lookahead_width_values = [1, 2, 3]
        lookahead_depth_values = [1, 2, 3]
    else:  # 'all'
        temperature_values = [0.8, 0.9, 1.0, 1.1, 1.2]
        top_k_values = [0, 16, 32]
        top_p_values = [0.9, 0.95]
        lookahead_width_values = [1, 2, 3]
        lookahead_depth_values = [1, 2, 3]

    # Prefix sweep values (in tokens); we will filter per-prompt based on prompt length
    if args.prefix_token_lengths:
        prefix_values_raw = parse_int_list(args.prefix_token_lengths)
    else:
        prefix_values_raw = []  # means "use full prompt only"

    print("Grid sizes (before prompt-specific filtering):")
    print(f"  L_values:           {L_values}")
    print(f"  accept_modes:       {accept_modes}")
    print(f"  lookahead_width:    {lookahead_width_values}")
    print(f"  lookahead_depth:    {lookahead_depth_values}")
    print(f"  temperatures:       {temperature_values}")
    print(f"  top_k:              {top_k_values}")
    print(f"  top_p:              {top_p_values}")
    print(f"  prefix_token_lengths (raw): {prefix_values_raw}")
    print(f"  target_context_values:      {target_context_values}")
    print(f"  draft_context_values:       {draft_context_values}")

    # Rough progress estimation (ignores per-prompt filtering of prefix lengths)
    prefix_estimate = prefix_values_raw if prefix_values_raw else [0]  # treat 0 as "full prompt"
    combos_per_prompt = (
        len(L_values) *
        len(accept_modes) *
        len(lookahead_width_values) *
        len(lookahead_depth_values) *
        len(temperature_values) *
        len(top_k_values) *
        len(top_p_values) *
        len(target_context_values) *
        len(draft_context_values) *
        len(prefix_estimate)
    )
    print(f"Approx. hyperparam combos per prompt: {combos_per_prompt}")
    print(f"Approx. total runs: {combos_per_prompt * num_prompts}")

    # ------------------------------------------------------------------
    # Open CSV and write header
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)

        header = [
            # prompt info
            "prompt_idx",
            "hg_row_index",
            "hg_id",
            "chrom",
            "start",
            "end",
            "prompt_len_bases",
            "prompt_len_tokens",
            "prefix_len_tokens",
            "target_context_len",
            "draft_context_len",

            # hyperparams
            "L",
            "accept_mode",
            "lookahead_width",
            "lookahead_depth",
            "temperature",
            "top_k",
            "top_p",

            # metrics
            "target_tps",
            "draft_tps",
            "specdec_tps",
            "speedup_vs_target",
            "mean_accept_rate",
            "mean_accepted_prefix",
            "target_tokens_total",
            "draft_tokens_total",
            "specdec_tokens_total",
            "target_time_total",
            "draft_time_total",
            "specdec_time_total",

            # example suffixes (first sample)
            "sample_target_suffix",
            "sample_draft_suffix",
            "sample_specdec_suffix",
        ]
        writer.writerow(header)

        # ------------------------------------------------------------------
        # Main grid loop
        # ------------------------------------------------------------------
        run_idx = 0
        t0 = time.time()

        for prompt_idx, row_idx in enumerate(selected_indices):
            row = rows[row_idx]
            seq = row['seq'].strip()
            hg_id = row.get('id', 'N/A')
            chrom = row.get('chrom', 'NA')
            start = row.get('start', 'NA')
            end = row.get('end', 'NA')

            prompt = args.hg_prefix + seq
            prompt_len_bases = len(seq)

            # Encode prompt once
            prompt_ids_full = tokenizer.encode(prompt, device=device)
            prompt_ids_full = prompt_ids_full[None, :]  # [1, T]
            prompt_len_tokens = prompt_ids_full.shape[1]

            # Determine which prefix lengths (in tokens) are valid for this prompt
            if prefix_values_raw:
                valid_prefix_lengths = sorted({
                    min(pl, prompt_len_tokens)
                    for pl in prefix_values_raw
                    if pl > 0
                })
                if not valid_prefix_lengths:
                    valid_prefix_lengths = [prompt_len_tokens]
            else:
                valid_prefix_lengths = [prompt_len_tokens]

            print(f"\n=== Prompt {prompt_idx+1}/{num_prompts} "
                  f"(hg_row_index={row_idx}, id={hg_id}) ===")
            print(f"  Chrom: {chrom}, start: {start}, end: {end}")
            print(f"  Prompt length: {prompt_len_bases} bases, "
                  f"{prompt_len_tokens} tokens")
            print(f"  Prefix lengths for this prompt (tokens): {valid_prefix_lengths}")

            for prefix_len in valid_prefix_lengths:
                if prefix_len == prompt_len_tokens:
                    prompt_ids = prompt_ids_full
                else:
                    prompt_ids = prompt_ids_full[:, :prefix_len]

                for tc_len in target_context_values:
                    for dc_len in draft_context_values:
                        for L in L_values:
                            for accept_mode in accept_modes:
                                for lw in lookahead_width_values:
                                    for ld in lookahead_depth_values:
                                        for temp in temperature_values:
                                            for top_k in top_k_values:
                                                for top_p in top_p_values:
                                                    run_idx += 1
                                                    print(
                                                        f"[Run {run_idx}] "
                                                        f"prompt={prompt_idx}, prefix={prefix_len}, "
                                                        f"tc_len={tc_len}, dc_len={dc_len}, "
                                                        f"L={L}, mode={accept_mode}, "
                                                        f"lw={lw}, ld={ld}, "
                                                        f"T={temp}, top_k={top_k}, top_p={top_p}"
                                                    )

                                                    metrics = run_benchmarks_for_prompt(
                                                        target_model=target_model,
                                                        draft_model=draft_model,
                                                        tokenizer=tokenizer,
                                                        prompt_ids=prompt_ids,
                                                        max_new_tokens=args.num_tokens,
                                                        num_samples=args.num_samples,
                                                        temperature=temp,
                                                        top_k=top_k,
                                                        top_p=top_p,
                                                        L=L,
                                                        accept_mode=accept_mode,
                                                        target_context_len=tc_len,
                                                        draft_context_len=dc_len,
                                                        lookahead_width=lw,
                                                        lookahead_depth=ld,
                                                        debug=args.debug,
                                                        verbose=False,
                                                        prompt_text=prompt,
                                                    )

                                                    writer.writerow([
                                                        # prompt info
                                                        prompt_idx,
                                                        row_idx,
                                                        hg_id,
                                                        chrom,
                                                        start,
                                                        end,
                                                        prompt_len_bases,
                                                        prompt_len_tokens,
                                                        prefix_len,
                                                        tc_len,
                                                        dc_len,

                                                        # hyperparams
                                                        L,
                                                        accept_mode,
                                                        lw,
                                                        ld,
                                                        temp,
                                                        top_k,
                                                        top_p,

                                                        # metrics
                                                        metrics["target_tps"],
                                                        metrics["draft_tps"],
                                                        metrics["specdec_tps"],
                                                        metrics["speedup_vs_target"],
                                                        metrics["mean_accept_rate"],
                                                        metrics["mean_accepted_prefix"],
                                                        metrics["target_tokens_total"],
                                                        metrics["draft_tokens_total"],
                                                        metrics["specdec_tokens_total"],
                                                        metrics["target_time_total"],
                                                        metrics["draft_time_total"],
                                                        metrics["specdec_time_total"],

                                                        # example suffixes
                                                        metrics.get("sample_target_text", ""),
                                                        metrics.get("sample_draft_text", ""),
                                                        metrics.get("sample_specdec_text", ""),
                                                    ])

                                                    # Flush so partial results survive if job dies
                                                    f_out.flush()

        t1 = time.time()
        print(f"\nGrid search finished in {(t1 - t0)/3600:.2f} hours.")
        print(f"Results written to: {args.output_csv}")


if __name__ == "__main__":
    main()
