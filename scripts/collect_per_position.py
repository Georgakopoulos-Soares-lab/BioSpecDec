#!/usr/bin/env python3
"""
Reviewer 2.a.3 — Per-position acceptance rate data collection.

Runs speculative decoding with log_per_position=True for a subset of prompts,
saves per-token accept/reject decisions to a CSV for downstream analysis.

Usage (from BioSpecDec root):
  export LD_LIBRARY_PATH=/scratch/10906/arisk/envs/BioSpecDec/lib:$LD_LIBRARY_PATH
  PYTHONDONTWRITEBYTECODE=1 python scripts/collect_per_position.py \
      --model protgpt2 --num_samples 3 --gamma 5 --draft_layers 6

Models supported: protgpt2, progen2
DNAGPT support requires the custom checkpoint loading — use --model dnagpt.
"""

import argparse
import json
import os
import sys
import time

import torch
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

# Default prompts per model (same ones used in sweeps)
DEFAULT_PROMPTS = {
    "protgpt2": [
        "<|endoftext|>",
        "<|endoftext|>M",
        "<|endoftext|>MKTLL",
        "<|endoftext|>MKTLLLTLVVVTIVCL",
    ],
    "progen2": [
        "1M",
        "1MKTLL",
        "1MKTLLLTLVVVTIVCLD",
    ],
}


def run_protgpt2(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from specdec_protein import speculative_sampling

    print(f"Loading ProtGPT2 target model: nferruz/ProtGPT2")
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    full_model = AutoModelForCausalLM.from_pretrained(
        "nferruz/ProtGPT2", torch_dtype=dtype
    ).to(args.device).eval()

    n_layers = len(full_model.transformer.h)
    print(f"  {n_layers} layers, using {args.draft_layers} for draft")

    # Build truncated draft
    import copy
    draft_model = copy.deepcopy(full_model)
    keep = list(range(args.draft_layers))
    draft_model.transformer.h = torch.nn.ModuleList(
        [draft_model.transformer.h[i] for i in keep]
    )
    draft_model.config.n_layer = len(keep)

    prompts = args.prompts or DEFAULT_PROMPTS["protgpt2"]
    eos_id = tokenizer.eos_token_id

    all_records = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        print(f"\nPrompt {pi}: '{prompt}' ({input_ids.size(1)} tokens)")

        for s in range(args.num_samples):
            result = speculative_sampling(
                full_model, draft_model, input_ids,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                accept_mode=args.accept_mode,
                eos_token_id=eos_id,
                debug=False,
                log_per_position=True,
            )
            ids, duration, acc_rate, pos_log = result
            print(f"  sample {s}: acc_rate={acc_rate:.3f}, {len(pos_log)} decisions, {duration:.2f}s")

            for rec in pos_log:
                rec["model"] = "protgpt2"
                rec["prompt_idx"] = pi
                rec["sample_idx"] = s
                rec["prompt_text"] = prompt
                rec["gamma"] = args.gamma
                rec["draft_layers"] = args.draft_layers
            all_records.extend(pos_log)

    return all_records


def run_progen2(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from specdec_progen2_truncated import speculative_sampling

    model_name = "hugohrban/progen2-xlarge"
    print(f"Loading ProGen2 target model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    ).to(args.device).eval()

    n_layers = len(full_model.transformer.h)
    print(f"  {n_layers} layers, using {args.draft_layers} for draft")

    import copy
    draft_model = copy.deepcopy(full_model)
    keep = list(range(args.draft_layers))
    draft_model.transformer.h = torch.nn.ModuleList(
        [draft_model.transformer.h[i] for i in keep]
    )
    draft_model.config.n_layer = len(keep)

    prompts = args.prompts or DEFAULT_PROMPTS["progen2"]

    all_records = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        print(f"\nPrompt {pi}: '{prompt}' ({input_ids.size(1)} tokens)")

        for s in range(args.num_samples):
            result = speculative_sampling(
                full_model, draft_model, input_ids,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                accept_mode=args.accept_mode,
                eos_token_id=None,
                debug=False,
                log_per_position=True,
            )
            ids, duration, acc_rate, pos_log = result
            print(f"  sample {s}: acc_rate={acc_rate:.3f}, {len(pos_log)} decisions, {duration:.2f}s")

            for rec in pos_log:
                rec["model"] = "progen2"
                rec["prompt_idx"] = pi
                rec["sample_idx"] = s
                rec["prompt_text"] = prompt
                rec["gamma"] = args.gamma
                rec["draft_layers"] = args.draft_layers
            all_records.extend(pos_log)

    return all_records


def run_dnagpt(args):
    sys.path.insert(0, os.path.join(ROOT, "DNAGPT"))
    from scoring_specdec_beam_search import speculative_sampling, build_model, build_tokenizer

    target_weight = os.path.join(ROOT, "DNAGPT", "checkpoints", "dna_gpt3b_m.pth")
    draft_weight = os.path.join(ROOT, "DNAGPT", "checkpoints", "dna_gpt0.1b_m.pth")

    print("Loading DNAGPT models...")
    tokenizer = build_tokenizer()
    target_model = build_model("dna_gpt3b_m", target_weight, args.device)
    draft_model = build_model("dna_gpt0.1b_m", draft_weight, args.device)

    # Use a few chr1 prompts from the sweep
    prompts = [
        "<R>ACGTACGTACGTACGTACGTACGTACGTACGT",
        "<R>TGCATGCATGCATGCATGCATGCATGCATGCA",
        "<R>ATAATAATAATAATAATAATAATAATAATAATAA",
    ]

    all_records = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
        print(f"\nPrompt {pi}: '{prompt[:40]}...' ({input_ids.size(1)} tokens)")

        for s in range(args.num_samples):
            result = speculative_sampling(
                target_model, draft_model, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                accept_mode=args.accept_mode,
                debug=False,
                log_per_position=True,
            )
            ids, acc_rate, mean_pref, pos_log = result
            print(f"  sample {s}: acc_rate={acc_rate:.3f}, {len(pos_log)} decisions")

            for rec in pos_log:
                rec["model"] = "dnagpt"
                rec["prompt_idx"] = pi
                rec["sample_idx"] = s
                rec["prompt_text"] = prompt[:40]
                rec["gamma"] = args.gamma
            all_records.extend(pos_log)

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Collect per-position acceptance data")
    parser.add_argument("--model", required=True, choices=["protgpt2", "progen2", "dnagpt"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--draft_layers", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=950)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--accept_mode", default="prob")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--prompts", nargs="*", default=None, help="Custom prompts (overrides defaults)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(ROOT, "results", f"per_position_{args.model}.csv")

    print(f"Model: {args.model}, gamma={args.gamma}, draft_layers={args.draft_layers}")
    print(f"num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}")
    print(f"Output: {args.output}")

    if args.model == "protgpt2":
        records = run_protgpt2(args)
    elif args.model == "progen2":
        records = run_progen2(args)
    elif args.model == "dnagpt":
        records = run_dnagpt(args)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} per-position records to {args.output}")

    # Quick summary
    if len(df) > 0:
        print(f"\n--- Quick Summary ---")
        print(f"Total decisions: {len(df)}")
        print(f"Overall acceptance: {df.accepted.mean():.3f}")
        by_pos = df.groupby("seq_pos")["accepted"].mean()
        print(f"Acceptance by seq_pos (first 10):")
        print(by_pos.head(10).to_string())
        by_block_pos = df.groupby("pos_in_block")["accepted"].mean()
        print(f"\nAcceptance by pos_in_block:")
        print(by_block_pos.to_string())


if __name__ == "__main__":
    main()
