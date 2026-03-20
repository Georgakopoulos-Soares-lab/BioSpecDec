import argparse
import json
import os
import sys
import time
import csv
import random
import math
from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
_DNAGPT_DIR = os.path.join(_REPO_ROOT, "DNAGPT")
if os.path.isdir(_DNAGPT_DIR) and _DNAGPT_DIR not in sys.path:
    sys.path.append(_DNAGPT_DIR)

_CWD_DNAGPT_DIR = os.path.join(os.getcwd(), "DNAGPT")
if os.path.isdir(_CWD_DNAGPT_DIR) and _CWD_DNAGPT_DIR not in sys.path:
    sys.path.append(_CWD_DNAGPT_DIR)

from dna_gpt.model import DNAGPT
from dna_gpt.tokenizer import KmerTokenizer
from dna_gpt.utils import seed_all_rng, top_k_top_p_filter


# ----------------------------------------------------------------------
# Model + tokenizer helpers
# ----------------------------------------------------------------------

def get_model(model_name):
    special_tokens = (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
                      ["+", '-', '*', '/', '=', "&", "|", "!"] +
                      ['M', 'B'] + ['P'] +
                      ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] +
                      ['W', 'Y', 'X', 'Z'])
    if model_name in ('dna_gpt0.1b_h',):
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
    else:
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

    vocab_size = len(tokenizer)
    model = DNAGPT.from_name(model_name, vocab_size)
    return model, tokenizer


def load_model(model, weight_path, device=None, dtype=None):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    state = torch.load(weight_path, map_location="cpu")
    if 'model' in state.keys():
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"Loaded model weights from {weight_path}")
    model.to(device=device, dtype=dtype)
    model = model.eval()
    return model


# ----------------------------------------------------------------------
# Baseline autoregressive generation
# ----------------------------------------------------------------------

def generate_baseline(model, tokenizer, prompt_ids, max_new_tokens,
                      temperature=1.0, top_k=0, top_p=0.0,
                      context_len=None):
    """Standard autoregressive generation with sampling.

    context_len: sliding window length (<= model.max_len)
    """
    device = next(model.parameters()).device
    idx = prompt_ids.to(device).clone()

    if context_len is None:
        context_len = getattr(model, "max_len", idx.size(1))

    for _ in range(max_new_tokens):
        if idx.size(1) <= context_len:
            idx_cond = idx
        else:
            idx_cond = idx[:, -context_len:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        # multinomial sampling from the (possibly truncated) distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() in (tokenizer.unk_id, tokenizer.pad_id):
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def _timeit(device: torch.device, fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Accurate wall-time measurement.

    - CUDA: uses cuda events + synchronize (accounts for async kernels)
    - CPU: uses perf_counter
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return out, float(start.elapsed_time(end) / 1000.0)
    t0 = time.perf_counter()
    out = fn()
    return out, float(time.perf_counter() - t0)


def _decode_suffix(tokenizer, full_ids_1d: list[int], prompt_len: int) -> str:
    # Token-exact suffix decoding (no fragile string slicing).
    return tokenizer.decode(full_ids_1d[prompt_len:])


# ----------------------------------------------------------------------
# Draft block (plain sampling)
# ----------------------------------------------------------------------

def draft_block(
    draft_model,
    tokenizer,
    base_prefix,          # [1, T_prefix] on draft device
    max_block_tokens,     # total number of *draft* tokens we may propose in this block (L)
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    context_len=None,
    debug=False,
):
    """
    Generate up to `max_block_tokens` draft tokens by plain autoregressive
    sampling from the draft model.

    This is intentionally plain multinomial sampling (no beam search).
    """
    device = base_prefix.device

    if context_len is None:
        context_len = getattr(draft_model, "max_len", base_prefix.size(1))

    idx = base_prefix.clone()
    draft_tokens = []
    draft_distributions = []

    for step in range(max_block_tokens):
        # respect sliding context window
        if idx.size(1) <= context_len:
            idx_cond = idx
        else:
            idx_cond = idx[:, -context_len:]

        with torch.no_grad():
            logits = draft_model(idx_cond)      # [1, T, V]

        logits = logits[:, -1, :]              # [1, V]

        # temperature
        if temperature != 1.0:
            logits = logits / temperature

        # top-k / top-p filter in logit space, then softmax → probs
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)      # [1, V]

        # multinomial sampling (not argmax!)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # stop on pad/unk
        if next_token.item() in (tokenizer.unk_id, tokenizer.pad_id):
            if debug:
                print(f"[draft_block] stopping at step {step} due to special token {next_token.item()}")
            break

        # record token and distribution so target can compute Pt/Pd later
        draft_tokens.append(next_token.to(device))
        draft_distributions.append(probs.to(device))

        # extend prefix
        idx = torch.cat((idx, next_token.to(device)), dim=1)

    if debug:
        print(f"[draft_block] produced {len(draft_tokens)} draft tokens")

    return draft_tokens, draft_distributions



# ----------------------------------------------------------------------
# Speculative decoding
# ----------------------------------------------------------------------

def speculative_sampling(target_model, draft_model, tokenizer, prompt_ids, max_new_tokens,
                         gamma=5, temperature=1.0, top_k=0, top_p=0.0,
                         accept_mode='prob',
                         target_context_len=None, draft_context_len=None,
                         debug=False, log_per_position=False):
    """Speculative Sampling (Draft-Verification-Correction) with configurable
    acceptance and different context windows for target/draft.

    If log_per_position=True, returns an extra list of per-token decision dicts.
    """
    device_target = next(target_model.parameters()).device
    device_draft = next(draft_model.parameters()).device

    idx = prompt_ids.to(device_target).clone()
    T_total = idx.shape[1] + max_new_tokens
    prompt_len = idx.size(1)

    if target_context_len is None:
        target_context_len = getattr(target_model, "max_len", idx.size(1))
    if draft_context_len is None:
        draft_context_len = getattr(draft_model, "max_len", idx.size(1))

    accepted_count = 0
    total_draft_tokens = 0
    accepted_prefix_sum = 0
    per_position_log = []
    blocks = 0

    while idx.shape[1] < T_total:
        remaining = T_total - idx.shape[1]
        if remaining <= 0:
            break

        # ---- 1. Draft generation ----
        curr_idx = idx.clone().to(device_draft)
        block_limit = min(gamma, remaining)

        draft_tokens, draft_distributions = draft_block(
            draft_model=draft_model,
            tokenizer=tokenizer,
            base_prefix=curr_idx,
            max_block_tokens=block_limit,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            context_len=draft_context_len,
            debug=debug,
        )

        if not draft_tokens:
            # Fallback: one target step
            if idx.size(1) <= target_context_len:
                idx_cond = idx
            else:
                idx_cond = idx[:, -target_context_len:]

            with torch.no_grad():
                logits = target_model(idx_cond)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            filtered_logits = top_k_top_p_filter(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() in (tokenizer.unk_id, tokenizer.pad_id):
                break

            idx = torch.cat((idx, next_token), dim=1)
            continue

        num_draft = len(draft_tokens)
        total_draft_tokens += num_draft
        blocks += 1
        accepted_in_block = 0

        # ---- 2. Target Verification ----
        draft_seq = torch.cat(draft_tokens, dim=1)    # [1, num_draft]
        target_input = torch.cat((idx, draft_seq), dim=1)

        if target_input.size(1) <= target_context_len:
            target_input_cond = target_input
        else:
            target_input_cond = target_input[:, -target_context_len:]

        with torch.no_grad():
            target_logits = target_model(target_input_cond)

        relevant_logits = target_logits[:, -(num_draft + 1):, :]  # [1, num_draft+1, V]

        # ---- 3. Verification loop ----
        all_accepted = True

        for i in range(num_draft):
            token_id = draft_tokens[i].item()
            p_d = draft_distributions[i]  # [1, V]

            # Target distribution P_t at this position
            t_logits_i = relevant_logits[:, i, :]
            if temperature != 1.0:
                t_logits_i = t_logits_i / temperature
            t_logits_i = top_k_top_p_filter(t_logits_i, top_k=top_k, top_p=top_p)
            p_t = F.softmax(t_logits_i, dim=-1)

            p_d_token = p_d[0, token_id].item()
            p_t_token = p_t[0, token_id].item()
            p_d_token = max(p_d_token, 1e-12)  # safety
            _seq_pos = idx.size(1) - prompt_len

            if accept_mode == 'prob':
                r = torch.rand(1).item()
                accept_prob = min(1.0, p_t_token / p_d_token)

                if r < accept_prob:
                    idx = torch.cat((idx, draft_tokens[i]), dim=1)
                    accepted_count += 1
                    accepted_in_block += 1
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": accept_prob})
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": accept_prob})

                    residual_probs = torch.clamp(p_t - p_d, min=0.0)
                    residual_sum = residual_probs.sum(dim=-1, keepdim=True)

                    if residual_sum.item() > 0:
                        residual_probs = residual_probs / residual_sum
                        new_token = torch.multinomial(residual_probs, num_samples=1)
                    else:
                        new_token = torch.multinomial(p_t, num_samples=1)

                    idx = torch.cat((idx, new_token), dim=1)
                    break

            elif accept_mode == 'pt_gt_pd':
                if p_t_token > p_d_token:
                    idx = torch.cat((idx, draft_tokens[i]), dim=1)
                    accepted_count += 1
                    accepted_in_block += 1
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token})
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token})

                    residual_probs = torch.clamp(p_t - p_d, min=0.0)
                    residual_sum = residual_probs.sum(dim=-1, keepdim=True)

                    if residual_sum.item() > 0:
                        residual_probs = residual_probs / residual_sum
                        new_token = torch.multinomial(residual_probs, num_samples=1)
                    else:
                        new_token = torch.multinomial(p_t, num_samples=1)

                    idx = torch.cat((idx, new_token), dim=1)
                    break

            elif accept_mode == 'match':
                target_argmax = t_logits_i.argmax(dim=-1, keepdim=True)
                target_token_id = target_argmax.item()

                if token_id == target_token_id:
                    idx = torch.cat((idx, draft_tokens[i]), dim=1)
                    accepted_count += 1
                    accepted_in_block += 1
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token})
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": blocks, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token})
                    idx = torch.cat((idx, target_argmax), dim=1)
                    break

            else:
                raise ValueError(f"Unknown accept_mode: {accept_mode}")

        accepted_prefix_sum += accepted_in_block

        # ---- 4. Bonus token if all draft tokens accepted ----
        if all_accepted:
            bonus_logits = relevant_logits[:, num_draft, :]
            if temperature != 1.0:
                bonus_logits = bonus_logits / temperature
            bonus_logits = top_k_top_p_filter(bonus_logits, top_k=top_k, top_p=top_p)
            p_bonus = F.softmax(bonus_logits, dim=-1)

            bonus_token = torch.multinomial(p_bonus, num_samples=1)

            if bonus_token.item() not in (tokenizer.unk_id, tokenizer.pad_id):
                idx = torch.cat((idx, bonus_token), dim=1)

        if idx[0, -1].item() in (tokenizer.unk_id, tokenizer.pad_id):
            break

    acceptance_rate = accepted_count / max(1, total_draft_tokens)
    mean_accepted_prefix = accepted_prefix_sum / max(1, blocks)
    if log_per_position:
        return idx, acceptance_rate, mean_accepted_prefix, per_position_log
    return idx, acceptance_rate, mean_accepted_prefix


# ----------------------------------------------------------------------
# Reusable benchmark function (no argparse, used by grid search)
# ----------------------------------------------------------------------

def run_benchmarks_for_prompt(
    target_model,
    draft_model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    num_samples,
    temperature,
    top_k,
    top_p,
    L,
    accept_mode,
    target_context_len=None,
    draft_context_len=None,
    debug=False,
    verbose=False,
):
    """Run target baseline, draft baseline, and speculative decoding for a
    single prompt.

        Returns a dict with aggregated metrics (TPS, speedup, acceptance rate, etc.)
        plus example generated suffixes (after the prompt prefix) for:
      - target baseline
      - draft baseline
      - specdec
    """
    metrics = {
        'target': {'tokens': [], 'time': []},
        'draft': {'tokens': [], 'time': []},
        'specdec': {'tokens': [], 'time': [], 'acceptance_rate': []}
    }

    if target_context_len is None:
        target_context_len = getattr(target_model, "max_len", None)
    if draft_context_len is None:
        draft_context_len = getattr(draft_model, "max_len", None)

    prompt_len = int(prompt_ids.shape[1])
    prompt_ids_1d = prompt_ids[0].detach().cpu().tolist()

    sample_prompt_ids_json = json.dumps(prompt_ids_1d)
    sample_target_new_ids_json = None
    sample_draft_new_ids_json = None
    sample_specdec_new_ids_json = None

    sample_target_suffix = None
    sample_draft_suffix = None
    sample_specdec_suffix = None

    if verbose:
        print("\n" + "=" * 50)
        print("Starting Benchmarks")
        print("=" * 50)
        print("\nRunning Target Baseline...")

    # 1. Target Baseline
    device_target = next(target_model.parameters()).device
    for i in range(num_samples):
        idx, duration = _timeit(
            device_target,
            lambda: generate_baseline(
                target_model,
                tokenizer,
                prompt_ids,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                context_len=target_context_len,
            ),
        )
        n_new = idx.shape[1] - prompt_ids.shape[1]
        metrics['target']['tokens'].append(n_new)
        metrics['target']['time'].append(duration)

        # Capture example target suffix for first sample
        if i == 0:
            full_ids = idx[0].detach().cpu().tolist()
            sample_target_suffix = _decode_suffix(tokenizer, full_ids, prompt_len)
            sample_target_new_ids_json = json.dumps(full_ids[prompt_len:])

        if verbose:
            print(f"Target sample {i+1}: {n_new} tokens in {duration:.2f}s ({n_new/duration:.2f} tps)")

    # 2. Draft Baseline
    if verbose:
        print("\nRunning Draft Baseline...")

    device_draft = next(draft_model.parameters()).device
    for i in range(num_samples):
        idx, duration = _timeit(
            device_draft,
            lambda: generate_baseline(
                draft_model,
                tokenizer,
                prompt_ids,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                context_len=draft_context_len,
            ),
        )
        n_new = idx.shape[1] - prompt_ids.shape[1]
        metrics['draft']['tokens'].append(n_new)
        metrics['draft']['time'].append(duration)

        # Capture example draft suffix for first sample
        if i == 0:
            full_ids = idx[0].detach().cpu().tolist()
            sample_draft_suffix = _decode_suffix(tokenizer, full_ids, prompt_len)
            sample_draft_new_ids_json = json.dumps(full_ids[prompt_len:])

        if verbose:
            print(f"Draft sample {i+1}: {n_new} tokens in {duration:.2f}s ({n_new/duration:.2f} tps)")

    # 3. Speculative Decoding
    if verbose:
        print("\nRunning Speculative Decoding...")

    prefix_means = []
    for i in range(num_samples):
        (idx, acc_rate, mean_accepted_prefix), duration = _timeit(
            device_target,
            lambda: speculative_sampling(
                target_model,
                draft_model,
                tokenizer,
                prompt_ids,
                max_new_tokens,
                gamma=L,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                accept_mode=accept_mode,
                target_context_len=target_context_len,
                draft_context_len=draft_context_len,
                debug=debug,
            ),
        )
        n_new = idx.shape[1] - prompt_ids.shape[1]
        metrics['specdec']['tokens'].append(n_new)
        metrics['specdec']['time'].append(duration)
        metrics['specdec']['acceptance_rate'].append(acc_rate)
        prefix_means.append(mean_accepted_prefix)

        # Capture example SpecDec suffix for first sample
        if i == 0:
            full_ids = idx[0].detach().cpu().tolist()
            sample_specdec_suffix = _decode_suffix(tokenizer, full_ids, prompt_len)
            sample_specdec_new_ids_json = json.dumps(full_ids[prompt_len:])

        if verbose:
            print(f"SpecDec sample {i+1}: {n_new} tokens in {duration:.2f}s ({n_new/duration:.2f} tps) "
                  f"- Acc Rate: {acc_rate:.2f}")

    # Summary stats
    def get_stats(key):
        total_tokens = sum(metrics[key]['tokens'])
        total_time = sum(metrics[key]['time'])
        tps = total_tokens / total_time if total_time > 0 else 0.0
        return tps, total_tokens, total_time

    target_tps, target_tokens_total, target_time_total = get_stats('target')
    draft_tps, draft_tokens_total, draft_time_total = get_stats('draft')
    specdec_tps, specdec_tokens_total, specdec_time_total = get_stats('specdec')

    avg_acc_rate = (sum(metrics['specdec']['acceptance_rate']) /
                    max(1, len(metrics['specdec']['acceptance_rate'])))
    avg_prefix = (sum(prefix_means) / max(1, len(prefix_means))) if prefix_means else 0.0
    speedup_vs_target = specdec_tps / target_tps if target_tps > 0 else 0.0

    if verbose:
        print("\n" + "=" * 50)
        print("Summary Results")
        print("=" * 50)
        print(f"Target Model TPS: {target_tps:.2f}")
        print(f"Draft Model TPS:  {draft_tps:.2f}")
        print(f"SpecDec TPS:      {specdec_tps:.2f}")
        if target_tps > 0:
            print(f"Speedup vs Target: {speedup_vs_target:.2f}x")
        else:
            print("Speedup vs Target: N/A")
        print(f"Mean Acceptance Rate: {avg_acc_rate:.2f}")
        print(f"Mean Accepted Prefix Length: {avg_prefix:.2f}")

        print("\n=== Example generated suffixes (first sample) ===")
        print("Target baseline suffix:")
        print(sample_target_suffix if sample_target_suffix is not None else "<none>")
        print("\nDraft baseline suffix:")
        print(sample_draft_suffix if sample_draft_suffix is not None else "<none>")
        print("\nSpecDec suffix:")
        print(sample_specdec_suffix if sample_specdec_suffix is not None else "<none>")

    out = {
        "target_tps": target_tps,
        "draft_tps": draft_tps,
        "specdec_tps": specdec_tps,
        "speedup_vs_target": speedup_vs_target,
        "mean_accept_rate": avg_acc_rate,
        "mean_accepted_prefix": avg_prefix,
        "target_tokens_total": target_tokens_total,
        "draft_tokens_total": draft_tokens_total,
        "specdec_tokens_total": specdec_tokens_total,
        "target_time_total": target_time_total,
        "draft_time_total": draft_time_total,
        "specdec_time_total": specdec_time_total,
        "sample_target_suffix": sample_target_suffix,
        "sample_draft_suffix": sample_draft_suffix,
        "sample_specdec_suffix": sample_specdec_suffix,
        "sample_prompt_ids": sample_prompt_ids_json,
        "sample_target_new_ids": sample_target_new_ids_json,
        "sample_draft_new_ids": sample_draft_new_ids_json,
        "sample_specdec_new_ids": sample_specdec_new_ids_json,
    }

    return out


# ----------------------------------------------------------------------
# Main script (CLI) – thin wrapper around run_benchmarks_for_prompt
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Speculative Decoding for DNAGPT (configurable acceptance + hg38 CSV)'
    )

    parser.add_argument('--task', default="generation", help='dtype of the model weights')
    parser.add_argument('--input', '-i', default='<R>CTGTATACCACAGA',
                        help='fallback input prompt text (if no CSV is used)')
    parser.add_argument('--numbers', help='input number list')  # Ignored
    parser.add_argument('--num_samples', '-ns', type=int, default=10,
                        help='number of samples for benchmarking')
    parser.add_argument('--num_tokens', '-nt', type=int, default=256,
                        help='number of new tokens to generate (after the prompt)')
    parser.add_argument('--temperature', type=float, default=1.0, help='sample temperature')
    parser.add_argument('--topk', type=int, default=0, help='sample topk')
    parser.add_argument('--topp', type=float, default=0.95, help='sample topp')
    parser.add_argument('--seed', default=40, type=int, help='random seed for sampling')
    parser.add_argument('--device', default=None, help='device of the model, cuda or cpu')
    parser.add_argument('--dtype', default=None, help='dtype of the model weights')

    # Speculative Decoding Arguments
    parser.add_argument('--draft_model_name', default='dna_gpt0.1b_m', help='Draft model name')
    parser.add_argument('--target_model_name', default='dna_gpt3b_m', help='Target model name')
    parser.add_argument('--draft_weight', default='DNAGPT/checkpoints/dna_gpt0.1b_m.pth',
                        help='Draft model weights')
    parser.add_argument('--target_weight', default='DNAGPT/checkpoints/dna_gpt3b_m.pth',
                        help='Target model weights')
    parser.add_argument('--L', type=int, default=5, help='Speculation window size (gamma)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    parser.add_argument(
        '--accept_mode',
        choices=['prob', 'pt_gt_pd', 'match'],
        default='prob',
        help=(
            "Acceptance rule for draft tokens:\n"
            "  prob     - Monte Carlo: accept with probability min(1, Pt/Pd)\n"
            "  pt_gt_pd - accept iff Pt(x) > Pd(x); else residual sampling\n"
            "  match    - deterministic: accept only if draft token == target argmax"
        )
    )

    # separate context window sizes for target/draft
    parser.add_argument('--target_context_len', type=int, default=None,
                        help='Max context (tokens) for target model (default: model.max_len)')
    parser.add_argument('--draft_context_len', type=int, default=None,
                        help='Max context (tokens) for draft model (default: model.max_len)')


    # hg38 CSV sampling
    parser.add_argument('--hg_csv', default=None,
                        help='Path to hg38 CSV with a "seq" column')
    parser.add_argument('--hg_row_index', type=int, default=-1,
                        help='Row index (0-based) to use from CSV; if <0, choose a random row')
    parser.add_argument('--hg_prefix', default='<R>',
                        help='Prefix to prepend to genome sequence when building the prompt')

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    seed_all_rng(args.seed)
    random.seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = args.dtype or 'float16'
    dtype = getattr(torch, dtype)

    if args.num_tokens <= 0:
        print(f"num_tokens must be > 0 (got {args.num_tokens}). Exiting.")
        return

    print(f"Device: {device}, Dtype: {dtype}, accept_mode: {args.accept_mode}")
    print(f"Target model: {args.target_model_name}")
    print(f"Draft  model: {args.draft_model_name}")
    print(f"L={args.L}")

    # Load Tokenizer
    _, tokenizer = get_model(args.target_model_name)

    # Load Target Model
    print(f"Loading Target Model: {args.target_model_name}")
    target_model, _ = get_model(args.target_model_name)
    target_model = load_model(target_model, args.target_weight, device=device, dtype=dtype)

    # Load Draft Model
    print(f"Loading Draft Model: {args.draft_model_name}")
    draft_model, _ = get_model(args.draft_model_name)
    draft_model = load_model(draft_model, args.draft_weight, device=device, dtype=dtype)

    # Resolve context lengths
    target_context_len = args.target_context_len or getattr(target_model, "max_len", None)
    draft_context_len = args.draft_context_len or getattr(draft_model, "max_len", None)
    print(f"Target context (tokens): {target_context_len}")
    print(f"Draft context (tokens):  {draft_context_len}")

    # Prepare Input prompt
    prompt = args.input

    if args.hg_csv is not None:
        if not os.path.exists(args.hg_csv):
            raise FileNotFoundError(f"hg38 CSV not found: {args.hg_csv}")

        with open(args.hg_csv, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"No rows found in CSV: {args.hg_csv}")

        if 0 <= args.hg_row_index < len(rows):
            row = rows[args.hg_row_index]
            print(f"Using hg38 row index {args.hg_row_index} / {len(rows)}")
        else:
            row = random.choice(rows)
            print(f"Using RANDOM hg38 row (total {len(rows)} rows)")

        seq = row['seq'].strip()
        prompt = args.hg_prefix + seq
        print(f"Selected hg38 id: {row.get('id', 'N/A')}")
        print(f"Chrom: {row.get('chrom', 'NA')}  start: {row.get('start', 'NA')}  end: {row.get('end', 'NA')}")
        print(f"Prompt sequence length (bases): {len(seq)}")

    # Encode prompt (no truncation by num_tokens; context is controlled by *_context_len)
    prompt_ids = tokenizer.encode(prompt, device=device)
    prompt_ids = prompt_ids[None, :]  # Batch dim

    print(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Prompt token IDs length: {prompt_ids.shape[1]}")
    print(f"Prompt IDs (truncated view): {prompt_ids[0, :20].tolist()}")
    print(f"Requested new tokens: {args.num_tokens}")

    max_new_tokens = args.num_tokens

    # Run benchmarks for this prompt (verbose=True prints sequences including draft)
    _ = run_benchmarks_for_prompt(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        L=args.L,
        accept_mode=args.accept_mode,
        target_context_len=target_context_len,
        draft_context_len=draft_context_len,
        debug=args.debug,
        verbose=True,
    )


if __name__ == "__main__":
    main()
