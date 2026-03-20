#!/usr/bin/env python
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# Optional matmul speedups on newer GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------- truncated GPT2-style draft ----------------- #

class TruncatedProtGPT2(nn.Module):
    '''
    Draft model built by reusing the embeddings + a subset of transformer blocks
    + LM head from a full GPT2-style CausalLM.

    - full_model: AutoModelForCausalLM with a .transformer.h list of blocks
    - layer_indices: list of int indices into full_model.transformer.h
    '''
    def __init__(self, full_model, layer_indices):
        super().__init__()
        self.config = full_model.config

        tr = full_model.transformer
        self.wte = tr.wte
        self.wpe = tr.wpe
        self.drop = tr.drop

        self.layer_indices = list(layer_indices)
        self.blocks = nn.ModuleList([tr.h[i] for i in self.layer_indices])
        self.ln_f = tr.ln_f

        # shared LM head
        self.lm_head = full_model.lm_head

    def forward(self, input_ids=None, **kwargs):
        '''
        Minimal GPT-2 style forward (no attention mask, no KV cache).

        input_ids: [B, T]
        '''
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        bsz, seq_len = input_ids.size()
        device = input_ids.device

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)  # [B, T]

        inputs_embeds = self.wte(input_ids)          # [B, T, D]
        pos_embeds = self.wpe(position_ids)          # [B, T, D]
        hidden_states = inputs_embeds + pos_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            # GPT2Block returns (hidden_states, presents, ...)
            hidden_states = block(hidden_states)[0]

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(logits=logits)


# ----------------- sampling utils ----------------- #

def top_k_top_p_filter(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    '''
    Apply top-k and/or nucleus (top-p) filtering to logits.

    logits: (..., vocab_size)
    returns: filtered logits with some entries set to filter_value
    '''
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values_to_keep, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values_to_keep[..., -1, None]
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, filter_value),
            logits,
        )

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to always keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def generate_baseline(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    context_len=None,   # kept for API compat, ignored in KV path
    eos_token_id=None,
    use_kv_cache=True,
):
    '''
    Standard autoregressive generation with sampling from a causal LM.

    If use_kv_cache=True:
      - One initial forward on the full prompt (untimed),
      - Then one-token incremental decoding with past_key_values.
    '''
    device = next(model.parameters()).device
    ids = input_ids.to(device)

    if max_new_tokens <= 0:
        return ids, 0.0

    # ---------- non-KV path (for completeness) ---------- #
    if not use_kv_cache:
        start_time = time.time()
        for _ in range(max_new_tokens):
            if context_len is not None and ids.size(1) > context_len:
                input_cond = ids[:, -context_len:]
            else:
                input_cond = ids

            with torch.no_grad():
                out = model(input_ids=input_cond)
                logits = out.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        duration = time.time() - start_time
        return ids, duration

    # ---------- KV-cache path ---------- #
    # Initial forward on the full prompt (not timed)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        logits = out.logits[:, -1, :]          # logits for next token
        past = out.past_key_values

    start_time = time.time()
    new_tokens = 0

    while new_tokens < max_new_tokens:
        cur_logits = logits
        if temperature != 1.0:
            cur_logits = cur_logits / temperature

        cur_logits = top_k_top_p_filter(cur_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(cur_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        ids = torch.cat([ids, next_token], dim=1)
        new_tokens += 1

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # One-step incremental forward
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

    duration = time.time() - start_time
    return ids, duration


# ----------------- speculative decoding (no KV for target) ----------------- #

def speculative_sampling(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens,
    gamma=5,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    accept_mode="prob",           # 'prob', 'pt_gt_pd', or 'match'
    target_context_len=None,
    draft_context_len=None,
    eos_token_id=None,
    debug=False,
    log_per_position=False,
):
    '''
    Speculative decoding for causal LMs with acceptance computed on FULL distributions.

    Design:
      - draft_model runs on full prefix each round (no KV cache) to propose up to gamma tokens;
      - target_model runs on prefix + proposed window to verify (no KV cache);
      - acceptance uses either:
          * 'prob':     accept with prob min(1, Pt/Pd)
          * 'pt_gt_pd': accept deterministically if Pt > Pd
          * 'match':    accept only if argmax_t == sampled_d

    If log_per_position=True, returns an extra list of per-token decision dicts.
    '''
    device_target = next(target_model.parameters()).device
    device_draft = next(draft_model.parameters()).device

    ids = input_ids.to(device_target)
    T = ids.size(1) + max_new_tokens
    prompt_len = ids.size(1)

    accepted_count = 0
    total_draft_tokens = 0
    block_idx = 0
    per_position_log = []  # populated only when log_per_position=True

    start_time = time.time()

    while ids.size(1) < T:
        # --------- 1. Draft phase --------- #
        draft_tokens = []
        draft_full_probs = []  # list of P_d (full softmax, no top-k/top-p)
        curr_ids = ids.to(device_draft)  # no clone, just move to draft device

        for _ in range(gamma):
            if curr_ids.size(1) >= T:
                break

            if draft_context_len is not None and curr_ids.size(1) > draft_context_len:
                draft_input = curr_ids[:, -draft_context_len:]
            else:
                draft_input = curr_ids

            with torch.no_grad():
                out = draft_model(input_ids=draft_input)
                logits = out.logits[:, -1, :]  # [1, V]

            if temperature != 1.0:
                logits = logits / temperature

            # FULL distribution for acceptance
            p_d_full = F.softmax(logits, dim=-1)

            # Filtered distribution for candidate sampling
            filtered_logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            p_d_sample = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(p_d_sample, num_samples=1)
            token_id = next_token.item()

            draft_tokens.append(next_token.to(device_target))
            draft_full_probs.append(p_d_full.to(device_target))

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and token_id == eos_token_id:
                break

        if not draft_tokens:
            # fallback: one target step
            if target_context_len is not None and ids.size(1) > target_context_len:
                target_input = ids[:, -target_context_len:]
            else:
                target_input = ids

            with torch.no_grad():
                out = target_model(input_ids=target_input)
                logits = out.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            filtered_logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            p_t_sample = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(p_t_sample, num_samples=1)
            ids = torch.cat([ids, next_token.to(device_target)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            continue

        num_draft = len(draft_tokens)
        total_draft_tokens += num_draft
        block_idx += 1

        # --------- 2. Target verification (one forward) --------- #
        draft_seq = torch.cat(draft_tokens, dim=1)  # [1, num_draft]
        target_input_full = torch.cat([ids, draft_seq], dim=1)

        if target_context_len is not None and target_input_full.size(1) > target_context_len:
            target_input = target_input_full[:, -target_context_len:]
        else:
            target_input = target_input_full

        with torch.no_grad():
            out_t = target_model(input_ids=target_input)
            target_logits = out_t.logits  # [1, L, V]

        # We only need last num_draft + 1 positions
        relevant_logits = target_logits[:, -(num_draft + 1):, :]  # [1, num_draft+1, V]

        # --------- 3. Verify token by token --------- #
        all_accepted = True
        reached_eos = False

        for i in range(num_draft):
            token = draft_tokens[i]
            token_id = token.item()
            p_d_full = draft_full_probs[i]  # [1, V]

            # FULL target logits for this position
            logits_i = relevant_logits[:, i, :]  # [1, V]
            if temperature != 1.0:
                logits_i = logits_i / temperature
            p_t_full = F.softmax(logits_i, dim=-1)

            # filtered version of P_t for resampling
            logits_i_filt = top_k_top_p_filter(logits_i, top_k=top_k, top_p=top_p)
            p_t_sample = F.softmax(logits_i_filt, dim=-1)

            p_d_token = p_d_full[0, token_id].item()
            p_t_token = p_t_full[0, token_id].item()
            p_d_token = max(p_d_token, 1e-12)  # avoid division by zero
            # position in generated sequence (0-indexed from start of generation)
            _seq_pos = ids.size(1) - prompt_len

            if accept_mode == "prob":
                # Original rule: accept with probability min(1, Pt/Pd)
                r = torch.rand(1).item()
                acc_prob = min(1.0, p_t_token / p_d_token)

                if r < acc_prob:
                    ids = torch.cat([ids, token], dim=1)
                    accepted_count += 1
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": acc_prob})
                    if debug:
                        print(
                            f"[prob] step {i}: accept {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e}, "
                            f"r={r:.3f}, acc={acc_prob:.3f})"
                        )
                    if eos_token_id is not None and token_id == eos_token_id:
                        reached_eos = True
                        break
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": acc_prob})
                    if debug:
                        print(
                            f"[prob] step {i}: reject {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e}, "
                            f"r={r:.3f}, acc={acc_prob:.3f})"
                        )

                    # residual distribution: (Pt - Pd)_+
                    residual = torch.clamp(p_t_full - p_d_full, min=0.0)
                    residual_sum = residual.sum(dim=-1, keepdim=True)
                    if residual_sum.item() > 0:
                        residual = residual / residual_sum
                        new_token = torch.multinomial(residual, num_samples=1)
                    else:
                        # fallback: sample from target (filtered) dist
                        new_token = torch.multinomial(p_t_sample, num_samples=1)

                    ids = torch.cat([ids, new_token.to(device_target)], dim=1)
                    if debug:
                        print(f"   -> resampled: {new_token.item()}")

                    if eos_token_id is not None and new_token.item() == eos_token_id:
                        reached_eos = True
                    break

            elif accept_mode == "pt_gt_pd":
                if p_t_token > p_d_token:
                    ids = torch.cat([ids, token], dim=1)
                    accepted_count += 1
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token})
                    if debug:
                        print(
                            f"[pt_gt_pd] step {i}: accept {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e})"
                        )
                    if eos_token_id is not None and token_id == eos_token_id:
                        reached_eos = True
                        break
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token})
                    if debug:
                        print(
                            f"[pt_gt_pd] step {i}: reject {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e})"
                        )

                    residual = torch.clamp(p_t_full - p_d_full, min=0.0)
                    residual_sum = residual.sum(dim=-1, keepdim=True)
                    if residual_sum.item() > 0:
                        residual = residual / residual_sum
                        new_token = torch.multinomial(residual, num_samples=1)
                    else:
                        new_token = torch.multinomial(p_t_sample, num_samples=1)

                    ids = torch.cat([ids, new_token.to(device_target)], dim=1)
                    if debug:
                        print(f"   -> resampled: {new_token.item()}")

                    if eos_token_id is not None and new_token.item() == eos_token_id:
                        reached_eos = True
                    break

            elif accept_mode == "match":
                # Deterministic: accept only if argmax match
                target_argmax = logits_i.argmax(dim=-1, keepdim=True)  # argmax of full_logits
                target_token_id = target_argmax.item()

                if token_id == target_token_id:
                    ids = torch.cat([ids, token], dim=1)
                    accepted_count += 1
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token})
                    if debug:
                        print(f"[match] step {i}: accept {token_id}")
                    if eos_token_id is not None and token_id == eos_token_id:
                        reached_eos = True
                        break
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token})
                    ids = torch.cat([ids, target_argmax.to(device_target)], dim=1)
                    if debug:
                        print(f"[match] step {i}: reject {token_id}, use {target_token_id}")
                    if eos_token_id is not None and target_token_id == eos_token_id:
                        reached_eos = True
                    break

            else:
                raise ValueError(f"Unknown accept_mode: {accept_mode}")

        # --------- 4. Bonus token (if all accepted & no EOS) --------- #
        if all_accepted and not reached_eos:
            bonus_logits = relevant_logits[:, num_draft, :]  # next position
            if temperature != 1.0:
                bonus_logits = bonus_logits / temperature
            bonus_logits = top_k_top_p_filter(bonus_logits, top_k=top_k, top_p=top_p)
            p_bonus = F.softmax(bonus_logits, dim=-1)

            bonus_token = torch.multinomial(p_bonus, num_samples=1)
            ids = torch.cat([ids, bonus_token.to(device_target)], dim=1)

            if debug:
                print(f"Bonus token: {bonus_token.item()}")

            if eos_token_id is not None and bonus_token.item() == eos_token_id:
                reached_eos = True

        if eos_token_id is not None and ids[0, -1].item() == eos_token_id:
            break

    duration = time.time() - start_time
    acceptance_rate = accepted_count / max(1, total_draft_tokens)
    if log_per_position:
        return ids, duration, acceptance_rate, per_position_log
    return ids, duration, acceptance_rate


# ----------------- main script ----------------- #

def clean_protein(text, eos_token_str):
    '''Strip EOS token and non-letter chars from decoded text.'''
    if eos_token_str:
        text = text.replace(eos_token_str, "")
    text = text.replace("\n", "")
    text = "".join(ch for ch in text if ch.isalpha())
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Speculative decoding with GPT2-style protein LMs "
                    "(truncated draft + KV-cache baselines, no-KV spec-dec)."
    )

    parser.add_argument(
        "--prompt", "-p", default="<|endoftext|>",
        help="Prompt string for generation (default: '<|endoftext|>')",
    )
    parser.add_argument(
        "--num_samples", "-ns", type=int, default=5,
        help="Number of samples for benchmarking",
    )
    parser.add_argument(
        "--num_tokens", "-nt", type=int, default=256,
        help="Number of new tokens to generate after the prompt",
    )

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=950,
                        help="Top-k for sampling (ProtGPT2 paper uses 950)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Top-p (nucleus) for sampling (0 disables)")

    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--dtype", default="float16",
                        help="Torch dtype for model weights on GPU "
                             "(e.g. float16, bfloat16, float32)")

    # model + draft config
    parser.add_argument("--target_model_name", default="nferruz/ProtGPT2",
                        help="Target (large) protein LM name (HF Hub id)")
    parser.add_argument("--draft_layers", type=int, default=8,
                        help="Number of transformer blocks from the bottom "
                             "to use in truncated draft (ignored if "
                             "--draft_layer_indices is set)")
    parser.add_argument(
        "--draft_layer_indices", type=str, default="",
        help="Comma-separated list of layer indices for the draft model "
             "(e.g. '0,12,24'). If set, overrides --draft_layers."
    )

    # speculative decoding hyperparams
    parser.add_argument("--gamma", "-L", type=int, default=5,
                        help="Speculation window size (draft tokens per round)")
    parser.add_argument("--accept_mode", choices=["prob", "pt_gt_pd", "match"],
                        default="prob",
                        help="Acceptance rule: prob (min(1, Pt/Pd)), pt_gt_pd, or match")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")

    # context windows (for non-cache target/draft; KV path ignores these)
    parser.add_argument("--target_context_len", type=int, default=None,
                        help="Max context length for target model "
                             "(default: model config if available)")
    parser.add_argument("--draft_context_len", type=int, default=None,
                        help="Max context length for draft model "
                             "(default: same as target_context_len)")

    parser.add_argument("--max_prompt_tokens", type=int, default=64,
                        help="Truncate encoded prompt to at most this many tokens "
                             "(0 disables)")

    args = parser.parse_args()

    if args.num_tokens <= 0:
        print(f"num_tokens must be > 0 (got {args.num_tokens}). Exiting.")
        return

    # device / dtype
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}, accept_mode: {args.accept_mode}")
    print(f"Draft layers (if no indices): {args.draft_layers}")
    if args.draft_layer_indices:
        print(f"Draft layer indices (raw): {args.draft_layer_indices}")

    # ----- tokenizer ----- #
    print(f"Loading tokenizer from {args.target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    eos_token_str = tokenizer.eos_token

    # ----- full (target) model ----- #
    print(f"Loading full target model: {args.target_model_name}")
    full_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)
    full_model.to(device=device, dtype=dtype)
    full_model.eval()
    target_model = full_model

    # ----- truncated draft model ----- #
    if not hasattr(full_model, "transformer") or not hasattr(full_model.transformer, "h"):
        raise ValueError(
            "Target model does not appear to be GPT2-style (no .transformer.h). "
            "Truncated draft construction won't work."
        )

    n_layers_full = len(full_model.transformer.h)
    print(f"Target model has {n_layers_full} transformer layers.")

    # Decide which layers to use for draft
    if args.draft_layer_indices:
        # parse comma-separated indices
        idx_strs = [s for s in args.draft_layer_indices.split(",") if s.strip() != ""]
        draft_indices = [int(s) for s in idx_strs]
        for idx in draft_indices:
            if idx < 0 or idx >= n_layers_full:
                raise ValueError(
                    f"draft_layer_indices contains invalid index {idx}; "
                    f"valid range is [0, {n_layers_full-1}]"
                )
        print(f"Using custom draft layer indices: {draft_indices}")
    else:
        if args.draft_layers <= 0 or args.draft_layers > n_layers_full:
            raise ValueError(
                f"draft_layers={args.draft_layers} is invalid; "
                f"must be in [1, {n_layers_full}]"
            )
        draft_indices = list(range(args.draft_layers))
        print(f"Using first {args.draft_layers} layers as draft: {draft_indices}")

    draft_model = TruncatedProtGPT2(full_model, draft_indices)
    draft_model.to(device=device, dtype=dtype)
    draft_model.eval()

    # ----- resolve context lengths (used only in spec-dec, non-KV path) ----- #
    if args.target_context_len is not None:
        target_context_len = args.target_context_len
    else:
        target_context_len = getattr(target_model.config, "n_positions", None)

    draft_context_len = args.draft_context_len or target_context_len  # same by default

    print(f"Target context length: {target_context_len}")
    print(f"Draft context length:  {draft_context_len}")

    # ----- encode prompt ----- #
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    if args.max_prompt_tokens > 0 and input_ids.size(1) > args.max_prompt_tokens:
        input_ids = input_ids[:, -args.max_prompt_tokens:]
        print(f"Truncated prompt to {args.max_prompt_tokens} tokens.")
    else:
        print(f"Prompt uses {input_ids.size(1)} tokens (no truncation).")

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Prompt token IDs length: {input_ids.size(1)}")
    print(f"Prompt IDs (first 20): {input_ids[0, :20].tolist()}")
    max_new_tokens = args.num_tokens
    print(f"Requested new tokens: {max_new_tokens}")

    # --------- WARM-UP (not timed) --------- #
    warmup_tokens = min(32, max_new_tokens)
    print("\nRunning warm-up for all modes (not timed)...")

    with torch.inference_mode():
        _ids, _ = generate_baseline(
            target_model,
            input_ids,
            warmup_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_len=target_context_len,
            eos_token_id=eos_id,
            use_kv_cache=True,
        )
        _ids, _ = generate_baseline(
            draft_model,
            input_ids,
            warmup_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_len=draft_context_len,
            eos_token_id=eos_id,
            use_kv_cache=True,
        )
        _ids, _, _ = speculative_sampling(
            target_model,
            draft_model,
            input_ids,
            warmup_tokens,
            gamma=args.gamma,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            accept_mode=args.accept_mode,
            target_context_len=target_context_len,
            draft_context_len=draft_context_len,
            eos_token_id=eos_id,
            debug=False,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up complete.\n")

    metrics = {
        "target": {"tokens": [], "time": []},
        "draft": {"tokens": [], "time": []},
        "specdec": {"tokens": [], "time": [], "acceptance_rate": []},
    }

    print("\n" + "=" * 50)
    print("Baseline & Speculative Decoding Benchmarks")
    print("=" * 50)

    # ---------- 1. Target baseline (with KV cache) ---------- #
    print("\nRunning target baseline (KV cache)...")
    with torch.inference_mode():
        for i in range(args.num_samples):
            ids, duration = generate_baseline(
                target_model,
                input_ids,
                max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                context_len=target_context_len,
                eos_token_id=eos_id,
                use_kv_cache=True,
            )
            n_new = ids.size(1) - input_ids.size(1)
            metrics["target"]["tokens"].append(n_new)
            metrics["target"]["time"].append(duration)
            tps = n_new / duration if duration > 0 else 0.0
            print(f"Target sample {i+1}: {n_new} tokens in {duration:.2f}s ({tps:.2f} tps)")

    # ---------- 2. Draft baseline (with KV cache) ---------- #
    print("\nRunning draft baseline (KV cache)...")
    with torch.inference_mode():
        for i in range(args.num_samples):
            ids, duration = generate_baseline(
                draft_model,
                input_ids,
                max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                context_len=draft_context_len,
                eos_token_id=eos_id,
                use_kv_cache=True,
            )
            n_new = ids.size(1) - input_ids.size(1)
            metrics["draft"]["tokens"].append(n_new)
            metrics["draft"]["time"].append(duration)
            tps = n_new / duration if duration > 0 else 0.0
            print(f"Draft sample {i+1}: {n_new} tokens in {duration:.2f}s ({tps:.2f} tps)")

    # ---------- 3. Speculative decoding (target w/o KV) ---------- #
    print("\nRunning speculative decoding...")
    with torch.inference_mode():
        for i in range(args.num_samples):
            ids, duration, acc_rate = speculative_sampling(
                target_model,
                draft_model,
                input_ids,
                max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                accept_mode=args.accept_mode,
                target_context_len=target_context_len,
                draft_context_len=draft_context_len,
                eos_token_id=eos_id,
                debug=args.debug,
            )
            n_new = ids.size(1) - input_ids.size(1)
            metrics["specdec"]["tokens"].append(n_new)
            metrics["specdec"]["time"].append(duration)
            metrics["specdec"]["acceptance_rate"].append(acc_rate)
            tps = n_new / duration if duration > 0 else 0.0
            print(
                f"SpecDec sample {i+1}: {n_new} tokens in {duration:.2f}s "
                f"({tps:.2f} tps) - acc rate: {acc_rate:.2f}"
            )

            if i == 0:
                decoded = tokenizer.decode(ids[0], skip_special_tokens=False)
                protein = clean_protein(decoded, eos_token_str)
                print("\nSpecDec Sample 1 decoded protein (cleaned):")
                print(f">specdec_sample_1\n{protein}")

    # ---------- summary ---------- #
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    def get_tps(key):
        total_tokens = sum(metrics[key]["tokens"])
        total_time = sum(metrics[key]["time"])
        return total_tokens / total_time if total_time > 0 else 0.0

    target_tps = get_tps("target")
    draft_tps = get_tps("draft")
    specdec_tps = get_tps("specdec")
    avg_acc_rate = (
        sum(metrics["specdec"]["acceptance_rate"])
        / max(1, len(metrics["specdec"]["acceptance_rate"]))
    )

    print(f"Target model TPS (KV): {target_tps:.2f}")
    print(f"Draft model TPS (KV):  {draft_tps:.2f}")
    print(f"Speculative TPS:       {specdec_tps:.2f}")
    if target_tps > 0:
        print(f"Speedup vs target:     {specdec_tps / target_tps:.2f}x")
    else:
        print("Speedup vs target:     N/A")
    print(f"Mean acceptance rate:  {avg_acc_rate:.2f}")
    print(f"Mean accepted prefix:  {avg_acc_rate * args.gamma:.2f} tokens")


if __name__ == "__main__":
    main()
