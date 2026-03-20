#!/usr/bin/env python
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# Optional matmul speedups on newer GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------- helpers to grab ProGen stack + truncated draft ----------------- #

def get_progen_stack(model: nn.Module):
    """
    Try to locate the GPT2-like stack (Module with .h = ModuleList[...]).

    For ProGen2, target_model.transformer is a ProGenModel with .h.
    If that fails for some reason, we fall back to scanning submodules.
    """
    # Direct ProGen2 path
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer

    # A few generic fallbacks (shouldn't be needed for ProGen2)
    for attr in ["model", "base_model"]:
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "transformer") and hasattr(sub.transformer, "h"):
            return sub.transformer

    for _, module in model.named_modules():
        if hasattr(module, "h") and isinstance(getattr(module, "h"), nn.ModuleList):
            return module

    raise ValueError(
        "Model does not expose a GPT2/ProGen-like stack with '.h' ModuleList; "
        "cannot build truncated draft."
    )


class TruncatedProGenDraft(nn.Module):
    """
    Truncated draft model built from a full ProGen2 CausalLM.

    - Shares:
        * embeddings (wte)
        * dropout
        * a subset of transformer blocks (by layer indices)
        * final layer norm (ln_f)
        * lm_head
    - Supports KV cache so it can be used in baselines and KV-aware spec-dec.
    """

    def __init__(self, full_model: nn.Module, layer_indices):
        super().__init__()
        self.config = full_model.config

        stack = get_progen_stack(full_model)  # ProGenModel
        self.wte = stack.wte
        self.drop = stack.drop
        self.h = nn.ModuleList([stack.h[i] for i in layer_indices])
        self.ln_f = stack.ln_f

        self.lm_head = full_model.lm_head

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Minimal ProGen-style forward with optional KV cache.

        input_ids: [B, T_new]
        past_key_values: tuple/list of length len(self.h), each (k,v) or None.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        input_shape = input_ids.size()
        batch_size, seq_len = input_shape[:2]

        # Normalize past_key_values to a list of length len(self.h)
        if past_key_values is None:
            past_list = [None] * len(self.h)
        else:
            if isinstance(past_key_values, tuple):
                past_list = list(past_key_values)
            else:
                past_list = list(past_key_values)

            if len(past_list) < len(self.h):
                past_list = past_list + [None] * (len(self.h) - len(past_list))
            elif len(past_list) > len(self.h):
                past_list = past_list[:len(self.h)]

        past_key_values = past_list

        # No separate position embeddings: rotary is handled inside attention.
        inputs_embeds = self.wte(input_ids)        # [B, T_new, D]
        hidden_states = self.drop(inputs_embeds)

        next_past = [] if use_cache else None

        for block, layer_past in zip(self.h, past_key_values):
            # ProGenBlock returns (hidden_states, present, (attentions?))
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,
                use_cache=use_cache,
                output_attentions=False,
            )

            hidden_states = outputs[0]
            if use_cache:
                # outputs[1] is present = (k, v)
                next_past.append(outputs[1])

        if use_cache:
            next_past = tuple(next_past)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=next_past if use_cache else None,
        )


# ----------------- sampling utils ----------------- #

def top_k_top_p_filter(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.

    logits: (..., vocab_size)
    returns: filtered logits with some entries set to filter_value
    """
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


# ----------------- baseline generation (with optional KV cache) ----------------- #

def _model_input_device(model: nn.Module) -> torch.device:
    """Best-effort device for inputs when a model is sharded via HF `device_map`."""
    try:
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb):
            emb = get_emb()
            w = getattr(emb, "weight", None)
            if w is not None:
                return w.device
    except Exception:
        pass
    return next(model.parameters()).device

def generate_baseline(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    eos_token_id=None,       # ProGen2 usually has no EOS; pass None
    use_kv_cache=True,
):
    """
    Standard autoregressive generation with sampling from a causal LM.

    If use_kv_cache=True (default), uses HuggingFace KV cache:
      - One initial forward on the full prompt (untimed),
      - Then one-token incremental decoding with past_key_values.
    """
    device = _model_input_device(model)
    ids = input_ids.to(device).clone()

    if max_new_tokens <= 0:
        return ids, 0.0

    # non-cache path (mainly for debugging / if use_cache isn't supported)
    if not use_kv_cache:
        start_time = time.time()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=ids)
                logits = out.logits[:, -1, :].to(ids.device)  # [1, V]

            logits_temp = logits
            if temperature != 1.0:
                logits_temp = logits_temp / temperature

            logits_temp = top_k_top_p_filter(logits_temp.clone(), top_k=top_k, top_p=top_p)
            probs = F.softmax(logits_temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1,1]

            ids = torch.cat([ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        duration = time.time() - start_time
        return ids, duration

    # ---------- KV cache path ---------- #
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        logits = out.logits[:, -1, :].to(ids.device)          # [1, V]
        past = out.past_key_values

    start_time = time.time()
    new_tokens = 0

    while new_tokens < max_new_tokens:
        logits_temp = logits
        if temperature != 1.0:
            logits_temp = logits_temp / temperature

        logits_temp = top_k_top_p_filter(logits_temp.clone(), top_k=top_k, top_p=top_p)
        probs = F.softmax(logits_temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1,1]

        ids = torch.cat([ids, next_token], dim=1)
        new_tokens += 1

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :].to(ids.device)
            past = out.past_key_values

    duration = time.time() - start_time
    return ids, duration


# ----------------- speculative decoding (no KV in the loop) ----------------- #

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
    eos_token_id=None,
    debug=False,
    log_per_position=False,
):
    """
    Speculative decoding for two arbitrary causal LMs
    (e.g. ProGen2-base as target, ProGen2-small or truncated target as draft).

    Assumptions:
      - Both models share the same tokenizer / vocabulary.
      - We do NOT use KV-cache inside this speculative loop.

    Algorithm:
      1) Draft proposes up to gamma tokens, sampling from its own filtered
         distribution, but we keep the FULL draft probs for acceptance.
      2) Target scores the entire prefix + proposed window in one forward.
      3) Token-by-token, we accept/reject using:
         - prob:  accept w.p. min(1, Pt/Pd)
         - pt_gt_pd: accept if Pt > Pd
         - match: accept if argmax match
      4) If all accepted, we also sample one bonus token from target.

    If log_per_position=True, returns an extra list of per-token decision dicts.
    """
    device_target = _model_input_device(target_model)
    device_draft = _model_input_device(draft_model)

    ids = input_ids.to(device_target).clone()
    T = ids.size(1) + max_new_tokens
    prompt_len = ids.size(1)

    accepted_count = 0
    total_draft_tokens = 0
    block_idx = 0
    per_position_log = []

    start_time = time.time()

    while ids.size(1) < T:
        # --------- 1. Draft phase --------- #
        draft_tokens = []
        draft_full_probs = []  # full P_d over vocab at each step
        curr_ids = ids.to(device_draft).clone()

        for _ in range(gamma):
            if curr_ids.size(1) >= T:
                break

            with torch.no_grad():
                out = draft_model(input_ids=curr_ids)
                logits = out.logits[:, -1, :].to(curr_ids.device)  # [1, V]

            if temperature != 1.0:
                logits = logits / temperature

            # FULL draft distribution for acceptance
            p_d_full = F.softmax(logits, dim=-1)

            # Filtered distribution for sampling
            filtered_logits = top_k_top_p_filter(
                logits.clone(),
                top_k=top_k,
                top_p=top_p,
            )
            p_d_sample = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(p_d_sample, num_samples=1)  # [1,1]
            token_id = next_token.item()

            draft_tokens.append(next_token.to(ids.device))
            draft_full_probs.append(p_d_full.to(ids.device))

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and token_id == eos_token_id:
                break

        if not draft_tokens:
            # fallback: 1 target step
            with torch.no_grad():
                out = target_model(input_ids=ids)
                logits = out.logits[:, -1, :].to(ids.device)

            if temperature != 1.0:
                logits = logits / temperature

            logits_f = top_k_top_p_filter(logits.clone(), top_k=top_k, top_p=top_p)
            p_t_sample = F.softmax(logits_f, dim=-1)
            next_token = torch.multinomial(p_t_sample, num_samples=1)

            ids = torch.cat([ids, next_token.to(ids.device)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            continue

        num_draft = len(draft_tokens)
        total_draft_tokens += num_draft
        block_idx += 1

        # --------- 2. Target verification (single forward) --------- #
        draft_seq = torch.cat(draft_tokens, dim=1)  # [1, num_draft]
        target_input_full = torch.cat([ids, draft_seq], dim=1)

        with torch.no_grad():
            out_t = target_model(input_ids=target_input_full)
            target_logits = out_t.logits  # [1, L, V]

        relevant_logits = target_logits[:, -(num_draft + 1):, :].to(ids.device)  # [1, num_draft+1, V]

        # --------- 3. Acceptance loop --------- #
        all_accepted = True
        reached_eos = False

        for i in range(num_draft):
            token = draft_tokens[i]
            token_id = token.item()
            p_d_full = draft_full_probs[i]  # [1, V]

            logits_i = relevant_logits[:, i, :]  # [1, V]
            if temperature != 1.0:
                logits_i = logits_i / temperature
            p_t_full = F.softmax(logits_i, dim=-1)

            logits_i_filt = top_k_top_p_filter(
                logits_i.clone(),
                top_k=top_k,
                top_p=top_p,
            )
            p_t_sample = F.softmax(logits_i_filt, dim=-1)

            p_d_token = p_d_full[0, token_id].item()
            p_t_token = p_t_full[0, token_id].item()
            p_d_token = max(p_d_token, 1e-12)
            _seq_pos = ids.size(1) - prompt_len

            if accept_mode == "prob":
                r = torch.rand(1).item()
                acc_prob = min(1.0, p_t_token / p_d_token)

                if r < acc_prob:
                    # accept
                    ids = torch.cat([ids, token], dim=1)
                    accepted_count += 1
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": acc_prob})
                    if debug:
                        print(
                            f"[prob] step {i}: ACCEPT {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e}, r={r:.3f}, acc={acc_prob:.3f})"
                        )
                    if eos_token_id is not None and token_id == eos_token_id:
                        reached_eos = True
                        break
                else:
                    # reject -> residual sample
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token, "acc_prob": acc_prob})
                    if debug:
                        print(
                            f"[prob] step {i}: REJECT {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e}, r={r:.3f}, acc={acc_prob:.3f})"
                        )

                    residual = torch.clamp(p_t_full - p_d_full, min=0.0)
                    residual_sum = residual.sum(dim=-1, keepdim=True)
                    if residual_sum.item() > 0:
                        residual = residual / residual_sum
                        new_token = torch.multinomial(residual, num_samples=1)
                    else:
                        new_token = torch.multinomial(p_t_sample, num_samples=1)

                    ids = torch.cat([ids, new_token.to(ids.device)], dim=1)
                    if debug:
                        print(f"   -> residual token: {new_token.item()}")

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
                            f"[pt_gt_pd] step {i}: ACCEPT {token_id} "
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
                            f"[pt_gt_pd] step {i}: REJECT {token_id} "
                            f"(Pd={p_d_token:.3e}, Pt={p_t_token:.3e})"
                        )

                    residual = torch.clamp(p_t_full - p_d_full, min=0.0)
                    residual_sum = residual.sum(dim=-1, keepdim=True)
                    if residual_sum.item() > 0:
                        residual = residual / residual_sum
                        new_token = torch.multinomial(residual, num_samples=1)
                    else:
                        new_token = torch.multinomial(p_t_sample, num_samples=1)

                    ids = torch.cat([ids, new_token.to(ids.device)], dim=1)
                    if debug:
                        print(f"   -> residual token: {new_token.item()}")
                    if eos_token_id is not None and new_token.item() == eos_token_id:
                        reached_eos = True
                    break

            elif accept_mode == "match":
                target_argmax = logits_i.argmax(dim=-1, keepdim=True)
                target_token_id = target_argmax.item()
                if token_id == target_token_id:
                    ids = torch.cat([ids, token], dim=1)
                    accepted_count += 1
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": True, "p_d": p_d_token, "p_t": p_t_token})
                    if debug:
                        print(f"[match] step {i}: ACCEPT {token_id}")
                    if eos_token_id is not None and token_id == eos_token_id:
                        reached_eos = True
                        break
                else:
                    all_accepted = False
                    if log_per_position:
                        per_position_log.append({"block": block_idx, "pos_in_block": i, "seq_pos": _seq_pos, "token_id": token_id, "accepted": False, "p_d": p_d_token, "p_t": p_t_token})
                    ids = torch.cat([ids, target_argmax.to(ids.device)], dim=1)
                    if debug:
                        print(
                            f"[match] step {i}: REJECT {token_id}, "
                            f"use {target_token_id}"
                        )
                    if eos_token_id is not None and target_token_id == eos_token_id:
                        reached_eos = True
                    break

            else:
                raise ValueError(f"Unknown accept_mode: {accept_mode}")

        # --------- 4. Bonus token if all accepted --------- #
        if all_accepted and not reached_eos:
            bonus_logits = relevant_logits[:, num_draft, :]
            if temperature != 1.0:
                bonus_logits = bonus_logits / temperature
            bonus_logits = top_k_top_p_filter(
                bonus_logits.clone(), top_k=top_k, top_p=top_p
            )
            p_bonus = F.softmax(bonus_logits, dim=-1)
            bonus_token = torch.multinomial(p_bonus, num_samples=1)
            ids = torch.cat([ids, bonus_token.to(ids.device)], dim=1)

            if debug:
                print(f"[bonus] token: {bonus_token.item()}")

            if eos_token_id is not None and bonus_token.item() == eos_token_id:
                reached_eos = True

        if eos_token_id is not None and ids[0, -1].item() == eos_token_id:
            break

    duration = time.time() - start_time
    acceptance_rate = accepted_count / max(1, total_draft_tokens)
    if log_per_position:
        return ids, duration, acceptance_rate, per_position_log
    return ids, duration, acceptance_rate


# ----------------- util: clean protein ----------------- #

def clean_protein(text: str):
    """
    Strip newlines and keep only letters (A–Z, a–z).
    ProGen2 sequences also contain '1' and '2' tokens; those will be dropped
    here so you see just the amino-acid string.
    """
    text = text.replace("\n", "")
    text = "".join(ch for ch in text if ch.isalpha())
    return text


# ----------------- main ----------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Speculative decoding with ProGen2 (pretrained or truncated draft, no KV in spec-dec loop)"
    )

    parser.add_argument(
        "--prompt", "-p", default="1M",
        help="Prompt string for generation (include tags + '1' if you follow ProGen2 format).",
    )
    parser.add_argument(
        "--num_samples", "-ns", type=int, default=5,
        help="Number of samples for benchmarking.",
    )
    parser.add_argument(
        "--num_tokens", "-nt", type=int, default=256,
        help="Number of new tokens to generate after the prompt.",
    )

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=950,
                        help="Top-k for sampling.")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Top-p (nucleus) for sampling (0 disables).")

    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--dtype", default="float16",
                        help="Torch dtype on GPU: float16, bfloat16, float32.")

    parser.add_argument(
        "--target_model_name",
        default="hugohrban/progen2-base",
        help="Target ProGen2 model (e.g. hugohrban/progen2-base).",
    )
    parser.add_argument(
        "--draft_mode",
        choices=["pretrained", "truncated"],
        default="pretrained",
        help="Draft type: separate pretrained model or truncated view of target.",
    )
    parser.add_argument(
        "--draft_model_name",
        default="hugohrban/progen2-small",
        help="Draft ProGen2 model (used only if --draft_mode=pretrained).",
    )
    parser.add_argument(
        "--draft_layers",
        type=int,
        default=6,
        help="Number of bottom layers to use in truncated draft (ignored if --draft_layer_indices is set).",
    )
    parser.add_argument(
        "--draft_layer_indices",
        type=str,
        default="",
        help="Comma-separated list of layer indices for truncated draft (e.g. '0,4,8'). "
             "If set, overrides --draft_layers.",
    )

    parser.add_argument("--gamma", "-L", type=int, default=5,
                        help="Speculation window size.")
    parser.add_argument("--accept_mode", choices=["prob", "pt_gt_pd", "match"],
                        default="prob",
                        help="Acceptance rule.")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging.")

    parser.add_argument("--max_prompt_tokens", type=int, default=128,
                        help="Truncate encoded prompt to at most this many tokens (0 disables).")

    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Disable KV-cache in baselines (spec-dec is non-KV anyway).")

    args = parser.parse_args()

    if args.num_tokens <= 0:
        print(f"num_tokens must be > 0 (got {args.num_tokens}). Exiting.")
        return

    # ----- device / dtype ----- #
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}, accept_mode: {args.accept_mode}")
    print(f"Target model: {args.target_model_name}")
    print(f"Draft mode:   {args.draft_mode}")
    if args.draft_mode == "pretrained":
        print(f"Draft model:  {args.draft_model_name}")
    else:
        if args.draft_layer_indices:
            print(f"Truncated draft layer indices: {args.draft_layer_indices}")
        else:
            print(f"Truncated draft first {args.draft_layers} layers.")

    # ----- tokenizer (always from target) ----- #
    print(f"Loading tokenizer from {args.target_model_name}")
    tokenizer = Tokenizer.from_pretrained(args.target_model_name)
    tokenizer.no_padding()

    # ----- target model ----- #
    print(f"Loading target model: {args.target_model_name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name,
        trust_remote_code=True,
    )
    target_model.to(device=device, dtype=dtype)
    target_model.eval()

    # ----- draft model (pretrained or truncated) ----- #
    if args.draft_mode == "pretrained":
        print(f"Loading draft model:  {args.draft_model_name}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model_name,
            trust_remote_code=True,
        )
    else:
        print("Building truncated draft from target model...")
        stack = get_progen_stack(target_model)
        n_layers_full = len(stack.h)
        if args.draft_layer_indices:
            idx_strs = [s for s in args.draft_layer_indices.split(",") if s.strip() != ""]
            layer_indices = [int(s) for s in idx_strs]
            for idx in layer_indices:
                if idx < 0 or idx >= n_layers_full:
                    raise ValueError(
                        f"draft_layer_indices contains invalid index {idx}; "
                        f"valid range is [0, {n_layers_full - 1}]"
                    )
        else:
            if args.draft_layers <= 0 or args.draft_layers > n_layers_full:
                raise ValueError(
                    f"draft_layers={args.draft_layers} is invalid; must be in [1, {n_layers_full}]"
                )
            layer_indices = list(range(args.draft_layers))
        print(f"Using truncated draft layers: {layer_indices}")
        draft_model = TruncatedProGenDraft(target_model, layer_indices)

    draft_model.to(device=device, dtype=dtype)
    draft_model.eval()

    # ----- encode prompt ----- #
    prompt_ids = tokenizer.encode(args.prompt).ids  # list[int]
    if args.max_prompt_tokens > 0 and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens:]
        print(f"Truncated prompt to {args.max_prompt_tokens} tokens.")

    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Prompt token IDs length: {input_ids.size(1)}")
    print(f"Prompt IDs (first 20): {input_ids[0, :20].tolist()}")
    max_new_tokens = args.num_tokens
    print(f"Requested new tokens: {max_new_tokens}")

    eos_id = None  # ProGen2 uses '2' char at end, but not as a standard EOS token ID

    # --------- warm-up (not timed) --------- #
    warmup_tokens = min(32, max_new_tokens)
    print("\nRunning warm-up for all modes (not timed)...")
    _ids, _ = generate_baseline(
        target_model,
        input_ids,
        warmup_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_id,
        use_kv_cache=not args.no_kv_cache,
    )
    _ids, _ = generate_baseline(
        draft_model,
        input_ids,
        warmup_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_id,
        use_kv_cache=not args.no_kv_cache,
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
    print("Baseline & Speculative Decoding Benchmarks (ProGen2)")
    print("=" * 50)

    # ---------- 1. Target baseline ---------- #
    print("\nRunning target baseline (KV cache = %s)..." % (not args.no_kv_cache))
    for i in range(args.num_samples):
        ids, duration = generate_baseline(
            target_model,
            input_ids,
            max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
            use_kv_cache=not args.no_kv_cache,
        )
        n_new = ids.size(1) - input_ids.size(1)
        metrics["target"]["tokens"].append(n_new)
        metrics["target"]["time"].append(duration)
        tps = n_new / duration if duration > 0 else 0.0
        print(f"Target sample {i+1}: {n_new} tokens in {duration:.2f}s ({tps:.2f} tps)")

    # ---------- 2. Draft baseline ---------- #
    print("\nRunning draft baseline (KV cache = %s)..." % (not args.no_kv_cache))
    for i in range(args.num_samples):
        ids, duration = generate_baseline(
            draft_model,
            input_ids,
            max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
            use_kv_cache=not args.no_kv_cache,
        )
        n_new = ids.size(1) - input_ids.size(1)
        metrics["draft"]["tokens"].append(n_new)
        metrics["draft"]["time"].append(duration)
        tps = n_new / duration if duration > 0 else 0.0
        print(f"Draft sample {i+1}: {n_new} tokens in {duration:.2f}s ({tps:.2f} tps)")

    # ---------- 3. Speculative decoding ---------- #
    print("\nRunning speculative decoding (no KV in loop)...")
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
            decoded = tokenizer.decode(ids[0].tolist())
            protein = clean_protein(decoded)
            print("\nSpecDec Sample 1 decoded protein (cleaned AAs only):")
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

    print(f"Target model TPS:      {target_tps:.2f}")
    print(f"Draft model TPS:       {draft_tps:.2f}")
    print(f"Speculative TPS:       {specdec_tps:.2f}")
    if target_tps > 0:
        print(f"Speedup vs target:     {specdec_tps / target_tps:.2f}x")
    else:
        print("Speedup vs target:     N/A")
    print(f"Mean acceptance rate:  {avg_acc_rate:.2f}")
    print(f"Mean accepted prefix:  {avg_acc_rate * args.gamma:.2f} tokens")


if __name__ == "__main__":
    main()

# python specdec_progen2_trunc_nokv.py \
#   --target_model_name hugohrban/progen2-base \
#   --draft_mode pretrained \
#   --draft_model_name hugohrban/progen2-small \
#   --prompt "1M" --num_tokens 256 --num_samples 5 --gamma 5

#   python specdec_progen2_truncated.py \
#   --target_model_name hugohrban/progen2-base \
#   --draft_mode truncated \
#   --draft_layers 6 \
#   --prompt "1M" --num_tokens 256 --num_samples 5 --gamma 5

