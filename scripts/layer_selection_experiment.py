#!/usr/bin/env python3
"""
Layer-selection ablation for reviewer comment 2.a.1:
Compare first-N, last-N, and non-contiguous layer selection strategies
for truncated draft models across ProtGPT2 and ProGen2.

Measures:
  - Mean acceptance rate (from speculative decoding)
  - Mean KL divergence D_KL(target || draft) over autoregressive positions
  - Top-1 agreement between target and draft argmax predictions

Usage (single GPU):
  python scripts/layer_selection_experiment.py --model protgpt2 --device cuda
  python scripts/layer_selection_experiment.py --model progen2  --device cuda

  # or both:
  python scripts/layer_selection_experiment.py --model both --device cuda

Output:
  results/layer_selection/protgpt2_layer_selection.csv
  results/layer_selection/progen2_layer_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# TF32 causes NaN in ProGen2-xlarge on A100s
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── Make repo-local imports work ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return model.transformer.wte.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


class LayerSubsetModel(nn.Module):
    """Wraps a GPT2/ProGen2-like CausalLM and uses only a subset of layers.

    Instead of reimplementing the forward pass (fragile across transformers
    versions), we temporarily swap the model's layer list during forward
    so the model's own forward method handles everything correctly.
    """

    def __init__(self, full_model: nn.Module, layer_indices: List[int]):
        super().__init__()
        self.full_model = full_model
        self.layer_indices = list(layer_indices)

        # Find the transformer block list (GPT2: .transformer.h, ProGen2: .transformer.h)
        stack = self._find_stack(full_model)
        self._stack_parent = stack
        self._original_h = stack.h
        self._subset_h = nn.ModuleList([stack.h[i] for i in layer_indices])

        # Expose config for compatibility
        self.config = full_model.config

    @staticmethod
    def _find_stack(model: nn.Module):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer
        for _, m in model.named_modules():
            if hasattr(m, "h") and isinstance(getattr(m, "h"), nn.ModuleList):
                return m
        raise ValueError("Cannot find transformer block list (.h)")

    def forward(self, input_ids=None, **kwargs):
        # Temporarily swap layers via _modules to bypass nn.Module registration
        self._stack_parent._modules["h"] = self._subset_h
        try:
            # Explicitly disable KV cache to avoid stale state
            kwargs.setdefault("use_cache", False)
            return self.full_model(input_ids=input_ids, **kwargs)
        finally:
            self._stack_parent._modules["h"] = self._original_h

    def get_input_embeddings(self):
        return self.full_model.get_input_embeddings()

    def parameters(self, recurse=True):
        return self.full_model.parameters(recurse=recurse)


def top_k_top_p_filter(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k, dim=-1)
        logits = torch.where(logits < vals[..., -1, None],
                             torch.full_like(logits, filter_value), logits)
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_idx, src=remove)
        logits = logits.masked_fill(mask, filter_value)
    return logits


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace NaN/inf in logits with zeros so softmax/multinomial won't crash."""
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.where(torch.isfinite(logits), logits,
                             torch.zeros_like(logits))
    return logits


def _safe_multinomial(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """multinomial that falls back to uniform if probs are degenerate."""
    probs = probs.clone()
    probs[probs != probs] = 0  # NaN -> 0
    probs = probs.clamp(min=0)
    total = probs.sum(dim=-1, keepdim=True)
    if (total <= 0).any():
        probs = torch.ones_like(probs)
    return torch.multinomial(probs, num_samples=num_samples)


def build_layer_indices(n_total: int, n_draft: int, strategy: str) -> List[int]:
    """Return layer indices for a given strategy.

    Strategies:
      first  – first n_draft layers:  [0, 1, ..., n_draft-1]
      last   – last n_draft layers:   [n_total-n_draft, ..., n_total-1]
      mixed  – first layer, evenly-spaced intermediates, last layer
               (guarantees first and last are always included)
    """
    if n_draft >= n_total:
        return list(range(n_total))

    if strategy == "first":
        return list(range(n_draft))

    if strategy == "last":
        return list(range(n_total - n_draft, n_total))

    if strategy == "mixed":
        if n_draft == 1:
            return [0]
        if n_draft == 2:
            return [0, n_total - 1]
        # first + evenly-spaced interior + last
        interior_count = n_draft - 2
        step = (n_total - 1) / (interior_count + 1)
        indices = [0]
        for i in range(1, interior_count + 1):
            idx = int(round(i * step))
            if idx not in indices and idx != n_total - 1:
                indices.append(idx)
        # pad if rounding collapsed any
        while len(indices) < n_draft - 1:
            for candidate in range(1, n_total - 1):
                if candidate not in indices:
                    indices.append(candidate)
                    break
        indices.append(n_total - 1)
        indices = sorted(set(indices))[:n_draft]
        return indices

    raise ValueError(f"Unknown strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════════
# Distributional alignment metrics
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def measure_alignment(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> Dict[str, float]:
    """Generate tokens from the *target* model and measure how well the draft
    model's distributions align at each step.

    Returns dict with:
      kl_div_mean     – mean KL(target || draft) over positions
      top1_agreement  – fraction of positions where argmax matches
      mean_prob_ratio – mean P_draft(token) / P_target(token) for the target's chosen token
    """
    device_t = _model_input_device(target_model)
    device_d = _model_input_device(draft_model)

    ids = input_ids.to(device_t).clone()
    kl_divs = []
    top1_matches = 0
    prob_ratios = []
    n_positions = 0

    for _ in range(max_new_tokens):
        try:
            # Target forward
            out_t = target_model(input_ids=ids.to(device_t), use_cache=False)
            logits_t = _sanitize_logits(out_t.logits[:, -1, :].float())
            if temperature != 1.0:
                logits_t = logits_t / temperature

            # Draft forward (same prefix)
            out_d = draft_model(input_ids=ids.to(device_d))
            logits_d = _sanitize_logits(out_d.logits[:, -1, :].float().to(logits_t.device))
            if temperature != 1.0:
                logits_d = logits_d / temperature

            # Full distributions for KL
            log_p_t = F.log_softmax(logits_t, dim=-1)
            log_p_d = F.log_softmax(logits_d, dim=-1)
            p_t = log_p_t.exp()

            # KL(target || draft) = sum p_t * (log_p_t - log_p_d)
            kl = F.kl_div(log_p_d, p_t, reduction="batchmean", log_target=False)
            kl_val = kl.item()
            if math.isfinite(kl_val):
                kl_divs.append(kl_val)

            # Top-1 agreement
            if logits_t.argmax(dim=-1).item() == logits_d.argmax(dim=-1).item():
                top1_matches += 1

            # Sample from target (with filtering) to extend sequence
            filtered_t = top_k_top_p_filter(logits_t.clone(), top_k=top_k, top_p=top_p)
            probs_t = F.softmax(filtered_t, dim=-1)
            next_token = _safe_multinomial(probs_t, num_samples=1)

            # P_d / P_t for chosen token
            p_d = F.softmax(logits_d, dim=-1)
            p_d_tok = p_d[0, next_token.item()].item()
            p_t_tok = p_t[0, next_token.item()].item()
            if p_t_tok > 1e-12:
                prob_ratios.append(min(1.0, p_d_tok / p_t_tok))

            ids = torch.cat([ids, next_token.to(device_t)], dim=1)
            n_positions += 1
        except Exception as e:
            # Broken layer config produced degenerate outputs; record what we have
            print(f"    [WARN] alignment step failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            break

    return {
        "kl_div_mean": sum(kl_divs) / max(1, len(kl_divs)),
        "top1_agreement": top1_matches / max(1, n_positions),
        "mean_prob_ratio": sum(prob_ratios) / max(1, len(prob_ratios)),
    }


@torch.inference_mode()
def measure_acceptance_rate(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    gamma: int = 5,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> float:
    """Run speculative decoding and return average acceptance rate."""
    device_t = _model_input_device(target_model)
    device_d = _model_input_device(draft_model)

    ids = input_ids.to(device_t).clone()
    T = ids.size(1) + max_new_tokens

    accepted = 0
    total_draft = 0

    while ids.size(1) < T:
        try:
            # Draft phase
            curr = ids.to(device_d)
            draft_tokens = []
            draft_probs = []

            for _ in range(min(gamma, T - ids.size(1))):
                out_d = draft_model(input_ids=curr)
                logits_d = _sanitize_logits(out_d.logits[:, -1, :].float())
                if temperature != 1.0:
                    logits_d = logits_d / temperature
                p_d_full = F.softmax(logits_d, dim=-1)
                filtered = top_k_top_p_filter(logits_d.clone(), top_k=top_k, top_p=top_p)
                p_d_samp = F.softmax(filtered, dim=-1)
                tok = _safe_multinomial(p_d_samp, num_samples=1)
                draft_tokens.append(tok.to(device_t))
                draft_probs.append(p_d_full.to(device_t))
                curr = torch.cat([curr, tok], dim=1)

            if not draft_tokens:
                break

            num_draft = len(draft_tokens)
            total_draft += num_draft

            # Target verification
            draft_seq = torch.cat(draft_tokens, dim=1)
            target_input = torch.cat([ids, draft_seq], dim=1)
            out_t = target_model(input_ids=target_input.to(device_t), use_cache=False)
            rel_logits = _sanitize_logits(out_t.logits[:, -(num_draft + 1):, :].float())

            all_accepted = True
            for i in range(num_draft):
                tok_id = draft_tokens[i].item()
                p_d = draft_probs[i]
                logits_i = rel_logits[:, i, :]
                if temperature != 1.0:
                    logits_i = logits_i / temperature
                p_t = F.softmax(logits_i, dim=-1)

                p_d_tok = max(p_d[0, tok_id].item(), 1e-12)
                p_t_tok = p_t[0, tok_id].item()

                r = torch.rand(1).item()
                if r < min(1.0, p_t_tok / p_d_tok):
                    ids = torch.cat([ids, draft_tokens[i]], dim=1)
                    accepted += 1
                else:
                    all_accepted = False
                    residual = torch.clamp(p_t - p_d, min=0.0)
                    rsum = residual.sum(dim=-1, keepdim=True)
                    if rsum.item() > 0:
                        residual = residual / rsum
                        new_tok = _safe_multinomial(residual, num_samples=1)
                    else:
                        filt = top_k_top_p_filter(logits_i.clone(), top_k=top_k, top_p=top_p)
                        new_tok = _safe_multinomial(F.softmax(filt, dim=-1), num_samples=1)
                    ids = torch.cat([ids, new_tok.to(device_t)], dim=1)
                    break

            if all_accepted:
                bonus_logits = _sanitize_logits(rel_logits[:, num_draft, :])
                if temperature != 1.0:
                    bonus_logits = bonus_logits / temperature
                filt = top_k_top_p_filter(bonus_logits.clone(), top_k=top_k, top_p=top_p)
                bonus = _safe_multinomial(F.softmax(filt, dim=-1), num_samples=1)
                ids = torch.cat([ids, bonus.to(device_t)], dim=1)

        except Exception as e:
            print(f"    [WARN] acceptance step failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            break

    return accepted / max(1, total_draft)


# ═══════════════════════════════════════════════════════════════════════
# Model-specific setup
# ═══════════════════════════════════════════════════════════════════════


def _repair_progen2_buffers(model):
    """Re-initialise non-persistent buffers that get corrupted by
    ``from_pretrained`` with transformers ≥ 5.x + meta-tensor loading.

    Affected buffers (registered with ``persistent=False``):
      * ``attn.bias``        – causal mask (should be lower-triangular bool)
      * ``attn.masked_bias`` – fill value for masked positions (should be −1e9)
      * ``attn.scale_attn``  – √head_dim  (should be 16.0 for head_dim=256)
    """
    import math
    cfg = model.config
    max_pos = cfg.n_positions
    head_dim = cfg.embed_dim // cfg.n_head

    correct_bias = torch.tril(
        torch.ones((max_pos, max_pos), dtype=torch.bool)
    ).view(1, 1, max_pos, max_pos)
    correct_masked_bias = torch.tensor(-1e9)
    correct_scale = torch.tensor(math.sqrt(head_dim), dtype=torch.float32)

    for block in model.transformer.h:
        dev = next(block.parameters()).device
        block.attn.bias = correct_bias.to(dev)
        block.attn.masked_bias = correct_masked_bias.to(dev)
        block.attn.scale_attn = correct_scale.to(dev)


def run_protgpt2(device: str, dtype: torch.dtype, output_dir: str, num_samples: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "nferruz/ProtGPT2"
    print(f"\n{'='*60}")
    print(f"ProtGPT2 layer-selection experiment")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModelForCausalLM.from_pretrained(model_name)
    full_model.to(device=device, dtype=dtype).eval()
    n_layers = len(full_model.transformer.h)
    print(f"Target model: {model_name} ({n_layers} layers)")

    # Same prompts as in protgpt2_speed_sweep.json
    prompts = [
        "<|endoftext|>M",
        "<|endoftext|>MK",
        "<|endoftext|>MKWVTFISLLLLFSSAYSRGVFRR",
        "<|endoftext|>MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        "<|endoftext|>MALWMRLLPLLALLALWGPDPAAA",
        "<|endoftext|>MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDL",
        "<|endoftext|>MGSSHHHHHHSSGLVPRGSHM",
        "<|endoftext|>MSDTLQPPPVSVRPGS",
    ]

    sampling_kwargs = dict(temperature=1.0, top_k=950, top_p=0.0)
    draft_layer_counts = [2, 3, 4, 5, 6]
    strategies = ["first", "last", "mixed"]
    max_new_tokens = 128
    gamma = 5

    return _run_experiment(
        full_model=full_model,
        tokenizer=tokenizer,
        truncated_cls=LayerSubsetModel,
        n_layers=n_layers,
        prompts=prompts,
        draft_layer_counts=draft_layer_counts,
        strategies=strategies,
        sampling_kwargs=sampling_kwargs,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        num_samples=num_samples,
        device=device,
        dtype=dtype,
        output_path=os.path.join(output_dir, "protgpt2_layer_selection.csv"),
        model_name="ProtGPT2",
    )


def run_progen2(device: str, dtype: torch.dtype, output_dir: str, num_samples: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_name = "hugohrban/progen2-xlarge"
    print(f"\n{'='*60}")
    print(f"ProGen2-xlarge layer-selection experiment")
    print(f"{'='*60}")

    # ── Compatibility patches for transformers 5.x + ProGen2 remote code ──
    # Must be applied BEFORE any from_pretrained calls that load remote code.
    import transformers.modeling_utils as _mu

    # 1) all_tied_weights_keys: removed/renamed in transformers 5.x
    if not hasattr(torch.nn.Module, "all_tied_weights_keys"):
        torch.nn.Module.all_tied_weights_keys = {}

    # 2) get_head_mask: removed from PreTrainedModel in transformers 5.x
    #    but ProGen2's remote code calls self.get_head_mask() in forward().
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("head_mask pruning not supported")
        return [None] * num_hidden_layers
    torch.nn.Module.get_head_mask = _get_head_mask
    _mu.PreTrainedModel.get_head_mask = _get_head_mask

    # ProGen2 tokenizer (loaded AFTER patches)
    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)

    full_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    # Verify patch took effect
    assert hasattr(full_model.transformer, "get_head_mask"), \
        "get_head_mask patch failed — ProGenModel still missing the method"

    full_model.to(device=device).eval()
    _repair_progen2_buffers(full_model)

    # Find number of layers
    stack = LayerSubsetModel._find_stack(full_model)
    n_layers = len(stack.h)
    print(f"Target model: {target_name} ({n_layers} layers)")

    # Same prompts as in progen2_speed_sweep.json
    prompts = [
        "1M",
        "1MK",
        "1MKWVTFISLLLLFSSAYSRGVFRR",
        "1MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        "1MALWMRLLPLLALLALWGPDPAAA",
        "1MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDL",
        "1MGSSHHHHHHSSGLVPRGSHM",
        "1MSDTLQPPPVSVRPGS",
    ]

    sampling_kwargs = dict(temperature=1.0, top_k=0, top_p=0.0)
    draft_layer_counts = [2, 3, 4, 5, 6]
    strategies = ["first", "last", "mixed"]
    max_new_tokens = 128
    gamma = 5

    tok_wrap = tokenizer

    return _run_experiment(
        full_model=full_model,
        tokenizer=tok_wrap,
        truncated_cls=LayerSubsetModel,
        n_layers=n_layers,
        prompts=prompts,
        draft_layer_counts=draft_layer_counts,
        strategies=strategies,
        sampling_kwargs=sampling_kwargs,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        num_samples=num_samples,
        device=device,
        dtype=dtype,
        output_path=os.path.join(output_dir, "progen2_layer_selection.csv"),
        model_name="ProGen2-xlarge",
    )


# ═══════════════════════════════════════════════════════════════════════
# Generic experiment runner
# ═══════════════════════════════════════════════════════════════════════

def _run_experiment(
    full_model,
    tokenizer,
    truncated_cls,
    n_layers: int,
    prompts: List[str],
    draft_layer_counts: List[int],
    strategies: List[str],
    sampling_kwargs: Dict[str, Any],
    max_new_tokens: int,
    gamma: int,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    output_path: str,
    model_name: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns = [
        "model", "n_draft_layers", "strategy", "layer_indices",
        "prompt_idx", "prompt_text", "sample_idx",
        "kl_div_mean", "top1_agreement", "mean_prob_ratio",
        "acceptance_rate",
    ]

    rows = []
    total_configs = len(draft_layer_counts) * len(strategies)
    config_i = 0

    for n_draft in draft_layer_counts:
        for strategy in strategies:
            config_i += 1
            indices = build_layer_indices(n_layers, n_draft, strategy)
            print(f"\n[{config_i}/{total_configs}] {model_name}: "
                  f"{n_draft} layers, strategy={strategy}, indices={indices}")

            # Build truncated draft
            draft_model = truncated_cls(full_model, indices)
            draft_model.to(device=device, dtype=dtype).eval()

            # Sanity check: verify layer swap actually changes output
            if config_i == 1:
                _test_ids = tokenizer.encode(prompts[0], return_tensors="pt").to(device)
                with torch.no_grad():
                    _lt = full_model(input_ids=_test_ids, use_cache=False).logits[:, -1, :]
                    _ld = draft_model(input_ids=_test_ids).logits[:, -1, :]
                assert not torch.isnan(_lt).any(), "BUG: target model outputs NaN — check dtype loading"
                assert not torch.isnan(_ld).any(), "BUG: draft model outputs NaN — check dtype loading"
                _diff = (_lt.float() - _ld.float()).abs().max().item()
                print(f"  [sanity] logit diff target vs draft: {_diff:.1f}")
                assert _diff > 0.1, f"BUG: draft ({len(indices)} layers) == target! diff={_diff}"
                del _test_ids, _lt, _ld

            for pi, prompt_text in enumerate(prompts):
                for si in range(num_samples):
                    # Encode prompt
                    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

                    # Truncate to 64 tokens max
                    if input_ids.size(1) > 64:
                        input_ids = input_ids[:, -64:]

                    torch.manual_seed(42 + si)

                    # Measure alignment
                    align = measure_alignment(
                        full_model, draft_model, input_ids,
                        max_new_tokens=max_new_tokens,
                        **sampling_kwargs,
                    )

                    torch.manual_seed(42 + si)

                    # Measure acceptance rate
                    acc_rate = measure_acceptance_rate(
                        full_model, draft_model, input_ids,
                        max_new_tokens=max_new_tokens, gamma=gamma,
                        **sampling_kwargs,
                    )

                    row = {
                        "model": model_name,
                        "n_draft_layers": n_draft,
                        "strategy": strategy,
                        "layer_indices": str(indices),
                        "prompt_idx": pi,
                        "prompt_text": prompt_text[:40],
                        "sample_idx": si,
                        "kl_div_mean": f"{align['kl_div_mean']:.6f}",
                        "top1_agreement": f"{align['top1_agreement']:.4f}",
                        "mean_prob_ratio": f"{align['mean_prob_ratio']:.4f}",
                        "acceptance_rate": f"{acc_rate:.4f}",
                    }
                    rows.append(row)

                    print(f"  prompt {pi}, sample {si}: "
                          f"KL={align['kl_div_mean']:.4f}, "
                          f"top1={align['top1_agreement']:.3f}, "
                          f"acc_rate={acc_rate:.3f}")

            # Free draft model memory
            del draft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer-selection ablation: first vs last vs mixed layers"
    )
    parser.add_argument("--model", choices=["protgpt2", "progen2", "both"],
                        default="both", help="Which model to test")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda / cpu)")
    parser.add_argument("--dtype", default="float16",
                        help="Torch dtype (float16 / bfloat16 / float32)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples per (prompt, config) pair")
    parser.add_argument("--output_dir", default="results/layer_selection",
                        help="Output directory for CSVs")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    dtype = getattr(torch, args.dtype) if device != "cpu" else torch.float32

    if args.model in ("protgpt2", "both"):
        run_protgpt2(device, dtype, args.output_dir, args.num_samples)

    if args.model in ("progen2", "both"):
        run_progen2(device, dtype, args.output_dir, args.num_samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
