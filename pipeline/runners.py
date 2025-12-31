from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from .model_cache import CachedModels, ModelCache


ModelFamily = Literal["dnagpt", "protgpt2", "progen2"]
Method = Literal["target_baseline", "draft_baseline", "specdec"]
Progen2DraftMode = Literal["pretrained", "truncated"]


@dataclass
class GenerationRequest:
    model_family: ModelFamily
    method: Method

    prompt_text: str
    max_new_tokens: int
    num_samples: int

    # sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0

    # specdec
    gamma: int = 5
    accept_mode: str = "prob"

    # model identifiers
    target_model_name: str = ""
    draft_model_name: str = ""

    # Optional weight paths (required for DNAGPT; ignored for HF models)
    target_weight: Optional[str] = None
    draft_weight: Optional[str] = None

    # truncation / context
    max_prompt_tokens: int = 0
    target_context_len: Optional[int] = None
    draft_context_len: Optional[int] = None

    # ProtGPT2 draft
    draft_layers: int = 8
    draft_layer_indices: Optional[List[int]] = None

    # ProGen2 tokenizer
    tokenizer_name: Optional[str] = None

    # ProGen2 draft options
    progen2_draft_mode: Progen2DraftMode = "pretrained"

    # runtime
    device: Optional[str] = None
    # Optional per-model device placement (used for "one GPU per model" workflows).
    # If unset, falls back to `device`.
    target_device: Optional[str] = None
    draft_device: Optional[str] = None
    # Optional HuggingFace sharding/offload setting (e.g. "auto").
    # Only used for ProGen2; when set, the model can be split across all visible GPUs.
    device_map: Optional[str] = None
    dtype: str = "float16"
    seed: int = 42


def _model_input_device(model: torch.nn.Module) -> torch.device:
    """Return the device where inputs should live (handles HF `device_map` models)."""
    try:
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb):
            emb = get_emb()
            w = getattr(emb, "weight", None)
            if w is not None:
                return w.device
    except Exception:
        pass
    # ProGen2 remote models may not implement get_input_embeddings(); fall back to known attribute.
    try:
        tr = getattr(model, "transformer", None)
        wte = getattr(tr, "wte", None)
        w = getattr(wte, "weight", None)
        if w is not None:
            return w.device
    except Exception:
        pass
    return next(model.parameters()).device


def _resolve_device_dtype(device: Optional[str], dtype_str: str) -> Tuple[str, torch.dtype]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Accept both "cuda" and explicit device strings like "cuda:0".
    if str(device).startswith("cuda"):
        dtype = getattr(torch, dtype_str)
    else:
        dtype = torch.float32
    return device, dtype


def _maybe_seed(seed: int) -> None:
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dnagpt(cache: ModelCache, req: GenerationRequest) -> CachedModels:
    if not req.target_model_name or not req.draft_model_name:
        raise ValueError("DNAGPT requires target_model_name and draft_model_name")
    if not req.target_weight or not req.draft_weight:
        raise ValueError("DNAGPT generation requires target_weight and draft_weight")

    import os
    import sys

    device, dtype = _resolve_device_dtype(req.device, req.dtype)
    target_device = req.target_device or req.device or device
    draft_device = req.draft_device or req.device or device

    # Make DNAGPT imports work regardless of CWD.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dnagpt_dir = os.path.join(repo_root, "DNAGPT")
    if os.path.isdir(dnagpt_dir) and dnagpt_dir not in sys.path:
        sys.path.append(dnagpt_dir)

    # Reuse the DNAGPT helpers from scoring_specdec_beam_search.py to keep behavior consistent
    # across sweeps, scoring, and one-off generation.
    from scoring_specdec_beam_search import get_model as dnagpt_get_model
    from scoring_specdec_beam_search import load_model as dnagpt_load_model

    key = (
        "dnagpt",
        req.target_model_name,
        str(req.target_weight),
        req.draft_model_name,
        str(req.draft_weight),
        str(target_device),
        str(draft_device),
        str(dtype),
    )
    hit = cache.get(key)
    if hit is not None:
        return hit

    # Tokenizer choice: use tokenizer for the target model.
    _target_model_unloaded, tokenizer = dnagpt_get_model(req.target_model_name)

    target_model, _ = dnagpt_get_model(req.target_model_name)
    target_model = dnagpt_load_model(target_model, str(req.target_weight), device=str(target_device), dtype=dtype)

    draft_model, _ = dnagpt_get_model(req.draft_model_name)
    draft_model = dnagpt_load_model(draft_model, str(req.draft_weight), device=str(draft_device), dtype=dtype)

    cm = CachedModels(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        meta={
            "device": str(device),
            "target_device": str(target_device),
            "draft_device": str(draft_device),
            "dtype": str(dtype),
            "target_weight": str(req.target_weight),
            "draft_weight": str(req.draft_weight),
        },
    )
    cache.set(key, cm)
    return cm


def _load_protgpt2(cache: ModelCache, req: GenerationRequest) -> CachedModels:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # draft topology depends on indices/layers
    draft_idx_tuple = tuple(req.draft_layer_indices or [])
    device, dtype = _resolve_device_dtype(req.device, req.dtype)

    key = (
        "protgpt2",
        req.target_model_name,
        req.draft_layers,
        draft_idx_tuple,
        device,
        str(dtype),
    )
    hit = cache.get(key)
    if hit is not None:
        return hit

    # Reuse your TruncatedProtGPT2 implementation.
    from specdec_protein import TruncatedProtGPT2

    tokenizer = AutoTokenizer.from_pretrained(req.target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModelForCausalLM.from_pretrained(req.target_model_name)
    full_model.to(device=device, dtype=dtype)
    full_model.eval()

    if not hasattr(full_model, "transformer") or not hasattr(full_model.transformer, "h"):
        raise ValueError(
            "Target model does not look GPT2-style (no .transformer.h). "
            "Cannot build truncated draft."
        )

    n_layers_full = len(full_model.transformer.h)

    if req.draft_layer_indices:
        draft_indices = list(req.draft_layer_indices)
    else:
        if req.draft_layers <= 0 or req.draft_layers > n_layers_full:
            raise ValueError(f"draft_layers must be in [1, {n_layers_full}]")
        draft_indices = list(range(req.draft_layers))

    draft_model = TruncatedProtGPT2(full_model, draft_indices)
    draft_model.to(device=device, dtype=dtype)
    draft_model.eval()

    cm = CachedModels(
        target_model=full_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        meta={"device": device, "dtype": str(dtype), "draft_indices": draft_indices},
    )
    cache.set(key, cm)
    return cm


def _load_progen2(cache: ModelCache, req: GenerationRequest) -> CachedModels:
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    # In pretrained draft mode, draft_layers / draft_layer_indices are irrelevant.
    # Avoid redundant cache entries if a sweep accidentally varies them.
    draft_layers_key: Any
    draft_layer_indices_key: Any
    if req.progen2_draft_mode == "pretrained":
        draft_layers_key = None
        draft_layer_indices_key = None
    else:
        draft_layers_key = req.draft_layers
        draft_layer_indices_key = tuple(req.draft_layer_indices or [])

    device, dtype = _resolve_device_dtype(req.device, req.dtype)
    target_device = req.target_device or req.device or device
    draft_device = req.draft_device or req.device or device
    device_map = (req.device_map or "").strip() or None

    if device_map is not None and req.progen2_draft_mode == "truncated":
        raise ValueError(
            "progen2_draft_mode=truncated is not compatible with device_map sharding; "
            "use progen2_draft_mode=pretrained when device_map is set."
        )

    if req.progen2_draft_mode == "truncated" and str(draft_device) != str(target_device):
        # Truncated draft shares modules with target; it cannot live on a different device.
        draft_device = target_device

    key = (
        "progen2",
        req.target_model_name,
        req.draft_model_name,
        req.tokenizer_name or req.target_model_name,
        req.progen2_draft_mode,
        draft_layers_key,
        draft_layer_indices_key,
        str(target_device),
        str(draft_device),
        device_map,
        str(dtype),
    )
    hit = cache.get(key)
    if hit is not None:
        return hit

    # Proactively free any previously cached ProGen2 model before loading a new one.
    # This avoids transient peak memory where both the old and new target models are resident.
    cache.evict_family("progen2", keep_key=key)

    def _validate_cuda_device(dev: str) -> None:
        if not str(dev).startswith("cuda"):
            return
        if not torch.cuda.is_available():
            raise ValueError(f"Requested {dev} but CUDA is not available")
        if str(dev) == "cuda":
            return
        try:
            idx = int(str(dev).split(":", 1)[1])
        except Exception:
            raise ValueError(f"Invalid CUDA device string: {dev}")
        n = int(torch.cuda.device_count())
        if idx < 0 or idx >= n:
            raise ValueError(f"Requested {dev} but only {n} CUDA device(s) visible")

    _validate_cuda_device(str(target_device))
    _validate_cuda_device(str(draft_device))

    tok_name = req.tokenizer_name or req.target_model_name
    tokenizer = Tokenizer.from_pretrained(tok_name)
    tokenizer.no_padding()

    # Load weights in the requested dtype.
    # - If device_map is set (e.g. "auto"), HF/accelerate may shard a single model across devices.
    # - Otherwise, we place the target and draft models on their requested single GPUs.
    target_load_device_map = device_map
    if target_load_device_map is None and str(target_device).startswith("cuda"):
        target_load_device_map = {"": str(target_device)}

    target_model = AutoModelForCausalLM.from_pretrained(
        req.target_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=target_load_device_map,
    )

    # Draft can either be a separate pretrained model, or a truncated view of the target.
    if req.progen2_draft_mode == "pretrained":
        if not req.draft_model_name:
            raise ValueError("progen2_draft_mode=pretrained requires draft_model_name")

        draft_load_device_map = device_map
        if draft_load_device_map is None and str(draft_device).startswith("cuda"):
            draft_load_device_map = {"": str(draft_device)}

        draft_model = AutoModelForCausalLM.from_pretrained(
            req.draft_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=draft_load_device_map,
        )
        draft_meta: Dict[str, Any] = {"draft_mode": "pretrained"}
    elif req.progen2_draft_mode == "truncated":
        from specdec_progen2_truncated import TruncatedProGenDraft, get_progen_stack

        stack = get_progen_stack(target_model)
        n_layers_full = len(stack.h)

        if req.draft_layer_indices:
            layer_indices = list(req.draft_layer_indices)
        else:
            if req.draft_layers <= 0 or req.draft_layers > n_layers_full:
                raise ValueError(f"draft_layers must be in [1, {n_layers_full}]")
            layer_indices = list(range(req.draft_layers))

        draft_model = TruncatedProGenDraft(target_model, layer_indices)
        draft_meta = {"draft_mode": "truncated", "draft_layer_indices": layer_indices}
    else:
        raise ValueError(f"Unknown progen2_draft_mode={req.progen2_draft_mode}")

    # If device_map is used, models are already placed; calling .to(...) would error.
    if device_map is None:
        # For single-device placement, we already loaded directly onto the desired devices.
        # Keep this path as a safety net if a backend ignores device_map placement.
        target_model.to(device=str(target_device), dtype=dtype)
        draft_model.to(device=str(draft_device), dtype=dtype)
    target_model.eval()
    draft_model.eval()

    cm = CachedModels(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        meta={
            "device": str(device),
            "target_device": str(target_device),
            "draft_device": str(draft_device),
            "device_map": device_map,
            "dtype": str(dtype),
            **draft_meta,
        },
    )
    cache.set(key, cm)
    return cm


def load_models(cache: ModelCache, req: GenerationRequest) -> CachedModels:
    if req.model_family == "dnagpt":
        return _load_dnagpt(cache, req)
    if req.model_family == "protgpt2":
        return _load_protgpt2(cache, req)
    if req.model_family == "progen2":
        return _load_progen2(cache, req)
    raise ValueError(f"Unknown model_family={req.model_family}")


def _encode_prompt(req: GenerationRequest, cm: CachedModels) -> torch.Tensor:
    device = _model_input_device(cm.target_model)

    if req.model_family == "dnagpt":
        # DNAGPT tokenizer returns a 1D tensor already.
        prompt_ids = cm.tokenizer.encode(req.prompt_text, device=device)
        input_ids = prompt_ids.unsqueeze(0)
    elif req.model_family == "protgpt2":
        input_ids = cm.tokenizer.encode(req.prompt_text, return_tensors="pt").to(device)
    elif req.model_family == "progen2":
        prompt_ids = cm.tokenizer.encode(req.prompt_text).ids
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        raise ValueError(req.model_family)

    if req.max_prompt_tokens and req.max_prompt_tokens > 0 and input_ids.size(1) > req.max_prompt_tokens:
        input_ids = input_ids[:, -req.max_prompt_tokens:]

    return input_ids


def _decode(cm: CachedModels, req: GenerationRequest, ids: torch.Tensor) -> str:
    if req.model_family == "dnagpt":
        return cm.tokenizer.decode(ids[0].tolist())
    if req.model_family == "protgpt2":
        return cm.tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
    if req.model_family == "progen2":
        return cm.tokenizer.decode(ids[0].tolist())
    raise ValueError(req.model_family)


def run_generation(cache: ModelCache, req: GenerationRequest) -> List[Dict[str, Any]]:
    """Runs num_samples and returns JSON-serializable records."""

    _maybe_seed(req.seed)

    cm = load_models(cache, req)
    input_ids = _encode_prompt(req, cm)

    # Import the algorithm implementations from your existing scripts to keep behavior consistent.
    if req.model_family == "dnagpt":
        from scoring_specdec_beam_search import _timeit as _dnagpt_timeit
        from scoring_specdec_beam_search import generate_baseline as dnagpt_generate_baseline
        from scoring_specdec_beam_search import speculative_sampling as dnagpt_speculative_sampling
        eos_id = None
        use_kv_cache = False
    elif req.model_family == "protgpt2":
        from specdec_protein import generate_baseline, speculative_sampling
        eos_id = cm.tokenizer.eos_token_id
        use_kv_cache = True
    elif req.model_family == "progen2":
        # Use the implementation that supports both pretrained and truncated drafts.
        from specdec_progen2_truncated import generate_baseline, speculative_sampling
        eos_id = None
        use_kv_cache = True
    else:
        raise ValueError(req.model_family)

    device = _model_input_device(cm.target_model)

    # Warm-up (not timed) to reduce first-iter noise.
    warmup_tokens = min(16, req.max_new_tokens)
    if warmup_tokens > 0:
        with torch.inference_mode():
            if req.method in ("target_baseline", "draft_baseline"):
                model = cm.target_model if req.method == "target_baseline" else cm.draft_model
                if req.model_family == "dnagpt":
                    _ids, _ = _dnagpt_timeit(
                        _model_input_device(model),
                        lambda: dnagpt_generate_baseline(
                            model,
                            cm.tokenizer,
                            input_ids,
                            warmup_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            context_len=(
                                req.target_context_len if req.method == "target_baseline" else req.draft_context_len
                            ),
                        ),
                    )
                else:
                    _ids, _ = generate_baseline(
                        model,
                        input_ids,
                        warmup_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        eos_token_id=eos_id,
                        use_kv_cache=use_kv_cache,
                    )
            else:
                if req.model_family == "dnagpt":
                    (_ids, _acc, _pref), _t = _dnagpt_timeit(
                        _model_input_device(cm.target_model),
                        lambda: dnagpt_speculative_sampling(
                            cm.target_model,
                            cm.draft_model,
                            cm.tokenizer,
                            input_ids,
                            warmup_tokens,
                            gamma=req.gamma,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            accept_mode=req.accept_mode,
                            target_context_len=req.target_context_len,
                            draft_context_len=req.draft_context_len,
                            debug=False,
                        ),
                    )
                elif req.model_family == "protgpt2":
                    _ids, _t, _acc = speculative_sampling(
                        cm.target_model,
                        cm.draft_model,
                        input_ids,
                        warmup_tokens,
                        gamma=req.gamma,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        accept_mode=req.accept_mode,
                        target_context_len=req.target_context_len,
                        draft_context_len=req.draft_context_len,
                        eos_token_id=eos_id,
                        debug=False,
                    )
                elif req.model_family == "progen2":
                    _ids, _t, _acc = speculative_sampling(
                        cm.target_model,
                        cm.draft_model,
                        input_ids,
                        warmup_tokens,
                        gamma=req.gamma,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        accept_mode=req.accept_mode,
                        eos_token_id=eos_id,
                        debug=False,
                    )
                else:
                    raise ValueError(req.model_family)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    records: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for sample_idx in range(req.num_samples):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

            if req.method in ("target_baseline", "draft_baseline"):
                model = cm.target_model if req.method == "target_baseline" else cm.draft_model

                if req.model_family == "dnagpt":
                    ids, duration = _dnagpt_timeit(
                        _model_input_device(model),
                        lambda: dnagpt_generate_baseline(
                            model,
                            cm.tokenizer,
                            input_ids,
                            req.max_new_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            context_len=(
                                req.target_context_len if req.method == "target_baseline" else req.draft_context_len
                            ),
                        ),
                    )
                else:
                    ids, duration = generate_baseline(
                        model,
                        input_ids,
                        req.max_new_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        eos_token_id=eos_id,
                        use_kv_cache=use_kv_cache,
                    )

                acceptance_rate = None
                accepted_count = None
                total_draft_tokens = None

            else:
                if req.model_family == "dnagpt":
                    (ids, acceptance_rate, _pref), duration = _dnagpt_timeit(
                        _model_input_device(cm.target_model),
                        lambda: dnagpt_speculative_sampling(
                            cm.target_model,
                            cm.draft_model,
                            cm.tokenizer,
                            input_ids,
                            req.max_new_tokens,
                            gamma=req.gamma,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            accept_mode=req.accept_mode,
                            target_context_len=req.target_context_len,
                            draft_context_len=req.draft_context_len,
                            debug=False,
                        ),
                    )
                elif req.model_family == "protgpt2":
                    ids, duration, acceptance_rate = speculative_sampling(
                        cm.target_model,
                        cm.draft_model,
                        input_ids,
                        req.max_new_tokens,
                        gamma=req.gamma,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        accept_mode=req.accept_mode,
                        target_context_len=req.target_context_len,
                        draft_context_len=req.draft_context_len,
                        eos_token_id=eos_id,
                        debug=False,
                    )
                elif req.model_family == "progen2":
                    ids, duration, acceptance_rate = speculative_sampling(
                        cm.target_model,
                        cm.draft_model,
                        input_ids,
                        req.max_new_tokens,
                        gamma=req.gamma,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        accept_mode=req.accept_mode,
                        eos_token_id=eos_id,
                        debug=False,
                    )
                else:
                    raise ValueError(req.model_family)
                accepted_count = None
                total_draft_tokens = None

            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

            prompt_len = int(input_ids.size(1))
            total_len = int(ids.size(1))
            n_new = total_len - prompt_len

            tps = (n_new / duration) if duration and duration > 0 else 0.0

            decoded = _decode(cm, req, ids)

            rec: Dict[str, Any] = {
                "model_family": req.model_family,
                "method": req.method,
                "sample_idx": sample_idx,
                "prompt_text": req.prompt_text,
                "prompt_len": prompt_len,
                "max_new_tokens": req.max_new_tokens,
                "num_new_tokens": n_new,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "gamma": req.gamma,
                "accept_mode": req.accept_mode,
                "target_model_name": req.target_model_name,
                "draft_model_name": req.draft_model_name,
                "target_weight": req.target_weight,
                "draft_weight": req.draft_weight,
                "draft_layers": req.draft_layers,
                "draft_layer_indices": req.draft_layer_indices,
                "tokenizer_name": req.tokenizer_name,
                "device": str(device),
                "dtype": req.dtype,
                "seed": req.seed,
                "wall_time_s": float(duration),
                "tokens_per_s": float(tps),
                "acceptance_rate": (float(acceptance_rate) if acceptance_rate is not None else None),
                "accepted_count": accepted_count,
                "total_draft_tokens": total_draft_tokens,
                "prompt_ids": input_ids[0].detach().cpu().tolist(),
                "full_ids": ids[0].detach().cpu().tolist(),
                "new_ids": ids[0, prompt_len:].detach().cpu().tolist(),
                "generated_text": decoded,
            }
            records.append(rec)

    return records
