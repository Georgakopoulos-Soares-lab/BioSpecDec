from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any, Dict, Hashable, Optional, Tuple


@dataclass
class CachedModels:
    target_model: Any
    draft_model: Any
    tokenizer: Any
    meta: Dict[str, Any]


class ModelCache:
    """Simple in-process cache to reuse loaded models across sweep runs."""

    def __init__(self):
        self._cache: Dict[Hashable, CachedModels] = {}

    def get(self, key: Hashable) -> Optional[CachedModels]:
        return self._cache.get(key)

    def evict_family(self, family: str, keep_key: Optional[Hashable] = None) -> None:
        to_delete = [
            k
            for k in list(self._cache.keys())
            if isinstance(k, tuple)
            and len(k) > 0
            and k[0] == family
            and (keep_key is None or k != keep_key)
        ]
        for k in to_delete:
            del self._cache[k]
        if to_delete:
            self._gc_cuda()

    def set(self, key: Hashable, value: CachedModels) -> None:
        # ProGen2 models can be extremely large; keeping multiple variants resident on GPU
        # (e.g., varying draft mode/layers) can quickly OOM.
        if isinstance(key, tuple) and len(key) > 0 and key[0] == "progen2":
            self.evict_family("progen2", keep_key=key)

        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
        self._gc_cuda()

    @staticmethod
    def _gc_cuda() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
