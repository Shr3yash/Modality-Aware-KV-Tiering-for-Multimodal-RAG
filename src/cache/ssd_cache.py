from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from safetensors.torch import load_file, save_file

from .kv_block import KVBlock


class SSDCache:
    """SSD-backed KV cache using safetensors for fast, safe tensor I/O."""

    def __init__(self, ssd_path: str, capacity_gb: float):
        self.base_path = Path(ssd_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = int(capacity_gb * (1024**3))

    def _block_path(self, block_id: str) -> Path:
        return self.base_path / f"{block_id}.safetensors"

    def _meta_path(self, block_id: str) -> Path:
        return self.base_path / f"{block_id}.meta"

    def store(
        self,
        block_id: str,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> bool:
        """Store KV tensors and metadata on disk. Returns False if capacity exceeded."""
        if self.usage_bytes() >= self.capacity_bytes:
            return False

        tensors = {"k": k_cache.cpu(), "v": v_cache.cpu()}
        save_file(tensors, str(self._block_path(block_id)))
        self._meta_path(block_id).write_text(repr(metadata), encoding="utf-8")
        return True

    def load(
        self, block_id: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """Load KV tensors and metadata from disk."""
        block_path = self._block_path(block_id)
        meta_path = self._meta_path(block_id)
        if not block_path.exists() or not meta_path.exists():
            return None
        tensors = load_file(str(block_path))
        try:
            metadata = eval(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        k_cache = tensors["k"]
        v_cache = tensors["v"]
        return k_cache, v_cache, metadata

    def delete(self, block_id: str) -> bool:
        """Remove KV tensors and metadata from disk."""
        removed = False
        for p in (self._block_path(block_id), self._meta_path(block_id)):
            try:
                if p.exists():
                    p.unlink()
                    removed = True
            except OSError:
                continue
        return removed

    def exists(self, block_id: str) -> bool:
        """Return True if the block is present on disk."""
        return self._block_path(block_id).exists()

    def usage_bytes(self) -> int:
        """Return current disk usage in bytes for all cached blocks."""
        total = 0
        if not self.base_path.exists():
            return 0
        for entry in self.base_path.iterdir():
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
        return total

