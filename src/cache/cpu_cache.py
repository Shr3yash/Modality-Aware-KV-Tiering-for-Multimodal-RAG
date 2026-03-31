from __future__ import annotations

from typing import Dict, Optional

import torch

from .kv_block import KVBlock


class CPUCache:
    """CPU DRAM tier for KV blocks using pinned memory for faster GPU transfers."""

    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self._blocks: Dict[str, KVBlock] = {}

    @property
    def size(self) -> int:
        return len(self._blocks)

    def has_capacity(self) -> bool:
        return self.size < self.max_blocks

    def get(self, block_id: str) -> Optional[KVBlock]:
        return self._blocks.get(block_id)

    def _to_cpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to("cpu", non_blocking=True, copy=True)
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def add(self, block: KVBlock) -> None:
        if block.k_cache is not None:
            block.k_cache = self._to_cpu_tensor(block.k_cache)
        if block.v_cache is not None:
            block.v_cache = self._to_cpu_tensor(block.v_cache)
        block.tier = "cpu"
        self._blocks[block.block_id] = block

    def pop(self, block_id: str) -> Optional[KVBlock]:
        return self._blocks.pop(block_id, None)
