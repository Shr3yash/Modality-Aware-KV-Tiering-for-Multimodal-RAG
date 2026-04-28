from __future__ import annotations

from typing import Dict, Optional

import torch

from .kv_block import KVBlock


class GPUCache:
    """In-memory GPU tier for KV blocks.

    This class does not directly manipulate vLLM internals; instead, it manages
    KVBlock tensors on a configured accelerator device with a fixed capacity in
    number of blocks.
    """

    def __init__(self, max_blocks: int, device: str = "cuda"):
        self.max_blocks = max_blocks
        self.device = device
        self._blocks: Dict[str, KVBlock] = {}

    @property
    def size(self) -> int:
        return len(self._blocks)

    def has_capacity(self) -> bool:
        return self.size < self.max_blocks

    def get(self, block_id: str) -> Optional[KVBlock]:
        return self._blocks.get(block_id)

    def add(self, block: KVBlock) -> None:
        if block.k_cache is not None:
            block.k_cache = block.k_cache.to(self.device, non_blocking=True)
        if block.v_cache is not None:
            block.v_cache = block.v_cache.to(self.device, non_blocking=True)
        block.tier = "gpu"
        self._blocks[block.block_id] = block

    def pop(self, block_id: str) -> Optional[KVBlock]:
        return self._blocks.pop(block_id, None)

