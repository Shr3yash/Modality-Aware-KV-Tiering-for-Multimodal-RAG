from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import time

import torch


class Modality(Enum):
    """Modality type associated with a KV block."""

    TEXT = "text"
    VISUAL = "visual"
    MIXED = "mixed"


@dataclass
class KVBlock:
    """Represents a block of KV cache along with metadata for tiering."""

    block_id: str
    modality: Modality
    token_ids: Tuple[int, ...]
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    num_tokens: int
    layer_range: Tuple[int, int]
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    tier: str = "gpu"  # "gpu", "cpu", "ssd"
    pinned: bool = False
    recompute_mask: Optional[torch.Tensor] = None

    def touch(self) -> None:
        """Update access metadata when the block is used."""
        self.last_access_time = time.time()
        self.access_count += 1

