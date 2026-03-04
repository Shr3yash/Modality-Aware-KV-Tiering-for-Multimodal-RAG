from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Dict, Optional

from .kv_block import Modality


class EvictionPolicy(ABC):
    """Abstract eviction policy for KV blocks."""

    @abstractmethod
    def on_access(self, block_id: str) -> None:
        ...

    @abstractmethod
    def evict(self, tier: str, modality_hint: Optional[Modality] = None) -> Optional[str]:
        ...

    @abstractmethod
    def add(self, block_id: str, modality: Modality, tier: str) -> None:
        ...

    @abstractmethod
    def remove(self, block_id: str) -> None:
        ...


class LRUEvictionPolicy(EvictionPolicy):
    """Standard modality-agnostic LRU across all tiers."""

    def __init__(self) -> None:
        self._order: "OrderedDict[str, None]" = OrderedDict()

    def on_access(self, block_id: str) -> None:
        if block_id in self._order:
            self._order.move_to_end(block_id)

    def evict(self, tier: str, modality_hint: Optional[Modality] = None) -> Optional[str]:
        if not self._order:
            return None
        block_id, _ = self._order.popitem(last=False)
        return block_id

    def add(self, block_id: str, modality: Modality, tier: str) -> None:
        if block_id in self._order:
            self._order.move_to_end(block_id)
        else:
            self._order[block_id] = None

    def remove(self, block_id: str) -> None:
        self._order.pop(block_id, None)


class ModalityAwareLRUEvictionPolicy(EvictionPolicy):
    """LRU policy that prefers to keep text blocks on GPU HBM."""

    def __init__(self, text_gpu_pin_ratio: float, gpu_blocks: int) -> None:
        self.text_gpu_pin_ratio = text_gpu_pin_ratio
        self.gpu_blocks = gpu_blocks
        self._gpu_text: "OrderedDict[str, None]" = OrderedDict()
        self._gpu_visual: "OrderedDict[str, None]" = OrderedDict()
        self._cpu_all: "OrderedDict[str, None]" = OrderedDict()
        self._ssd_all: "OrderedDict[str, None]" = OrderedDict()
        self._modality: Dict[str, Modality] = {}
        self._tier: Dict[str, str] = {}

    def _gpu_text_capacity(self) -> int:
        return int(self.gpu_blocks * self.text_gpu_pin_ratio)

    def _gpu_visual_capacity(self) -> int:
        return self.gpu_blocks - self._gpu_text_capacity()

    def on_access(self, block_id: str) -> None:
        tier = self._tier.get(block_id)
        if tier == "gpu":
            if block_id in self._gpu_text:
                self._gpu_text.move_to_end(block_id)
            elif block_id in self._gpu_visual:
                self._gpu_visual.move_to_end(block_id)
        elif tier == "cpu":
            if block_id in self._cpu_all:
                self._cpu_all.move_to_end(block_id)
        elif tier == "ssd":
            if block_id in self._ssd_all:
                self._ssd_all.move_to_end(block_id)

    def evict(self, tier: str, modality_hint: Optional[Modality] = None) -> Optional[str]:
        if tier == "gpu":
            # Prefer evicting visual blocks first.
            gpu_text_cap = self._gpu_text_capacity()
            gpu_vis_cap = self._gpu_visual_capacity()
            if self._gpu_visual and len(self._gpu_visual) >= 1:
                block_id, _ = self._gpu_visual.popitem(last=False)
            elif len(self._gpu_text) > gpu_text_cap and self._gpu_text:
                block_id, _ = self._gpu_text.popitem(last=False)
            else:
                # Fallback to any available GPU block.
                if self._gpu_visual:
                    block_id, _ = self._gpu_visual.popitem(last=False)
                elif self._gpu_text:
                    block_id, _ = self._gpu_text.popitem(last=False)
                else:
                    return None
        elif tier == "cpu":
            if not self._cpu_all:
                return None
            block_id, _ = self._cpu_all.popitem(last=False)
        elif tier == "ssd":
            if not self._ssd_all:
                return None
            block_id, _ = self._ssd_all.popitem(last=False)
        else:
            return None

        self._tier.pop(block_id, None)
        self._modality.pop(block_id, None)
        return block_id

    def add(self, block_id: str, modality: Modality, tier: str) -> None:
        self._modality[block_id] = modality
        self._tier[block_id] = tier
        if tier == "gpu":
            if modality == Modality.VISUAL:
                self._gpu_visual[block_id] = None
                self._gpu_visual.move_to_end(block_id)
            else:
                self._gpu_text[block_id] = None
                self._gpu_text.move_to_end(block_id)
        elif tier == "cpu":
            self._cpu_all[block_id] = None
            self._cpu_all.move_to_end(block_id)
        elif tier == "ssd":
            self._ssd_all[block_id] = None
            self._ssd_all.move_to_end(block_id)

    def remove(self, block_id: str) -> None:
        self._gpu_text.pop(block_id, None)
        self._gpu_visual.pop(block_id, None)
        self._cpu_all.pop(block_id, None)
        self._ssd_all.pop(block_id, None)
        self._modality.pop(block_id, None)
        self._tier.pop(block_id, None)

