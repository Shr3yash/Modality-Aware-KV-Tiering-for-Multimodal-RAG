from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from src.utils.metrics import (
    KV_EVICTIONS,
    KV_HITS,
    KV_MISSES,
    KV_PROMOTION_LATENCY,
    KV_TIER_USAGE,
)
from .cpu_cache import CPUCache
from .eviction import EvictionPolicy, LRUEvictionPolicy, ModalityAwareLRUEvictionPolicy
from .gpu_cache import GPUCache
from .kv_block import KVBlock, Modality
from .ssd_cache import SSDCache


class CacheManager:
    """Orchestrates movement of KV blocks across GPU, CPU, and SSD tiers."""

    def __init__(self, cache_config) -> None:
        self.gpu_cache = GPUCache(cache_config.gpu_blocks)
        self.cpu_cache = CPUCache(cache_config.cpu_blocks)
        self.ssd_cache = SSDCache(cache_config.ssd_path, cache_config.ssd_capacity_gb)

        if cache_config.eviction_policy == "modality_aware_lru":
            self.eviction: EvictionPolicy = ModalityAwareLRUEvictionPolicy(
                cache_config.text_gpu_pin_ratio, cache_config.gpu_blocks
            )
        else:
            self.eviction = LRUEvictionPolicy()

        KV_TIER_USAGE.labels(tier="gpu").set(0)
        KV_TIER_USAGE.labels(tier="cpu").set(0)
        KV_TIER_USAGE.labels(tier="ssd").set(0)

    def get_kv_block(self, block_id: str) -> Optional[KVBlock]:
        """Lookup a KV block by id, promoting from lower tiers as needed."""
        # GPU tier
        block = self.gpu_cache.get(block_id)
        if block is not None:
            block.touch()
            self.eviction.on_access(block_id)
            KV_HITS.labels(tier="gpu", modality=block.modality.value).inc()
            return block

        # CPU tier
        block = self.cpu_cache.get(block_id)
        if block is not None:
            KV_HITS.labels(tier="cpu", modality=block.modality.value).inc()
            self._promote_cpu_to_gpu(block)
            return block

        # SSD tier
        if self.ssd_cache.exists(block_id):
            k_cache, v_cache, metadata = self.ssd_cache.load(block_id)  # type: ignore[assignment]
            modality = metadata.get("modality", "text")
            modality_enum = Modality(modality) if isinstance(modality, str) else modality
            block = KVBlock(
                block_id=block_id,
                modality=modality_enum,
                token_ids=tuple(metadata.get("token_ids", ())),
                k_cache=k_cache,
                v_cache=v_cache,
                num_tokens=metadata.get("num_tokens", 0),
                layer_range=tuple(metadata.get("layer_range", (0, 0))),
                tier="ssd",
            )
            KV_HITS.labels(tier="ssd", modality=block.modality.value).inc()
            self._promote_ssd_to_cpu(block)
            self._promote_cpu_to_gpu(block)
            return block

        KV_MISSES.labels(tier="all", modality="unknown").inc()
        return None

    def put_kv_block(self, block: KVBlock) -> None:
        """Insert a new KV block, placing it in the appropriate tier."""
        target_tier = "gpu" if block.modality != Modality.VISUAL else "cpu"

        if target_tier == "gpu" and not self.gpu_cache.has_capacity():
            self._evict_from_tier("gpu", block.modality)
        if target_tier == "cpu" and not self.cpu_cache.has_capacity():
            self._evict_from_tier("cpu", block.modality)

        if target_tier == "gpu":
            self.gpu_cache.add(block)
            KV_TIER_USAGE.labels(tier="gpu").set(self.gpu_cache.size)
        else:
            self.cpu_cache.add(block)
            KV_TIER_USAGE.labels(tier="cpu").set(self.cpu_cache.size)

        self.eviction.add(block.block_id, block.modality, target_tier)

    def _evict_from_tier(self, tier: str, modality: Modality) -> None:
        block_id = self.eviction.evict(tier, modality_hint=modality)
        if block_id is None:
            return

        if tier == "gpu":
            block = self.gpu_cache.pop(block_id)
            KV_TIER_USAGE.labels(tier="gpu").set(self.gpu_cache.size)
        elif tier == "cpu":
            block = self.cpu_cache.pop(block_id)
            KV_TIER_USAGE.labels(tier="cpu").set(self.cpu_cache.size)
        elif tier == "ssd":
            self.ssd_cache.delete(block_id)
            KV_TIER_USAGE.labels(tier="ssd").set(
                max(KV_TIER_USAGE.labels(tier="ssd")._value.get(), 0)  # type: ignore[attr-defined]
            )
            KV_EVICTIONS.labels(tier="ssd", modality="unknown").inc()
            return
        else:
            return

        if block is None:
            return

        # Demote GPU -> CPU / CPU -> SSD
        if tier == "gpu":
            self._gpu_to_cpu(block)
            if not self.cpu_cache.has_capacity():
                self._evict_from_tier("cpu", block.modality)
            self.cpu_cache.add(block)
            KV_TIER_USAGE.labels(tier="cpu").set(self.cpu_cache.size)
        elif tier == "cpu":
            self._cpu_to_ssd(block)

        KV_EVICTIONS.labels(tier=tier, modality=block.modality.value).inc()

    def _gpu_to_cpu(self, block: KVBlock) -> None:
        with KV_PROMOTION_LATENCY.labels("gpu", "cpu").time():
            if block.k_cache is not None:
                block.k_cache = block.k_cache.to(
                    "cpu", non_blocking=True, copy=True
                ).pin_memory()
            if block.v_cache is not None:
                block.v_cache = block.v_cache.to(
                    "cpu", non_blocking=True, copy=True
                ).pin_memory()
            block.tier = "cpu"

    def _cpu_to_gpu(self, block: KVBlock) -> None:
        with KV_PROMOTION_LATENCY.labels("cpu", "gpu").time():
            if block.k_cache is not None:
                block.k_cache = block.k_cache.to("cuda", non_blocking=True)
            if block.v_cache is not None:
                block.v_cache = block.v_cache.to("cuda", non_blocking=True)
            block.tier = "gpu"

    def _cpu_to_ssd(self, block: KVBlock) -> None:
        with KV_PROMOTION_LATENCY.labels("cpu", "ssd").time():
            metadata = {
                "modality": block.modality.value,
                "token_ids": list(block.token_ids),
                "num_tokens": block.num_tokens,
                "layer_range": list(block.layer_range),
            }
            if block.k_cache is None or block.v_cache is None:
                return
            ok = self.ssd_cache.store(block.block_id, block.k_cache, block.v_cache, metadata)
            if ok:
                block.tier = "ssd"
                KV_TIER_USAGE.labels(tier="ssd").set(
                    self.ssd_cache.usage_bytes()
                )

    def _ssd_to_cpu(self, block: KVBlock) -> None:
        # This method is implicit in get_kv_block via SSDCache.load; keep for symmetry.
        block.tier = "cpu"

    def _promote_cpu_to_gpu(self, block: KVBlock) -> None:
        if not self.gpu_cache.has_capacity():
            self._evict_from_tier("gpu", block.modality)
        start = time.perf_counter()
        self._cpu_to_gpu(block)
        self.gpu_cache.add(block)
        self.cpu_cache.pop(block.block_id)
        KV_TIER_USAGE.labels(tier="gpu").set(self.gpu_cache.size)
        KV_TIER_USAGE.labels(tier="cpu").set(self.cpu_cache.size)
        elapsed = time.perf_counter() - start
        KV_PROMOTION_LATENCY.labels("cpu", "gpu").observe(elapsed)

    def _promote_ssd_to_cpu(self, block: KVBlock) -> None:
        if not self.cpu_cache.has_capacity():
            self._evict_from_tier("cpu", block.modality)
        start = time.perf_counter()
        self.cpu_cache.add(block)
        KV_TIER_USAGE.labels(tier="cpu").set(self.cpu_cache.size)
        elapsed = time.perf_counter() - start
        KV_PROMOTION_LATENCY.labels("ssd", "cpu").observe(elapsed)

