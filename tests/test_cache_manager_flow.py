from types import SimpleNamespace

import torch

from src.cache.cache_manager import CacheManager
from src.cache.kv_block import KVBlock, Modality
from src.utils.config import _gpu_device


def _dummy_config(tmp_path, **overrides) -> SimpleNamespace:
    config = {
        "gpu_blocks": 2,
        "gpu_device": _gpu_device(),
        "cpu_blocks": 2,
        "ssd_capacity_gb": 1.0,
        "ssd_path": str(tmp_path / "ssd"),
        "block_size": 16,
        "eviction_policy": "modality_aware_lru",
        "text_gpu_pin_ratio": 0.7,
        "recompute_top_k": 32,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _make_block(block_id: str, modality: Modality = Modality.TEXT) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        modality=modality,
        token_ids=tuple(range(16)),
        k_cache=torch.randn(1, 1, 16, 8),
        v_cache=torch.randn(1, 1, 16, 8),
        num_tokens=16,
        layer_range=(0, 0),
    )


def test_cache_manager_miss_returns_none(tmp_path) -> None:
    manager = CacheManager(_dummy_config(tmp_path))

    assert manager.get_kv_block("missing") is None


def test_visual_block_starts_in_cpu_tier(tmp_path) -> None:
    manager = CacheManager(_dummy_config(tmp_path))
    block = _make_block("visual1", modality=Modality.VISUAL)

    manager.put_kv_block(block)

    assert manager.cpu_cache.get("visual1") is not None
    assert manager.gpu_cache.get("visual1") is None


def test_ssd_hit_promotes_block_back_to_gpu(tmp_path) -> None:
    cfg = _dummy_config(tmp_path)
    manager = CacheManager(cfg)
    block = _make_block("ssd1")
    metadata = {
        "modality": block.modality.value,
        "token_ids": list(block.token_ids),
        "num_tokens": block.num_tokens,
        "layer_range": list(block.layer_range),
    }
    assert block.k_cache is not None
    assert block.v_cache is not None
    assert manager.ssd_cache.store(block.block_id, block.k_cache, block.v_cache, metadata)

    got = manager.get_kv_block(block.block_id)

    assert got is not None
    assert got.tier == "gpu"
    assert manager.gpu_cache.get(block.block_id) is not None
    assert manager.cpu_cache.get(block.block_id) is None


def test_gpu_eviction_demotes_old_block_to_cpu(tmp_path) -> None:
    manager = CacheManager(_dummy_config(tmp_path, gpu_blocks=1, cpu_blocks=2))

    manager.put_kv_block(_make_block("b1"))
    manager.put_kv_block(_make_block("b2"))

    assert manager.gpu_cache.get("b2") is not None
    assert manager.cpu_cache.get("b1") is not None
    assert manager.gpu_cache.size == 1


def test_cpu_eviction_demotes_old_block_to_ssd(tmp_path) -> None:
    manager = CacheManager(_dummy_config(tmp_path, gpu_blocks=0, cpu_blocks=1))

    manager.put_kv_block(_make_block("v1", modality=Modality.VISUAL))
    manager.put_kv_block(_make_block("v2", modality=Modality.VISUAL))

    assert manager.cpu_cache.get("v2") is not None
    assert manager.ssd_cache.exists("v1")


def test_cpu_promotion_updates_eviction_state_before_next_gpu_insert(tmp_path) -> None:
    manager = CacheManager(_dummy_config(tmp_path, gpu_blocks=1, cpu_blocks=1))

    manager.put_kv_block(_make_block("visual1", modality=Modality.VISUAL))
    promoted = manager.get_kv_block("visual1")
    assert promoted is not None
    assert manager.gpu_cache.get("visual1") is not None

    manager.put_kv_block(_make_block("text2", modality=Modality.TEXT))

    assert manager.gpu_cache.size == 1
    assert manager.gpu_cache.get("text2") is not None
    assert manager.cpu_cache.get("visual1") is not None

