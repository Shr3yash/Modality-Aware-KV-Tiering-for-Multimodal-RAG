from pathlib import Path

import torch

from src.cache.cpu_cache import CPUCache
from src.cache.gpu_cache import GPUCache
from src.cache.kv_block import KVBlock, Modality
from src.cache.ssd_cache import SSDCache
from src.utils.config import _gpu_device


def _make_block(block_id: str, modality: Modality = Modality.TEXT) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        modality=modality,
        token_ids=(1, 2, 3),
        k_cache=torch.randn(1, 1, 4, 2),
        v_cache=torch.randn(1, 1, 4, 2),
        num_tokens=4,
        layer_range=(0, 0),
    )


def test_gpu_cache_add_get_and_pop_uses_configured_device() -> None:
    cache = GPUCache(max_blocks=1, device=_gpu_device())
    block = _make_block("g1")

    cache.add(block)

    got = cache.get("g1")
    assert got is not None
    assert got.tier == "gpu"
    assert got.k_cache is not None
    assert got.v_cache is not None
    assert got.k_cache.device.type == cache.device
    assert got.v_cache.device.type == cache.device

    popped = cache.pop("g1")
    assert popped is not None
    assert cache.get("g1") is None


def test_cpu_cache_add_get_and_pop_keeps_cpu_tensors() -> None:
    cache = CPUCache(max_blocks=1)
    block = _make_block("c1")

    cache.add(block)

    got = cache.get("c1")
    assert got is not None
    assert got.tier == "cpu"
    assert got.k_cache is not None
    assert got.v_cache is not None
    assert got.k_cache.device.type == "cpu"
    assert got.v_cache.device.type == "cpu"

    popped = cache.pop("c1")
    assert popped is not None
    assert cache.get("c1") is None


def test_ssd_cache_missing_block_returns_none(tmp_path: Path) -> None:
    cache = SSDCache(str(tmp_path), capacity_gb=1.0)
    assert cache.load("missing") is None
    assert not cache.exists("missing")
    assert not cache.delete("missing")


def test_ssd_cache_zero_capacity_refuses_store(tmp_path: Path) -> None:
    cache = SSDCache(str(tmp_path), capacity_gb=0.0)
    block = _make_block("s1")

    ok = cache.store("s1", block.k_cache, block.v_cache, {"modality": "text"})

    assert not ok
    assert not cache.exists("s1")


def test_ssd_cache_invalid_metadata_falls_back_to_empty_dict(tmp_path: Path) -> None:
    cache = SSDCache(str(tmp_path), capacity_gb=1.0)
    block = _make_block("s2")

    assert cache.store("s2", block.k_cache, block.v_cache, {"modality": "text"})
    cache._meta_path("s2").write_text("{not-valid", encoding="utf-8")

    loaded = cache.load("s2")
    assert loaded is not None
    _, _, metadata = loaded
    assert metadata == {}

