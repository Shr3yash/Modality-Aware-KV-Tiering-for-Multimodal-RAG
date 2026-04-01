from types import SimpleNamespace

import torch

from src.cache.cache_manager import CacheManager
from src.cache.kv_block import KVBlock, Modality
from src.utils.config import _gpu_device

def _dummy_config() -> SimpleNamespace:
    return SimpleNamespace(
        gpu_blocks=2,
        gpu_device=_gpu_device(),
        cpu_blocks=2,
        ssd_capacity_gb=1.0,
        ssd_path="/tmp/kv_ssd_cache_test",
        block_size=16,
        eviction_policy="modality_aware_lru",
        text_gpu_pin_ratio=0.7,
        recompute_top_k=32,
    )


def test_cache_manager_insert_and_promote() -> None:
    cfg = _dummy_config()
    manager = CacheManager(cfg)

    k = torch.randn(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    block = KVBlock(
        block_id="b1",
        modality=Modality.TEXT,
        token_ids=tuple(range(16)),
        k_cache=k,
        v_cache=v,
        num_tokens=16,
        layer_range=(0, 0),
    )

    manager.put_kv_block(block)
    got = manager.get_kv_block("b1")
    assert got is not None
    assert got.tier == "gpu"
    assert got.k_cache is not None
    assert got.v_cache is not None
    assert got.k_cache.device.type == cfg.gpu_device
    assert got.v_cache.device.type == cfg.gpu_device
