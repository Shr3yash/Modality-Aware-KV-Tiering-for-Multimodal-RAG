from src.cache.eviction import LRUEvictionPolicy, ModalityAwareLRUEvictionPolicy
from src.cache.kv_block import Modality


def test_lru_eviction_basic() -> None:
    policy = LRUEvictionPolicy()
    policy.add("a", Modality.TEXT, "gpu")
    policy.add("b", Modality.VISUAL, "gpu")
    policy.on_access("a")
    evicted = policy.evict("gpu")
    assert evicted == "b"


def test_modality_aware_gpu_prefers_visual_eviction() -> None:
    policy = ModalityAwareLRUEvictionPolicy(text_gpu_pin_ratio=0.7, gpu_blocks=10)
    policy.add("t1", Modality.TEXT, "gpu")
    policy.add("t2", Modality.TEXT, "gpu")
    policy.add("v1", Modality.VISUAL, "gpu")
    policy.on_access("t1")
    evicted = policy.evict("gpu", modality_hint=Modality.TEXT)
    assert evicted == "v1"

