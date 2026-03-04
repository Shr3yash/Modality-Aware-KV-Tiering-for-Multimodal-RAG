import os
from pathlib import Path

import torch

from src.cache.ssd_cache import SSDCache


def test_ssd_cache_store_and_load_tmpdir(tmp_path: Path) -> None:
    cache = SSDCache(str(tmp_path), capacity_gb=1.0)
    k = torch.randn(2, 2)
    v = torch.randn(2, 2)
    metadata = {"modality": "text", "token_ids": [1, 2], "num_tokens": 2, "layer_range": [0, 1]}

    ok = cache.store("block1", k, v, metadata)
    assert ok
    assert cache.exists("block1")

    loaded = cache.load("block1")
    assert loaded is not None
    k2, v2, meta2 = loaded
    assert torch.allclose(k.cpu(), k2)
    assert torch.allclose(v.cpu(), v2)
    assert meta2["modality"] == "text"

    assert cache.delete("block1")
    assert not cache.exists("block1")

