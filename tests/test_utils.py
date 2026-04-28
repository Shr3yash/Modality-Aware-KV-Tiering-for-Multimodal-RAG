import logging

import structlog
import torch

from src.cache.chunk_hash import compute_chunk_hash
from src.utils import config as config_module
from src.utils.config import CacheConfig, load_config
from src.utils.logging import get_logger, log_exception
from src.utils.profiling import cuda_timing, memory_snapshot


def test_compute_chunk_hash_is_deterministic_and_modality_sensitive() -> None:
    text_hash_1 = compute_chunk_hash([1, 2, 3], modality="text")
    text_hash_2 = compute_chunk_hash([1, 2, 3], modality="text")
    visual_hash = compute_chunk_hash([1, 2, 3], modality="visual")

    assert text_hash_1 == text_hash_2
    assert text_hash_1 != visual_hash
    assert len(text_hash_1) == 32


def test_gpu_device_prefers_mps_then_cuda_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: True)
    assert config_module._gpu_device() == "mps"

    monkeypatch.setattr(config_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: True)
    assert config_module._gpu_device() == "cuda"

    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)
    assert config_module._gpu_device() == "cpu"


def test_cache_config_uses_gpu_device_helper_by_default() -> None:
    assert CacheConfig().gpu_device == config_module._gpu_device()


def test_load_config_applies_defaults_and_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model:',
                '  name: "demo-model"',
                'cache:',
                '  gpu_blocks: 4',
                '  gpu_device: "mps"',
                '  text_gpu_pin_ratio: 0.5',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.name == "demo-model"
    assert config.cache.gpu_blocks == 4
    assert config.cache.gpu_device == "mps"
    assert config.cache.cpu_blocks == 8192
    assert config.cache.text_gpu_pin_ratio == 0.5
    assert config.serving.port == 8000


def test_get_logger_configures_structlog() -> None:
    structlog.reset_defaults()

    logger = get_logger("tests.logging")

    assert structlog.is_configured()
    assert logger is not None


def test_log_exception_delegates_to_logger() -> None:
    calls = {}

    class FakeLogger:
        def exception(self, msg, **kwargs):
            calls["msg"] = msg
            calls["kwargs"] = kwargs

    log_exception(FakeLogger(), "boom", block_id="b1")

    assert calls["msg"] == "boom"
    assert calls["kwargs"] == {"block_id": "b1"}


def test_cuda_timing_disabled_is_safe() -> None:
    with cuda_timing(enable=False) as elapsed:
        assert elapsed == 0.0


def test_cuda_timing_uses_cuda_events_when_available(monkeypatch) -> None:
    calls = {"record": 0, "sync": 0, "elapsed": 0}

    class FakeEvent:
        def __init__(self, enable_timing: bool):
            assert enable_timing is True

        def record(self) -> None:
            calls["record"] += 1

        def elapsed_time(self, other) -> float:
            calls["elapsed"] += 1
            return 12.5

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.__setitem__("sync", calls["sync"] + 1))

    with cuda_timing(enable=True) as elapsed:
        assert elapsed == 0.0

    assert calls["record"] == 2
    assert calls["sync"] == 1
    assert calls["elapsed"] == 1


def test_memory_snapshot_returns_zeros_for_cpu() -> None:
    snapshot = memory_snapshot(device=torch.device("cpu"))
    assert snapshot == {"allocated_mb": 0.0, "reserved_mb": 0.0}


def test_memory_snapshot_uses_cuda_stats_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 8 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 16 * 1024**2)

    snapshot = memory_snapshot(device=torch.device("cuda"))

    assert snapshot == {"allocated_mb": 8.0, "reserved_mb": 16.0}
    