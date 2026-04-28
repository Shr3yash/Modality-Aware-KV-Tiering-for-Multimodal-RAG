from pathlib import Path

from src.utils.config import load_config


def test_phase2_default_config_smoke_test() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")

    assert config.model.name == "Qwen/Qwen2.5-Omni-3B"
    assert config.model.dtype == "float16"
    assert config.model.max_model_len == 4096
    assert config.rag.top_k == 5
