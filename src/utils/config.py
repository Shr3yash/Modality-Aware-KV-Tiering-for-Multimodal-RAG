from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
import torch

def _gpu_device() -> str:
    if torch.backends.mps.is_available():
        # print("The current device is mps ...")
        return "mps"
    if torch.cuda.is_available():
        # print("The current device is cuda ...")
        return "cuda"
    # print("The current device is cpu ...")
    return "cpu"


class ModelConfig(BaseModel):
    name: str
    dtype: str = "float16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True


class RagConfig(BaseModel):
    chunk_size_tokens: int = 512
    top_k: int = 5
    embedding_model: str = "openai/clip-vit-large-patch14"
    corpus_dir: str = "./data/corpus"


class CacheConfig(BaseModel):
    gpu_blocks: int = 2048
    gpu_device: str = Field(default_factory=_gpu_device)
    cpu_blocks: int = 8192
    ssd_capacity_gb: float = 50.0
    ssd_path: str = "/tmp/kv_ssd_cache"
    block_size: int = 16
    eviction_policy: str = "modality_aware_lru"
    text_gpu_pin_ratio: float = 0.7
    recompute_top_k: int = 32


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent: int = 16


class EvalConfig(BaseModel):
    num_samples: int = 500
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    warmup_queries: int = 10


class Config(BaseModel):
    model: ModelConfig
    rag: RagConfig
    cache: CacheConfig
    serving: ServingConfig
    eval: EvalConfig


def load_config(path: str | Path) -> Config:
    """Load YAML configuration into a strongly typed Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        rag=RagConfig(**raw.get("rag", {})),
        cache=CacheConfig(**raw.get("cache", {})),
        serving=ServingConfig(**raw.get("serving", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )
