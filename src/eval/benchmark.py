"""Evaluation harness for comparing modality-aware KV cache configurations.

Three configurations are benchmarked:
  - baseline:              standard vLLM RAG, no custom caching
  - eviction_only:         modality-aware LRU eviction, no pruning
  - pruning_plus_eviction: both modality-aware eviction + visual token pruning

Usage:
    python -m src.eval.benchmark --mock --dataset synthetic
    python -m src.eval.benchmark --config configs/default.yaml --dataset vidore
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import structlog

from src.cache.kv_block import Modality
from src.utils.config import Config, load_config
from src.utils.logging import configure_logging
from src.utils.profiling import memory_snapshot

logger = structlog.get_logger(__name__)

# ── Configuration matrix ────────────────────────────────────────────
CONFIGS: Dict[str, Dict[str, bool]] = {
    "baseline": {"cache_enabled": False, "pruning_enabled": False},
    "eviction_only": {"cache_enabled": True, "pruning_enabled": False},
    "pruning_plus_eviction": {"cache_enabled": True, "pruning_enabled": True},
}

# ── Synthetic workload (20 queries × 5 documents) ──────────────────
SYNTHETIC_DOCS = [
    {
        "id": "doc_finances",
        "content": (
            "The quarterly earnings report shows revenue growth of 15% "
            "year-over-year. Operating margins improved to 22.3% driven by "
            "cost optimization initiatives. The board approved a dividend "
            "increase of $0.05 per share."
        ),
        "modality": "text",
    },
    {
        "id": "doc_architecture",
        "content": (
            "The microservices architecture uses gRPC for inter-service "
            "communication. Each service maintains its own database following "
            "the database-per-service pattern. Circuit breakers prevent "
            "cascade failures across the mesh."
        ),
        "modality": "text",
    },
    {
        "id": "doc_chart",
        "content": "[Image: revenue_chart.png]",
        "modality": "visual",
    },
    {
        "id": "doc_research",
        "content": (
            "Recent studies in transformer attention mechanisms show that "
            "sparse attention patterns can reduce computational complexity "
            "from O(n^2) to O(n*sqrt(n)) without significant accuracy "
            "degradation on downstream tasks."
        ),
        "modality": "text",
    },
    {
        "id": "doc_diagram",
        "content": "[Image: system_diagram.png]",
        "modality": "visual",
    },
]

SYNTHETIC_QUERIES = [
    "What was the revenue growth percentage?",
    "Describe the system architecture",
    "What does the revenue chart show?",
    "Explain sparse attention mechanisms",
    "What is shown in the system diagram?",
    "What were the operating margins?",
    "How do services communicate?",
    "What chart data is available?",
    "How does sparse attention reduce complexity?",
    "Describe the system components in the diagram",
    "Did the board approve any changes?",
    "What database pattern is used?",
    "Summarize the financial performance",
    "What prevents cascade failures?",
    "What trends does the chart reveal?",
    "Compare sparse and dense attention",
    "What is the dividend per share?",
    "Explain the database-per-service pattern",
    "What information is in the revenue image?",
    "How is transformer complexity reduced?",
]


# ── Data classes ────────────────────────────────────────────────────
@dataclass
class QueryResult:
    query_id: int
    query: str
    response: str
    ttft_ms: float
    total_latency_ms: float
    cache_hits_text: int
    cache_hits_visual: int
    cache_misses_text: int
    cache_misses_visual: int
    gpu_memory_mib: float
    chunks_retrieved: int


@dataclass
class BenchmarkResult:
    config_name: str
    timestamp: str
    num_queries: int
    results: List[Dict[str, Any]]
    summary: Dict[str, float]


# ── Helpers ─────────────────────────────────────────────────────────
def build_synthetic_chunks():
    from src.serving.rag_pipeline import RetrievedChunk

    chunks = []
    for doc in SYNTHETIC_DOCS:
        modality = Modality.TEXT if doc["modality"] == "text" else Modality.VISUAL
        n_tokens = 128 if modality == Modality.TEXT else 576
        chunks.append(
            RetrievedChunk(
                chunk_id=doc["id"],
                content=doc["content"],
                modality=modality,
                token_ids=list(range(n_tokens)),
                score=0.0,
                source_doc=doc["id"],
                image_path=None,
            )
        )
    return chunks


def make_config_variant(base: Config, variant: str) -> Config:
    cfg = base.model_copy(deep=True)
    settings = CONFIGS[variant]
    cfg.cache.enabled = settings["cache_enabled"]
    cfg.pruning.enabled = settings["pruning_enabled"]
    return cfg


def load_vidore_queries(num_queries: int) -> List[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("vidore/vidore_v3_hr", split="test")
        return [row["query"] for row in ds.select(range(min(num_queries, len(ds))))]
    except Exception as e:
        logger.warning("vidore_load_failed_using_synthetic", error=str(e))
        return SYNTHETIC_QUERIES


def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {}

    ttfts = sorted(r["ttft_ms"] for r in results)
    lats = sorted(r["total_latency_ms"] for r in results)

    total_ht = sum(r["cache_hits_text"] for r in results)
    total_mt = sum(r["cache_misses_text"] for r in results)
    total_hv = sum(r["cache_hits_visual"] for r in results)
    total_mv = sum(r["cache_misses_visual"] for r in results)
    text_total = total_ht + total_mt
    vis_total = total_hv + total_mv

    return {
        "mean_ttft_ms": sum(ttfts) / n,
        "p50_ttft_ms": ttfts[n // 2],
        "p99_ttft_ms": ttfts[int(n * 0.99)],
        "mean_latency_ms": sum(lats) / n,
        "cache_hit_rate_text": total_ht / text_total if text_total else 0.0,
        "cache_hit_rate_visual": total_hv / vis_total if vis_total else 0.0,
        "mean_gpu_memory_mib": sum(r["gpu_memory_mib"] for r in results) / n,
        "total_queries": n,
    }


# ── Core benchmark loop ────────────────────────────────────────────
async def run_benchmark(
    config_name: str,
    config: Config,
    queries: List[str],
    warmup: int = 5,
    mock: bool = False,
) -> BenchmarkResult:
    from src.serving.engine import CachedVLLMEngine

    engine = CachedVLLMEngine(config)

    if mock:
        engine._initialized = True
        engine.rag_pipeline.load_synthetic_chunks(build_synthetic_chunks())
    else:
        engine.initialize()

    # Warmup
    for i in range(min(warmup, len(queries))):
        await engine.generate(query=queries[i], max_tokens=64)
        logger.info("warmup_query", idx=i)

    # Benchmark
    results: List[Dict[str, Any]] = []
    for i, query in enumerate(queries):
        mem_before = memory_snapshot()
        result = await engine.generate(query=query, max_tokens=128)
        mem_after = memory_snapshot()
        peak_mem = max(
            mem_before.get("allocated_mb", 0),
            mem_after.get("allocated_mb", 0),
        )

        qr = QueryResult(
            query_id=i,
            query=query,
            response=result["response"],
            ttft_ms=result["ttft_ms"],
            total_latency_ms=result["total_latency_ms"],
            cache_hits_text=result["cache_hits_text"],
            cache_hits_visual=result["cache_hits_visual"],
            cache_misses_text=result["cache_misses_text"],
            cache_misses_visual=result["cache_misses_visual"],
            gpu_memory_mib=peak_mem,
            chunks_retrieved=result["chunks_retrieved"],
        )
        results.append(asdict(qr))
        logger.info("query_complete", query_id=i, ttft_ms=f"{result['ttft_ms']:.1f}")

    summary = _compute_summary(results)
    return BenchmarkResult(
        config_name=config_name,
        timestamp=datetime.now().isoformat(),
        num_queries=len(results),
        results=results,
        summary=summary,
    )


# ── CLI entry point ────────────────────────────────────────────────
async def _main_async(args) -> None:
    configure_logging()
    base_config = load_config(args.config)

    queries = (
        SYNTHETIC_QUERIES
        if args.dataset == "synthetic"
        else load_vidore_queries(args.num_queries)
    )

    configs_to_run = (
        args.configs.split(",") if args.configs else list(CONFIGS.keys())
    )

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            logger.error("unknown_config", name=config_name)
            continue

        logger.info("starting_benchmark", config=config_name)
        variant = make_config_variant(base_config, config_name)
        result = await run_benchmark(
            config_name=config_name,
            config=variant,
            queries=queries,
            warmup=base_config.eval.warmup_queries,
            mock=args.mock,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{config_name}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info("results_saved", path=str(out_path))

        print(f"\n{'=' * 60}")
        print(f"Config: {config_name}")
        print(f"{'=' * 60}")
        for k, v in result.summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark modality-aware KV cache configurations"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "vidore"],
        default="synthetic",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config names to run (default: all)",
    )
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without vLLM model (tests harness + cache logic)",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
