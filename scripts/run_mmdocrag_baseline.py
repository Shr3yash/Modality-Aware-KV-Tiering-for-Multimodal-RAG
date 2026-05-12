#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psutil

from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge
from rag.vllm_metrics import scrape_prometheus_metrics


def get_host_memory_stats():
    vm = psutil.virtual_memory()
    return {
        "ram_total_gb": vm.total / (1024**3),
        "ram_used_gb": vm.used / (1024**3),
        "ram_available_gb": vm.available / (1024**3),
        "ram_percent": vm.percent,
    }

def get_dir_size_gb(path: str):
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)


def _lmcache_prometheus_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k.startswith("lmcache:")}


def build_lmcache_payload(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    dir_size_before_gb: float,
    dir_size_after_gb: float,
) -> dict:
    prom_before = _lmcache_prometheus_subset(metrics_before)
    prom_after = _lmcache_prometheus_subset(metrics_after)
    keys = set(prom_before) | set(prom_after)
    delta = {
        k: float(prom_after.get(k, 0.0)) - float(prom_before.get(k, 0.0)) for k in keys
    }
    return {
        "prometheus_metrics_before": prom_before,
        "prometheus_metrics_after": prom_after,
        "prometheus_metrics_delta": delta,
        "local_disk_size_gb_before": dir_size_before_gb,
        "local_disk_size_gb_after": dir_size_after_gb,
        "local_disk_size_gb_delta": dir_size_after_gb - dir_size_before_gb,
    }


def _parse_args() -> argparse.Namespace:
    defaults = RAGConfig()
    p = argparse.ArgumentParser(description="MMDocRAG baseline eval with LMCache metrics.")
    p.add_argument(
        "--eval-slice-start",
        type=int,
        default=None,
        metavar="I",
        help=f"First JSONL line index (0-based). Default: {defaults.eval_slice_start}.",
    )
    p.add_argument(
        "--eval-slice-stop",
        type=int,
        default=None,
        metavar="J",
        help="Exclusive end line index (Python slice stop). Omit for end of file.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        metavar="N",
        help=f"Keep at most N rows after slicing. Omit for RAGConfig default ({defaults.max_examples}). "
        "Use 0 for no cap (full slice window).",
    )
    return p.parse_args()


def _cfg_from_args(args: argparse.Namespace) -> RAGConfig:
    overrides: dict = {}
    if args.eval_slice_start is not None:
        overrides["eval_slice_start"] = args.eval_slice_start
    if args.eval_slice_stop is not None:
        overrides["eval_slice_stop"] = args.eval_slice_stop
    if args.max_examples is not None:
        overrides["max_examples"] = None if args.max_examples == 0 else args.max_examples
    return RAGConfig(**overrides)


if __name__ == "__main__":
    cfg = _cfg_from_args(_parse_args())
    path = "/home/runying2/lmcache_storage"
    metrics_before = scrape_prometheus_metrics()
    host_memory_stats_before = get_host_memory_stats()
    dir_size_before = get_dir_size_gb(path)
    baseline_results = run_baseline(cfg)
    metrics_after = scrape_prometheus_metrics()
    host_memory_stats_after = get_host_memory_stats()
    dir_size_after = get_dir_size_gb(path)
    offline_results = run_offline_judge(cfg)

    lmcache_payload = build_lmcache_payload(
        metrics_before, metrics_after, dir_size_before, dir_size_after
    )

    final_results = {
        "summary": offline_results["summary"],
        "rows": offline_results["rows"],
        "lmcache": lmcache_payload,
        "system_utilization": {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "host_memory_stats_before": host_memory_stats_before,
            "host_memory_stats_after": host_memory_stats_after,
            "lmcache_dir_size_gb_before": dir_size_before,
            "lmcache_dir_size_gb_after": dir_size_after,
            "lmcache_dir_size_gb_delta": dir_size_after - dir_size_before,
        },
    }

    output_path = cfg.output_dir / "final_results_with_utilization.json"
    judged_path = cfg.output_dir / "baseline_results_judged.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    with judged_path.open("r", encoding="utf-8") as f:
        judged_data = json.load(f)
    judged_data["lmcache"] = lmcache_payload
    with judged_path.open("w", encoding="utf-8") as f:
        json.dump(judged_data, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")
    print(f"Updated judged results with lmcache block: {judged_path}")
    pb = lmcache_payload["prometheus_metrics_before"]
    pa = lmcache_payload["prometheus_metrics_after"]
    print(
        "LMCache Prometheus metrics (before): "
        + (json.dumps(pb, indent=2) if pb else "(none — check vLLM /metrics or LMCACHE_METRICS_URL)")
    )
    print(
        "LMCache Prometheus metrics (after): "
        + (json.dumps(pa, indent=2) if pa else "(none — check vLLM /metrics or LMCACHE_METRICS_URL)")
    )
    print(
        "LMCache block in JSON (jq):\n"
        f"  jq '.lmcache' {output_path}\n"
        f"  jq '.lmcache' {judged_path}"
    )
