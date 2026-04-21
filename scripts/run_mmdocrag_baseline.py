from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge
import requests
from typing import Dict, Optional
import os
import psutil
from pathlib import Path
import json

def scrape_prometheus_metrics(metrics_url: str = "http://127.0.0.1:8000/metrics") -> Dict[str, float]:
    wanted = {
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_cached",
        "vllm:prompt_tokens_recomputed",
        "lmcache:num_requested_tokens",
        "lmcache:num_hit_tokens",
        "lmcache:num_stored_tokens",
        "lmcache:retrieve_hit_rate",
    }

    out: Dict[str, float] = {}
    text = requests.get(metrics_url, timeout=5).text

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split("{")[0]
        if metric_name in wanted:
            try:
                out[metric_name] = float(parts[-1])
            except ValueError:
                pass
    return out

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

if __name__ == "__main__":
    cfg = RAGConfig()
    path = "/home/runying2/lmcache_storage"
    metrics_before = scrape_prometheus_metrics()
    host_memory_stats_before = get_host_memory_stats()
    dir_size_before = get_dir_size_gb(path)
    baseline_results = run_baseline(cfg)
    metrics_after = scrape_prometheus_metrics()
    host_memory_stats_after = get_host_memory_stats()
    dir_size_after = get_dir_size_gb(path)
    offline_results = run_offline_judge(cfg)

    final_results = {
        "summary": offline_results["summary"],
        "rows": offline_results["rows"],
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")