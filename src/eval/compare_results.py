"""Compare two benchmark result JSON files and print a markdown table.

Usage:
    python -m src.eval.compare_results results/baseline.json results/eviction_only.json
"""
from __future__ import annotations

import argparse
import json
import sys


_METRICS = [
    ("Mean TTFT (ms)", "mean_ttft_ms", "latency"),
    ("P50 TTFT (ms)", "p50_ttft_ms", "latency"),
    ("P99 TTFT (ms)", "p99_ttft_ms", "latency"),
    ("Mean Latency (ms)", "mean_latency_ms", "latency"),
    ("Cache Hit Rate (text)", "cache_hit_rate_text", "rate"),
    ("Cache Hit Rate (visual)", "cache_hit_rate_visual", "rate"),
    ("Mean GPU Memory (MiB)", "mean_gpu_memory_mib", "latency"),
]


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(result_a: dict, result_b: dict) -> str:
    sa = result_a["summary"]
    sb = result_b["summary"]
    name_a = result_a["config_name"]
    name_b = result_b["config_name"]

    col_a = max(len(name_a), 12)
    col_b = max(len(name_b), 12)

    header = f"| {'Metric':<30} | {name_a:>{col_a}} | {name_b:>{col_b}} | {'Delta':>10} |"
    sep = f"|{'-' * 32}|{'-' * (col_a + 2)}|{'-' * (col_b + 2)}|{'-' * 12}|"
    lines = [header, sep]

    for label, key, kind in _METRICS:
        va = sa.get(key, 0.0)
        vb = sb.get(key, 0.0)

        if kind == "rate":
            va_s = f"{va:.1%}"
            vb_s = f"{vb:.1%}"
            delta = f"{vb - va:+.1%} pp"
        else:
            va_s = f"{va:.2f}"
            vb_s = f"{vb:.2f}"
            delta = f"{(vb - va) / va * 100:+.1f}%" if va else "N/A"

        lines.append(
            f"| {label:<30} | {va_s:>{col_a}} | {vb_s:>{col_b}} | {delta:>10} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files")
    parser.add_argument("file_a", help="First result JSON file")
    parser.add_argument("file_b", help="Second result JSON file")
    args = parser.parse_args()

    result_a = load_result(args.file_a)
    result_b = load_result(args.file_b)
    print(compare(result_a, result_b))


if __name__ == "__main__":
    main()
