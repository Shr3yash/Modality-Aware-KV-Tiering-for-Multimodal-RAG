"""Prometheus scrape helpers for vLLM (and optional LMCache) /metrics endpoints."""

from __future__ import annotations

import os
from typing import Dict
from urllib.parse import urlparse, urlunparse

import requests


def _merge_lines_into_metrics(
    lines: list[str],
    alias_to_canonical: dict[str, str],
    lmcache_only: bool,
) -> tuple[Dict[str, float], int]:
    out: Dict[str, float] = {}
    parse_errors = 0
    lmcache_keys = {k for k in alias_to_canonical.values() if k.startswith("lmcache:")}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split("{")[0]
        canonical_name = alias_to_canonical.get(metric_name)
        if canonical_name is None:
            continue
        if lmcache_only and canonical_name not in lmcache_keys:
            continue
        try:
            out[canonical_name] = float(parts[-1])
        except ValueError:
            parse_errors += 1
    return out, parse_errors


def scrape_prometheus_metrics(metrics_url: str = "http://127.0.0.1:8000/metrics") -> Dict[str, float]:
    """
    Scrape selected counters and gauges from a Prometheus text endpoint.

    **Model tokens processed (vLLM):** ``vllm:prompt_tokens_total`` plus
    ``vllm:generation_tokens_total`` are the cumulative prompt and generated
    token counters. ``vllm:prompt_tokens_cached`` / ``vllm:prompt_tokens_recomputed``
    and ``vllm:kv_cache_usage_perc`` add KV-cache detail.

    **LMCache:** usage bytes (``lmcache:local_cache_usage``,
    ``lmcache:remote_cache_usage``, ``lmcache:local_storage_usage``), hit rates,
    and request/token counters are merged when present on ``metrics_url``. If
    ``curl …/metrics`` has no ``lmcache:`` lines (common when multiprocess
    exposition is split), set env ``LMCACHE_METRICS_URL`` to the full URL of the
    endpoint that exposes LMCache series; those keys are merged into the same
    dict.
    """
    wanted = {
        "vllm:kv_cache_usage_perc": ["vllm:kv_cache_usage_perc"],
        "vllm:prompt_tokens_cached": [
            "vllm:prompt_tokens_cached",
            "vllm:prompt_tokens_cached_total",
        ],
        "vllm:prompt_tokens_recomputed": [
            "vllm:prompt_tokens_recomputed",
            "vllm:prompt_tokens_recomputed_total",
        ],
        "vllm:prompt_tokens_total": [
            "vllm:prompt_tokens_total",
            "vllm:prompt_tokens",
        ],
        "vllm:generation_tokens_total": [
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
        ],
        "lmcache:local_cache_usage": ["lmcache:local_cache_usage"],
        "lmcache:remote_cache_usage": ["lmcache:remote_cache_usage"],
        "lmcache:local_storage_usage": ["lmcache:local_storage_usage"],
        "lmcache:retrieve_hit_rate": ["lmcache:retrieve_hit_rate"],
        "lmcache:lookup_hit_rate": ["lmcache:lookup_hit_rate"],
        "lmcache:num_requested_tokens": [
            "lmcache:num_requested_tokens",
            "lmcache:num_requested_tokens_total",
        ],
        "lmcache:num_hit_tokens": [
            "lmcache:num_hit_tokens",
            "lmcache:num_hit_tokens_total",
        ],
        "lmcache:num_stored_tokens": [
            "lmcache:num_stored_tokens",
            "lmcache:num_stored_tokens_total",
        ],
        "lmcache:num_retrieve_requests": [
            "lmcache:num_retrieve_requests",
            "lmcache:num_retrieve_requests_total",
        ],
        "lmcache:num_store_requests": [
            "lmcache:num_store_requests",
            "lmcache:num_store_requests_total",
        ],
        "lmcache:num_lookup_requests": [
            "lmcache:num_lookup_requests",
            "lmcache:num_lookup_requests_total",
        ],
        "lmcache:num_lookup_tokens": [
            "lmcache:num_lookup_tokens",
            "lmcache:num_lookup_tokens_total",
        ],
        "lmcache:num_lookup_hits": [
            "lmcache:num_lookup_hits",
            "lmcache:num_lookup_hits_total",
        ],
        "lmcache:num_vllm_hit_tokens": [
            "lmcache:num_vllm_hit_tokens",
            "lmcache:num_vllm_hit_tokens_total",
        ],
    }
    alias_to_canonical = {
        alias: canonical
        for canonical, aliases in wanted.items()
        for alias in aliases
    }

    response = requests.get(metrics_url, timeout=5)
    lines = response.text.splitlines()
    out, _ = _merge_lines_into_metrics(lines, alias_to_canonical, lmcache_only=False)

    extra = os.getenv("LMCACHE_METRICS_URL", "").strip()
    if extra:
        try:
            r2 = requests.get(extra, timeout=5)
            lmcache_out, _ = _merge_lines_into_metrics(
                r2.text.splitlines(), alias_to_canonical, lmcache_only=True
            )
            out.update(lmcache_out)
        except Exception:
            pass

    return out


def default_metrics_url_from_vlm_api_base(vlm_api_base: str) -> str:
    """Map OpenAI-compatible ``.../v1`` base URL to same host/port ``/metrics``."""
    stripped = (vlm_api_base or "").strip()
    if not stripped:
        return "http://127.0.0.1:8000/metrics"
    parsed = urlparse(stripped)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:8000/metrics"
    return urlunparse((parsed.scheme, parsed.netloc, "/metrics", "", "", ""))


def diff_prometheus_metrics(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, float]:
    keys = set(metrics_before) | set(metrics_after)
    return {
        k: float(metrics_after.get(k, 0.0)) - float(metrics_before.get(k, 0.0))
        for k in keys
    }
