from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

KV_HITS = Counter(
    "kv_cache_hits_total",
    "Number of KV cache hits",
    ["tier", "modality"],
)

KV_MISSES = Counter(
    "kv_cache_misses_total",
    "Number of KV cache misses",
    ["tier", "modality"],
)

KV_EVICTIONS = Counter(
    "kv_cache_evictions_total",
    "Number of KV cache evictions",
    ["tier", "modality"],
)

KV_PROMOTION_LATENCY = Histogram(
    "kv_cache_promotion_latency_seconds",
    "Latency for promoting KV blocks between tiers",
    ["src_tier", "dst_tier"],
)

KV_TIER_USAGE = Gauge(
    "kv_cache_tier_usage_blocks",
    "Number of KV blocks currently stored in a tier",
    ["tier"],
)

