## Modality-Aware KV Tiering for Multimodal RAG

This project builds a serving system for large vision-language models (VLMs) that can answer questions over text and images.  
Its main goal is to **reuse expensive intermediate computations (KV cache)** and **store them across GPU, CPU, and SSD** so you can:

- **Serve more queries per second**
- **Fit larger batches on the same GPU**
- **Keep latency low**, especially the time to first token (TTFT)

This README explains everything that is implemented so far **from scratch**, in plain language.

---

## What is a KV cache and why tier it?

Modern language models process input in tokens (pieces of text, and for VLMs, also visual tokens).  
At each layer, the model computes **Key (K)** and **Value (V)** tensors for attention. These tensors:

- Are **large** (many GBs for long contexts)
- Are **expensive to recompute**
- Live in **GPU memory (HBM)** during generation

A **KV cache** is a way to remember these K/V tensors so future queries that reuse similar context do **not** have to recompute them.

However:

- GPU memory is small but fast.
- CPU RAM is larger but slower.
- SSD is much larger but much slower again.

So we build a **tiered cache**:

- **GPU tier**: hottest, smallest, fastest storage for KV blocks.
- **CPU tier**: warm, larger, still relatively fast.
- **SSD tier**: cold, largest, but slowest.

We then add a **modality-aware policy** so that:

- **Text** KV blocks are preferentially kept on GPU.
- **Visual** KV blocks are more likely to be evicted to CPU/SSD first.

This matches the observation that text tokens are often reused much more across queries than visual tokens.

---

## Project layout (current state)

At the moment, the following pieces have been implemented and are ready to extend:

- **Project metadata**
  - `pyproject.toml` – Python package configuration and pinned dependencies.
  - `requirements.txt` – A flat list of pinned dependencies for `pip install -r`.

- **Configuration & utilities**
  - `src/utils/config.py` – Typed config loader using Pydantic.
  - `src/utils/logging.py` – Structured logging with `structlog`.
  - `src/utils/metrics.py` – Prometheus-compatible metrics for cache stats.
  - `src/utils/profiling.py` – Simple CUDA timing and memory snapshots.

- **Cache subsystem (Phase 1 foundation)**
  - `src/cache/kv_block.py` – Core `KVBlock` data structure and modality tagging.
  - `src/cache/chunk_hash.py` – Content-based hashing for chunks.
  - `src/cache/ssd_cache.py` – SSD-backed storage using `safetensors`.
  - `src/cache/gpu_cache.py` – In-GPU KV block storage.
  - `src/cache/cpu_cache.py` – In-CPU KV block storage with pinned memory.
  - `src/cache/eviction.py` – LRU and Modality-Aware LRU eviction policies.
  - `src/cache/cache_manager.py` – Orchestrator that moves blocks across GPU/CPU/SSD.

- **Initial tests**
  - `tests/test_ssd_cache.py` – Verifies storing/loading KV blocks to/from SSD.
  - `tests/test_eviction.py` – Verifies basic behavior of the eviction policies.
  - `tests/test_cache_manager.py` – Smoke test that insertion and lookup work.

The RAG pipeline, model wrapper, serving API, and evaluation harness will be built **on top of this foundation**.

---

## Dependencies and environment (what is pinned and why)

The project pins exact versions in **both** `pyproject.toml` and `requirements.txt` so runs are reproducible.  
Key dependencies:

- **vLLM `0.6.3`** – High-performance LLM/VLM inference engine.  
- **PyTorch `2.2.1` + CUDA 12.1** – Core tensor library and GPU support.
- **Transformers `4.40.0`** – Model and tokenizer utilities.
- **FastAPI / Uvicorn** – HTTP server for the `/generate` and `/health` endpoints (to be wired later).
- **Pydantic `2.x`** – Typed configuration system.
- **FAISS + open-clip-torch** – Vector search and CLIP embeddings (for later RAG stages).
- **structlog** – Structured logging.
- **prometheus-client** – Metrics export.
- **safetensors** – Fast, safe tensor serialization for SSD storage.
- **pytest / pytest-asyncio** – Testing.

For now, you can install everything with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Later, `setup.sh` will automate this end-to-end.

---

## Config system (`configs/default.yaml` + `src/utils/config.py`)

### What the config file contains

`configs/default.yaml` holds all tunable parameters, grouped into sections:

- **`model`**: model name, dtype, max sequence length, GPU memory utilization.
- **`rag`**: RAG-specific settings (chunk size, number of retrieved chunks, embedding model, corpus directory).
- **`cache`**: number of GPU/CPU blocks, SSD capacity and path, block size, eviction policy, text GPU pin ratio, recompute top-K.
- **`serving`**: host, port, and concurrency settings.
- **`eval`**: number of samples, batch sizes, warmup queries.

The default values follow the specification, e.g.:

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Cache:
  - `gpu_blocks: 2048`
  - `cpu_blocks: 8192`
  - `ssd_capacity_gb: 50`
  - `eviction_policy: "modality_aware_lru"`
  - `text_gpu_pin_ratio: 0.7`

### How the config is loaded

`src/utils/config.py` defines Pydantic models:

- `ModelConfig`, `RagConfig`, `CacheConfig`, `ServingConfig`, `EvalConfig`
- A top-level `Config` that groups them all.

Usage pattern:

```python
from src.utils.config import load_config

config = load_config("configs/default.yaml")
print(config.cache.gpu_blocks)  # e.g., 2048
```

This gives you **type-checked** access to configuration fields instead of juggling raw dictionaries.

---

## Logging, metrics, and profiling

### Logging (`src/utils/logging.py`)

- **Goal**: provide structured logs that are easy to filter and analyze.
- Uses `structlog` on top of the standard `logging` module.
- `configure_logging()` sets up JSON-formatted logs with timestamps and log levels.
- `get_logger(name)` returns a logger you can call like:

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("cache_insert", block_id="abc123", tier="gpu")
```

### Metrics (`src/utils/metrics.py`)

This file declares **Prometheus metrics** for cache behavior:

- **`KV_HITS`** – counter labeled by `tier` and `modality`.
- **`KV_MISSES`** – counter labeled similarly.
- **`KV_EVICTIONS`** – number of evictions per tier and modality.
- **`KV_PROMOTION_LATENCY`** – histogram for latency when promoting between tiers.
- **`KV_TIER_USAGE`** – gauge, current number of blocks per tier.

The cache manager will update these metrics as it moves blocks around, enabling you to monitor:

- Hit/miss ratios
- How often text vs visual blocks are evicted
- How long promotions take (e.g., SSD → CPU → GPU)

### Profiling (`src/utils/profiling.py`)

Two helpers:

- **`cuda_timing` context manager** – wraps a code block with CUDA events (if GPU is available) to measure elapsed time in ms.
- **`memory_snapshot`** – returns a simple dictionary of CUDA allocated/reserved memory in MB.

These will later be used to profile TTFT and memory usage around generation and cache operations.

---

## KVBlock and modality (`src/cache/kv_block.py`)

### Modality enum

`Modality` is an `Enum` with three possible values:

- `TEXT`
- `VISUAL`
- `MIXED`

This allows us to tag each KV block with **what kind of content it mainly represents**, which is crucial for modality-aware eviction.

### KVBlock dataclass

`KVBlock` is a `dataclass` that holds:

- **`block_id`** – a unique identifier (SHA-256 hash string).
- **`modality`** – a `Modality` enum value.
- **`token_ids`** – the underlying token IDs (as an immutable tuple of ints).
- **`k_cache` / `v_cache`** – PyTorch tensors storing the K/V entries.
- **`num_tokens`** – number of tokens.
- **`layer_range`** – tuple `(start_layer, end_layer)` describing which layers this block covers.
- **`last_access_time` / `access_count`** – for LRU-style policies.
- **`tier`** – current tier (`"gpu"`, `"cpu"`, `"ssd"`).
- **`pinned`** – if `True`, the block should not be evicted from its current tier.
- **`recompute_mask`** – optional boolean tensor for selective recomputation (CacheClip-style).

There is also a `touch()` method that:

- Updates `last_access_time` to `now`.
- Increments `access_count`.

This is called whenever we use the block, so eviction policies can make better decisions.

---

## Chunk hashing (`src/cache/chunk_hash.py`)

To reuse KV blocks safely across queries, we need a **stable way to identify a chunk** of content.  
`compute_chunk_hash` does this by:

- Taking `token_ids` (list of ints) and a `modality` string.
- Building a string like `"text:1,2,3,4"`.
- Computing a **SHA-256 hash** of that string.
- Returning the **first 32 hex characters** as the block ID.

This means:

- The **same tokens with the same modality** produce the **same `block_id`**.
- Different content or different modality leads to a different `block_id`.

This is crucial so that repeated chunks across queries share the same KV cache entry.

---

## SSD cache (`src/cache/ssd_cache.py`)

The `SSDCache` class handles **persisting KV blocks on disk** using `safetensors`:

- **Constructor**:
  - Takes `ssd_path` (directory) and `capacity_gb` (how much data you allow on disk).
  - Ensures the directory exists.
  - Stores capacity in **bytes**.

- **`store(block_id, k_cache, v_cache, metadata)`**:
  - Checks if current disk usage is under `capacity_bytes`.
  - Saves `k_cache` and `v_cache` as a `.safetensors` file.
  - Saves `metadata` in a simple `.meta` text file.
  - Returns `True` if successful, `False` if over capacity.

- **`load(block_id)`**:
  - Reads tensors and metadata from disk.
  - Returns `(k_tensor, v_tensor, metadata_dict)` or `None` if missing.

- **`delete(block_id)`**:
  - Removes both `.safetensors` and `.meta` files if they exist.

- **`exists(block_id)`**:
  - Checks if the `.safetensors` file exists.

- **`usage_bytes()`**:
  - Sums sizes of all files in the cache directory.

There is a test (`tests/test_ssd_cache.py`) that:

- Creates a temporary directory.
- Writes a small KV block.
- Loads it back and checks for numerical equality.
- Deletes it again.

---

## GPU and CPU caches (`src/cache/gpu_cache.py`, `src/cache/cpu_cache.py`)

Both classes are simple in-memory maps from `block_id` → `KVBlock`, with a fixed maximum number of entries:

### GPUCache

- Stores blocks with their `k_cache` / `v_cache` tensors moved to **CUDA**:

  - `block.k_cache = block.k_cache.to("cuda", non_blocking=True)`
  - `block.v_cache = block.v_cache.to("cuda", non_blocking=True)`

- Tracks the number of blocks for capacity checks.
- Provides:
  - `get(block_id)`, `add(block)`, `pop(block_id)`, and `has_capacity()`.

### CPUCache

- Stores blocks with their tensors moved to **CPU pinned memory**:

  - `.to("cpu", non_blocking=True, copy=True).pin_memory()`

- Pinned memory speeds up subsequent transfers back to GPU.
- Same API as `GPUCache`: `get`, `add`, `pop`, `has_capacity`, `size`.

These caches **do not** implement eviction themselves—that’s handled by the eviction policies and `CacheManager`.

---

## Eviction policies (`src/cache/eviction.py`)

### Common interface: `EvictionPolicy`

Every eviction policy implements:

- **`on_access(block_id)`** – called whenever a block is used, to update its position in LRU structures.
- **`evict(tier, modality_hint)`** – returns the `block_id` that should be evicted from the specified tier (or `None`).
- **`add(block_id, modality, tier)`** – register a new block in the policy.
- **`remove(block_id)`** – remove a block from internal tracking.

### Standard LRU: `LRUEvictionPolicy`

- Keeps a single `OrderedDict` of all blocks.
- `on_access` moves the block to the end (most recently used).
- `evict` pops the least recently used block.
- Does **not** pay attention to modality or tier hints.

### Modality-aware LRU: `ModalityAwareLRUEvictionPolicy`

This is the core contribution for tiering behavior:

- Keeps **separate LRU queues**:
  - `gpu_text` – text blocks on GPU.
  - `gpu_visual` – visual blocks on GPU.
  - `cpu_all` – all blocks on CPU.
  - `ssd_all` – all blocks on SSD.

- Tracks per-block:
  - Modality (`TEXT`, `VISUAL`, `MIXED`).
  - Tier (`gpu`, `cpu`, `ssd`).

- The GPU tier is split into two “sub-quotas”:
  - `text_gpu_pin_ratio * gpu_blocks` reserved for text.
  - Remaining blocks reserved for visual.

- **Eviction behavior on GPU**:
  - Prefer to evict from the **visual queue** first.
  - If text is overflowing its quota, evict the least recently used text block.
  - Otherwise, fall back to any block if needed.

- **CPU and SSD tiers**:
  - Use a single LRU queue each (modality-agnostic).

There is a test (`tests/test_eviction.py`) that ensures:

- Vanilla LRU evicts the least recently used block.
- Modality-aware policy prefers evicting visual blocks over text when GPU is full.

---

## Cache manager (`src/cache/cache_manager.py`)

`CacheManager` is the **central orchestrator** that:

- Knows about all three tiers (GPU, CPU, SSD).
- Uses an eviction policy to decide which blocks to demote.
- Tracks metrics as blocks are moved or evicted.

### Construction

It takes a `cache_config` object (the `CacheConfig` from Pydantic or a similar namespace) with fields like:

- `gpu_blocks`, `cpu_blocks`, `ssd_capacity_gb`, `ssd_path`
- `eviction_policy` – `"modality_aware_lru"` or `"lru"`
- `text_gpu_pin_ratio`

Depending on `eviction_policy`, it builds:

- A `ModalityAwareLRUEvictionPolicy` (preferred), or
- A simple `LRUEvictionPolicy`.

It also initializes Prometheus gauges for tier usage.

### Lookup: `get_kv_block(block_id)`

The logic is:

1. **Check GPU**:
   - If found: mark a hit, update eviction policy, `touch()` the block, return it.
2. **Check CPU**:
   - If found:
     - Mark a CPU hit.
     - **Promote CPU → GPU**:
       - Evict another GPU block if needed.
       - Move tensors to GPU.
       - Update metrics and eviction policy.
3. **Check SSD**:
   - If file exists:
     - Load tensors and metadata from disk.
     - Build a `KVBlock` instance (initial tier `ssd`).
     - **Promote SSD → CPU** (via `_promote_ssd_to_cpu`).
     - Then **CPU → GPU** (as above).
4. If not found anywhere:
   - Record a **miss** and return `None`.

### Insert: `put_kv_block(block)`

When a new KV block is created:

- Decide **initial tier**:
  - **Text** and **mixed** blocks: try to place on **GPU** first.
  - **Visual** blocks: prefer **CPU** (less pressure on GPU).

- If the target tier has no capacity:
  - Call `_evict_from_tier` which:
    - Asks the eviction policy which block to evict.
    - Pops it from the corresponding tier.
    - Demotes it (e.g., GPU → CPU, or CPU → SSD).
    - Updates eviction and metrics.

- Finally:
  - Add the block to the target tier.
  - Register it with the eviction policy.

### Tier transfer helpers

`CacheManager` provides several internal methods:

- **`_gpu_to_cpu(block)`** – move tensors to pinned CPU memory.
- **`_cpu_to_gpu(block)`** – move tensors to GPU.
- **`_cpu_to_ssd(block)`** – serialize to disk via `SSDCache.store`.
- **`_ssd_to_cpu(block)`** – conceptual reverse step (loading is mostly handled by `SSDCache.load`).
- **`_promote_cpu_to_gpu(block)`** – composite method used during lookup.
- **`_promote_ssd_to_cpu(block)`** – composite method used during lookup.

These methods also:

- Measure promotion latency and record it in the Prometheus histogram.
- Update tier usage gauges.

### Basic test (`tests/test_cache_manager.py`)

This test:

- Constructs a small dummy config.
- Creates a single `KVBlock` with random tensors.
- Inserts it via `put_kv_block`.
- Fetches it back with `get_kv_block`.
- Asserts that:
  - The block is returned.
  - It is in the GPU tier (as expected for text).

---

## What comes next (roadmap)

Everything implemented so far is **Phase 1: Foundation**. Next phases will add:

- **RAG pipeline**:
  - Document store and multimodal chunking.
  - CLIP-based retriever with FAISS.
  - Prompt builder for feeding the VLM.

- **KV reuse (CacheClip-style)**:
  - Attention scorer and recomputation planner.
  - Chunk stitcher to merge cached and recomputed KV.

- **Serving stack**:
  - vLLM-based VLM wrapper.
  - Tiered engine that consults the cache.
  - FastAPI server exposing `/generate`, `/health`, `/metrics`.

- **Evaluation harness**:
  - Dataset loaders (HotpotQA, TextVQA, etc.).
  - TTFT, latency, throughput, and accuracy metrics.
  - Baselines (no cache, prefix cache) vs. our full system.

As these phases are implemented, this README can be extended with:

- End-to-end setup instructions (`setup.sh`).
- Example curl commands against the API.
- Benchmark commands and sample result tables.

For now, you can treat this document as a **guided tour of the cache foundation** that the rest of the system will rely on.

