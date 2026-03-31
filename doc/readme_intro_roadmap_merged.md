# Proposed README Sections

This draft merges the current implementation-focused README language with the
higher-level system plan from `plan_A_balanced_prototype.md`.

---

## Replacement Intro

## Modality-Aware KV Tiering for Multimodal RAG

This project builds a **working multimodal RAG serving prototype** for vision-language
models (VLMs) that answer questions over text and images. The system is designed around
two practical goals:

- **reuse expensive intermediate computation** through KV-cache reuse, and
- **tier KV blocks across GPU, CPU, and SSD** so serving remains efficient under memory pressure.

The longer-term target is a vLLM-based serving stack that combines:

- **CacheClip-style KV reuse** for repeated prefixes and repeated retrieved context,
- **Mooncake-style KV tiering** across GPU HBM and host memory, and
- a **modality-aware placement policy** that treats text and visual KV blocks differently.

The core idea is that multimodal prompts are not uniform. In many vision-language RAG
workloads, **instruction text, user queries, and top retrieved text evidence carry higher
semantic value per token**, while **visual tokens are often more numerous and more
redundant**. Instead of treating all KV blocks equally, this project prioritizes:

- keeping **high-value text KV** on GPU when possible,
- spilling **visual KV** to CPU first under pressure, and
- using **SSD as an optional cold tier** for overflow or stress-test settings.

This repository currently contains the **Phase 1 cache foundation** for that system:

- typed configuration, structured logging, metrics, and profiling utilities,
- core KV block data structures and chunk hashing,
- GPU, CPU, and SSD cache tiers,
- LRU and modality-aware eviction policies, and
- a cache manager that orchestrates promotion, demotion, and tier-aware lookup.

The multimodal RAG pipeline, vLLM integration, serving API, and evaluation harness are
the next layers to be built on top of this foundation.

---

## Replacement Roadmap

## Roadmap

The project is organized as a staged prototype that moves from cache infrastructure to a
full multimodal RAG serving system with measurable latency, memory, and quality tradeoffs.

### Phase 1 - Cache foundation

Status: complete as of 2026-03-31.

- Implement KV block metadata and modality tagging.
- Build GPU, CPU, and SSD cache tiers.
- Add LRU and modality-aware eviction.
- Add the cache manager, metrics hooks, and smoke tests.

Completion notes:

- The cache foundation now exists under `src/cache` and `src/utils`.
- Cross-tier behavior is covered by the current `tests/` suite.
- The remaining work starts at multimodal serving integration rather than cache internals.

### Phase 2 - Multimodal bring-up

- Run a stable vLLM-supported image-text model.
- Build a small multimodal RAG demo with text chunks and image inputs.
- Log prompt composition and token counts by modality and segment role.

Implementation note:

- A concrete Phase 2 execution plan based on the current cache/data-flow docs is captured in `logs/phase2_concrete_plan.md`.

### Phase 3 - MAST instrumentation and policy

- Tag KV blocks during prefill by modality and role:
  `system`, `user`, `retrieved_text`, `visual_context`, and `generated`.
- Introduce a rule-based **Modality-Aware Split-Tiering (MAST)** policy.
- Keep high-priority text blocks in GPU HBM when capacity allows.
- Spill visual KV to CPU first, with SSD as an optional cold tier.

### Phase 4 - KV reuse and optimization

- Reuse repeated system prompts and repeated retrieved prefixes when safe.
- Improve request grouping or routing to increase prefix reuse.
- Explore selective recomputation and visual-side reuse for repeated inputs.

### Phase 5 - Evaluation

- Benchmark TTFT, end-to-end latency, throughput, and tier usage.
- Compare against baselines:
  no reuse, reuse-only, tiering-only, and modality-aware tiering.
- Measure quality impact on text-only, image-grounded, and mixed-evidence workloads.
- Run ablations on visual offload thresholds, pin budgets, and optional SSD usage.

### Expected deliverables

- a working vLLM-based multimodal RAG demo,
- a modality-aware KV tiering policy implementation,
- a benchmark and evaluation harness,
- latency, cost, and quality plots, and
- engineering notes describing the system design and tradeoffs.

### Near-term direction

The most important next step is to connect the existing cache foundation to a real
multimodal serving path. Once that integration exists, the main research question becomes:

> Can modality-aware placement reduce TTFT and GPU memory pressure more effectively than
> modality-agnostic tiering, while preserving answer quality better than uniform offloading?

That question defines the rest of the roadmap.
