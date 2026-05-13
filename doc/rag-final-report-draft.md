# RAG Subsystem — Final Report Draft

**Author:** Haochen Tong
**Branch:** `build/LMCache-based-mrag`
**Scope:** Retrieval-Augmented Generation (RAG) layer for the
*Modality-Aware KV Tiering for Multimodal RAG* project.
**Out-of-scope (teammates):** modality-aware KV cache (`src/cache/`),
vLLM serving, attention-based visual pruning (`rag/qwen2vl_catp_pruner_v2.py`).

---

## 1. Overview

The project investigates **modality-aware KV cache tiering** for multimodal
RAG inference: text and image KV blocks are treated as separate populations
with distinct reuse statistics and storage costs, and routed across GPU /
pinned-CPU / SSD tiers accordingly. The KV cache is the headline contribution.

This report covers the **RAG subsystem** that feeds that cache: the
retrieval, query-rewriting, prompt-construction, and evaluation pipeline that
turns an MMDocRAG question into a multimodal prompt for the served Qwen3-Omni
VLM. Concretely, this report describes the existing pipeline architecture
(Section 2) and the four contributions I added in
`feat(rag): hybrid retrieval`, `feat(rag): query rewriter`,
`feat(rag): pipeline wiring`, and `docs(rag): notes` (Sections 4–7).

The contributions are deliberately **isolated to `rag/`**: the KV cache and
vLLM serving see byte-identical interfaces. All new behavior is gated by
config flags that default to the prior pipeline, so baseline measurements
remain a valid reference.

---

## 2. Existing RAG Pipeline (Background)

### 2.1 Data flow

```
example (MMDocRAG row)
  │   { question, text_quotes[], img_quotes[], answer, gold_quotes }
  ▼
QuoteRetriever.retrieve            ← rag/retriever.py
  │   top-k text quotes (SBERT dense)
  │   top-k image quotes (CLIP text→image, cosine similarity)
  ▼
image_prune_cache lookup           ← rag/query_pipeline.py
  │   reuse pruned image if similar tag exists (≥0.85 cosine)
  ▼
RetrievalPruner.apply              ← rag/pruner.py
  │   token reduction strategies: catp / uniform / patch / visual-only
  ▼
build_prompt                       ← rag/prompt_builder.py
  │   text+image evidence with [text_i] / [image_j] citations
  ▼
vLLM (OpenAI-compatible)           ← cfg.vlm_api_base
  │   streamed completion; first-token-time measured
  ▼
metrics: EM / F1 / retrieval_recall / LLM-judge / TTFT
```

### 2.2 Dataset characteristics

MMDocRAG provides **per-example candidate pools**: each row already includes
the relevant `text_quotes` and `img_quotes`. Retrieval is therefore
**ranking within candidates**, not corpus-wide search. This shapes the design
space:

- BM25 / cross-encoder reranking is cheap (small candidate set per query).
- Recall is bounded by candidate-pool quality, not the embedding model.
- Improvements in retrieval *precision* translate directly into shorter
  prompts (smaller k feasible) and therefore lower TTFT, which is the metric
  the KV-tiering benchmark consumes.

### 2.3 Embedding stack (pre-existing)

| Modality | Model | Source |
|---|---|---|
| Text | `sentence-transformers/all-MiniLM-L6-v2` | HF Hub |
| Image (CLIP text+image branches) | `openai/clip-vit-base-patch32` | HF Hub |

Cosine similarity over L2-normalized embeddings; no FAISS index is needed at
this scale (per-example pools, ≤ 20 candidates).

---

## 3. RAG Problem Statement

Two weaknesses in the original retriever motivated the work:

1. **No lexical signal.** SBERT/CLIP embeddings under-weight exact token
   matches on entity-, number-, and acronym-heavy queries common in MMDocRAG
   (regulatory filings, financial documents). Dense retrieval can score a
   semantically-related but factually-wrong quote above an exact match.
2. **Single query formulation.** A short or under-specified question yields
   one embedding; ambiguous queries produce mediocre rankings with no second
   chance. Standard practice in modern RAG stacks is multi-query expansion.

Constraint: changes must not destabilize the KV cache / pruner / vLLM
integration owned by the rest of the team. The cleanest answer is to add
**opt-in** features inside `QuoteRetriever` and a sibling rewriter module,
with default flags that reproduce the prior pipeline exactly.

---

## 4. Contribution 1 — Hybrid Retrieval (BM25 + Dense via RRF)

**Commit:** `feat(rag): add BM25 hybrid retrieval with RRF fusion`

### 4.1 Design

The text side of `QuoteRetriever.retrieve` is extended with a second ranker
(BM25 over the candidate pool's text fields) and the two rank-lists are fused
with **Reciprocal Rank Fusion**:

$$
\mathrm{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \mathrm{rank}_r(d) + 1}
$$

with $k = 60$ by default (the value originally proposed by Cormack et al.,
empirically robust across IR collections).

RRF was chosen over weighted score-sum for two reasons specific to this
project:

- BM25 and cosine similarity live on **incomparable scales**; score
  normalization requires per-query tuning that hurts robustness.
- RRF only consumes ranks, so the fusion code is independent of the
  underlying scorers — important for keeping the same fusion module reusable
  by the multi-variant path in Contribution 3.

BM25 is text-only: the image side keeps CLIP dense ranking. BM25 over image
captions (`img_quotes[i].img_description` where available) is a viable
future extension but was scoped out for this PR — MMDocRAG image descriptions
are short and noisy.

### 4.2 Implementation

| Symbol | Where | Purpose |
|---|---|---|
| `_bm25_tokenize` | `rag/retriever.py` | Lowercase + Unicode word split (no NLTK dep). |
| `_rrf_fuse` | `rag/retriever.py` | Pure-Python rank fusion, accepts weights. |
| `QuoteRetriever._bm25_rank` | `rag/retriever.py` | Constructs `BM25Okapi` per query (per-example corpus is tiny). |
| `retrieval_mode` / `bm25_weight` / `rrf_k` | `rag/config.py` | Flags. |

`rank_bm25` (pure-Python, no torch/CUDA cost) was added to a new
`requirements.txt` since the project had none.

### 4.3 Backward compatibility

When `retrieval_mode == "dense"` (default), the code path collapses to the
legacy top-k call. When `retrieval_mode == "hybrid"` and the candidate
corpus is empty or BM25 returns no ranking, the implementation falls back to
dense-only — never errors.

---

## 5. Contribution 2 — Query Rewriting Module

**Commit:** `feat(rag): query rewriting module with rule-based + LLM strategies`

### 5.1 Design

A new module `rag/query_rewriter.py` exposes a `QueryRewriter` `Protocol`
with three implementations:

| Implementation | Cost | Determinism | Use case |
|---|---|---|---|
| `NoopRewriter` | 0 | yes | Default; preserves single-query behavior. |
| `RuleBasedRewriter` | μs | yes | Strip stopwords → "keyword variant" alongside the original. Helps BM25 in hybrid mode by removing lexical noise. |
| `LLMRewriter` | one short generation per *unique* query, cached | no | Paraphrase via the existing vLLM endpoint; one-shot, no chain-of-thought, JSON-only output. |

The contract is simple: `rewrite(query: str) -> list[str]`. The **original
query is always first** so downstream RRF fusion can treat it as the
principal ranking signal — and so the existing image-conditioning tag
(`_image_conditioned_clip_tag`) and `image_prune_cache` keys remain stable
across rewriter modes.

### 5.2 LLM rewriter — robustness

The LLM path is the only one that touches the network. Three properties were
prioritized:

1. **Never destabilize the pipeline.** Any exception from the OpenAI client
   collapses the variant list to the original query. The pipeline degrades
   to single-query retrieval rather than failing.
2. **Idempotence across runs.** Variants are cached on disk by SHA-256 prefix
   of the question. The JSON layout mirrors the existing
   `image_prune_cache.json` written by `RAGPipeline` so the team can reuse
   tooling and operational practices.
3. **Tolerant parsing.** The model is asked to return a JSON array; the
   parser also strips ```` ```json ... ``` ```` code fences and gracefully
   handles malformed output by falling back to the original.

### 5.3 Rule-based variant — implementation note

The keyword variant reuses `metrics.normalize_text` (same regex, same
stopword handling logic) via a lazy import. The lazy import is necessary
because `rag/metrics.py` top-level-imports the OpenAI client; without it,
unit tests that exercise the rewriter cannot run in a clean environment.

Stopword list is a compact inline frozenset (~60 tokens). NLTK would have
been overkill for the size of MMDocRAG queries.

### 5.4 Factory

`build_rewriter(cfg)` reads `cfg.query_rewrite_mode` and constructs the right
implementation. Returns `NoopRewriter` by default, so simply importing the
module changes no behavior.

---

## 6. Contribution 3 — Pipeline Wiring (Multi-Variant Retrieval)

**Commit:** `feat(rag): wire multi-variant retrieval through pipeline`

### 6.1 Retriever signature

```python
QuoteRetriever.retrieve(
    example,
    text_top_k=4,
    image_top_k=2,
    variants: list[str] | None = None,   # new
) -> dict
```

When `variants` is `None` or single-element and `retrieval_mode == "dense"`,
the legacy top-k fast path runs — identical results bit-for-bit. Otherwise:

- **Text side.** For each variant: compute the SBERT embedding and a full
  dense rank list; if hybrid, also compute a BM25 rank list per variant.
  Fuse all rank lists with RRF, weighting dense rankings at 1.0 and BM25 at
  `bm25_weight`.
- **Image side.** Encode every variant with CLIP text once (single batched
  forward pass), produce a per-variant rank over the candidate images, fuse
  via RRF. The **principal (first) variant** drives the per-image
  conditioning tag fed back into `image_prune_cache`, which keeps cache keys
  stable across rewriter configurations.

### 6.2 Pipeline integration

`RAGPipeline.__init__` now calls `build_rewriter(cfg)` once. `run_one`
invokes `self.query_rewriter.rewrite(question)` before retrieval and passes
the variants through. The chosen variants are added to the per-example
result dict under `query_variants` for offline analysis (e.g. measuring how
often the rule-based variant differed from the original, or computing
retrieval recall conditioned on which variant won the RRF tiebreak).

### 6.3 Downstream contract

The shape of the dict returned by `QuoteRetriever.retrieve` is unchanged:
`selected_text_quotes` and `selected_img_quotes` with the same per-item
fields. The pruner, prompt builder, and KV cache all observe the same
downstream interface, satisfying the project-level invariant that the new
work must not destabilize the core feature.

---

## 7. Contribution 4 — Documentation

**Commit:** `docs(rag): notes on hybrid retrieval and query rewriting`

Single-page operator reference (`doc/rag-hybrid-and-rewriting.md`): flag
matrix, command-line snippets for the four configurations
(`dense`, `hybrid`, `hybrid + rule_based`, `hybrid + llm`), expected effects
on recall and TTFT, and an index of the touched files. Intended for the next
person running the benchmark, not as a deep design document — the report you
are reading now is that.

---

## 8. Configuration Reference

```python
# rag/config.py — additions to RAGConfig

# Hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion)
retrieval_mode: str = "dense"          # "dense" | "hybrid"
bm25_weight: float = 1.0
rrf_k: int = 60

# Query rewriting (multi-query retrieval)
query_rewrite_mode: str = "none"       # "none" | "rule_based" | "llm"
query_rewrite_max_variants: int = 3
query_rewrite_cache_path: Path = Path("data/mmdocrag/outputs/query_rewrite_cache.json")
query_rewrite_api_base: str | None = None    # falls back to vlm_api_base
query_rewrite_model_name: str | None = None  # falls back to vlm_model_name
```

Defaults reproduce the pre-existing baseline.

---

## 9. Evaluation Plan

### 9.1 Metrics already in the project

- **Retrieval recall:** `metrics.retrieval_recall(pred_ids, gold_ids)`
  computes intersection-over-gold; the principal retrieval-quality metric.
- **Answer correctness:** lexical EM / token-F1, plus a Qwen-as-judge
  semantic score.
- **Latency:** per-query `retrieval_sec`, `request_build_sec`, `ttft_sec`,
  `generation_sec`, `total_sec` (already emitted in result dicts).

### 9.2 Proposed report comparisons

Four runs over an MMDocRAG slice (e.g. first 50 rows):

| Run | `retrieval_mode` | `query_rewrite_mode` |
|---|---|---|
| A — baseline | `dense` | `none` |
| B — hybrid | `hybrid` | `none` |
| C — hybrid + rule | `hybrid` | `rule_based` |
| D — hybrid + LLM | `hybrid` | `llm` |

Report **retrieval_recall@k**, **token-F1**, **judge score**, **avg TTFT**,
and **avg total_sec** for each. The hypotheses:

- B improves recall on entity/number queries vs. A.
- C improves recall further with negligible CPU cost.
- D is the upper bound on recall but pays an extra short generation per
  unique question (amortized to ~0 across repeated runs via the cache).
- Prompt length and KV cache behavior are unchanged across runs (the top-k
  caps still apply), so TTFT is dominated by serving-side effects.

### 9.3 Repro

```
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_baseline.py \
  --eval-slice-start 0 --max-examples 50
```

Toggle `retrieval_mode` / `query_rewrite_mode` in `rag/config.py` (or wire a
CLI flag through `scripts/run_mmdocrag_baseline.py` as a follow-up) between
runs. The `query_rewrite_cache.json` populated by run D should be committed
or archived alongside results for reproducibility.

### 9.4 Tests

Pure-Python unit tests under `tests/`:

- `tests/test_rrf_fusion.py` — 5 tests for `_bm25_tokenize`, `_rrf_fuse`
  (consensus ranking, weighted skew, zero-weight disable, empty input).
- `tests/test_query_rewriter.py` — 9 tests covering `NoopRewriter`,
  `RuleBasedRewriter` (keyword extraction, max-variant cap, redundancy
  pruning), `LLMRewriter` (JSON parsing, code-fence stripping, on-disk
  cache hit/miss, malformed-output fallback), and the `build_rewriter`
  factory dispatch.

No GPU, no model downloads, no network — runnable in CI.

---

## 10. Limitations

- **No cross-encoder reranker.** The plan considered a `bge-reranker-base`
  pass after RRF; scoped out for this milestone to keep the latency profile
  unchanged for the cache benchmark.
- **BM25 only on text.** Image captions could feed a second BM25 ranker on
  the image side; deferred pending an audit of `img_description` quality
  in MMDocRAG.
- **LLM rewriter prompt is minimal.** No few-shot, no chain-of-thought, no
  per-domain instructions. A more elaborate prompt might raise recall further
  on the LLM path at the cost of generation tokens.
- **Single-tenant cache file.** `query_rewrite_cache.json` is a JSON document
  loaded into memory at init; this is fine at MMDocRAG scale (≤ few
  thousand unique queries) but would need a sharded format for production
  workloads. Same shape as the project's existing `image_prune_cache.json`.

---

## 11. Future Work

1. Wire `retrieval_mode` and `query_rewrite_mode` through CLI flags in
   `scripts/run_mmdocrag_baseline.py` so ablations are one-command runs.
2. Add a cross-encoder reranker between RRF fusion and the pruner; the
   smaller post-rerank top-k should shrink prompts and feed more uniform
   token distributions into the KV cache.
3. Add a retrieval-only evaluation harness (`rag/eval_retrieval.py`) that
   reports MRR / NDCG / Hit@k separately from generation, so the RAG layer
   can be optimized without spinning up the VLM.
4. Per-modality query rewriting: separate prompts for "what kind of text
   evidence is relevant?" vs. "what kind of figure should we look for?",
   feeding different variants into the text-side and image-side rankers.

---

## 12. Commit Summary

All commits authored by `haochentSC` on `build/LMCache-based-mrag`:

| SHA | Date | Subject |
|---|---|---|
| `469c51d` | 2026-05-12 15:00 PT | feat(rag): add BM25 hybrid retrieval with RRF fusion |
| `c9af0eb` | 2026-05-12 17:30 PT | feat(rag): query rewriting module with rule-based + LLM strategies |
| `78435ad` | 2026-05-13 10:00 PT | feat(rag): wire multi-variant retrieval through pipeline |
| `7342119` | 2026-05-13 12:30 PT | docs(rag): notes on hybrid retrieval and query rewriting |

Total: 4 commits, ~628 lines added, 23 lines removed across
`rag/retriever.py`, `rag/query_rewriter.py` (new), `rag/query_pipeline.py`,
`rag/config.py`, `tests/test_rrf_fusion.py` (new), `tests/test_query_rewriter.py` (new),
`requirements.txt` (new), `doc/rag-hybrid-and-rewriting.md` (new).

Zero lines changed in `src/cache/`, `config.yaml`, `rag/pruner.py`,
`rag/qwen2vl_catp_pruner_v2.py`, `rag/prompt_builder.py`, or any vLLM
serving code.
