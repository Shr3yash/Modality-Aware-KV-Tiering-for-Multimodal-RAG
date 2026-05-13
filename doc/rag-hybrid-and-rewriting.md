# RAG: Hybrid Retrieval + Query Rewriting

Two opt-in additions to the RAG layer, isolated to `rag/`. The KV cache, vLLM
serving, pruner, and prompt builder are untouched.

## What it does

**Hybrid retrieval.** `QuoteRetriever` now optionally fuses BM25 lexical ranks
with the existing SBERT dense ranks via Reciprocal Rank Fusion (RRF). Helps on
entity- and number-heavy questions where dense embeddings under-weight exact
token matches.

**Query rewriting.** A small `QueryRewriter` runs once per question and emits
up to N variants. The retriever scores every variant and fuses the per-variant
rankings (still via RRF), so a single rewriter knob trades recall for compute.

The principal (first) variant continues to drive the image-conditioning tag, so
the existing `image_prune_cache` keys stay stable across rewriter modes.

## Configuration (`rag/config.py` -> `RAGConfig`)

| Field | Default | Effect |
|---|---|---|
| `retrieval_mode` | `"dense"` | `"hybrid"` to enable BM25 + dense RRF fusion (text side). |
| `bm25_weight` | `1.0` | RRF contribution weight for BM25 rank lists. |
| `rrf_k` | `60` | RRF smoothing constant; higher = flatter score curve. |
| `query_rewrite_mode` | `"none"` | `"rule_based"` (deterministic keyword variant) or `"llm"` (paraphrases via the vLLM endpoint, cached on disk). |
| `query_rewrite_max_variants` | `3` | Cap on variants per query, original included. |
| `query_rewrite_cache_path` | `data/mmdocrag/outputs/query_rewrite_cache.json` | JSON cache for LLM rewrites (mirrors `image_prune_cache` layout). |
| `query_rewrite_api_base` / `query_rewrite_model_name` | `None` | Optional override; falls back to `vlm_api_base` / `vlm_model_name`. |

All defaults preserve prior behavior; the existing baseline run is byte-identical
with the flags untouched.

## How to run

Baseline (unchanged):

```
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_baseline.py \
  --eval-slice-start 0 --max-examples 15
```

Hybrid retrieval only -- edit `RAGConfig` defaults or wire a CLI flag through
`scripts/run_mmdocrag_baseline.py`:

```
retrieval_mode = "hybrid"
```

Hybrid + rule-based rewriting (no extra network calls):

```
retrieval_mode = "hybrid"
query_rewrite_mode = "rule_based"
```

Hybrid + LLM rewriting (requires the vLLM server to be up; first run populates
the on-disk cache, subsequent runs are fast):

```
retrieval_mode = "hybrid"
query_rewrite_mode = "llm"
```

## Expected effects

- Retrieval recall: `>=` baseline on the MMDocRAG evaluation slice. Largest
  lift is expected on factoid/entity queries where BM25 surfaces an exact
  match that dense embeddings rank lower.
- Prompt length: unchanged (top-k caps still apply); the cache and pruner see
  the same downstream shape.
- TTFT: rule-based rewriting adds negligible CPU work; LLM rewriting adds one
  short generation per *new* question -- amortized to ~0 across repeated runs
  thanks to the disk cache.

## Files

- `rag/retriever.py` -- BM25 helper, RRF helper, multi-variant `retrieve(...)`.
- `rag/query_rewriter.py` -- `QueryRewriter` protocol; `Noop`, `RuleBased`,
  `LLM` rewriters; `build_rewriter(cfg)` factory.
- `rag/query_pipeline.py` -- builds rewriter, passes variants into the
  retriever, surfaces `query_variants` per example.
- `rag/config.py` -- the fields above.
- `requirements.txt` -- adds `rank_bm25`.
- `tests/test_rrf_fusion.py`, `tests/test_query_rewriter.py` -- pure-Python
  unit tests (no GPU / model downloads required).
