# Phase 2 Concrete Plan

This plan assumes the current repository state described in:

- `logs/readme_intro_roadmap_merged.md`
- `logs/src_architecture_block.md`
- `logs/src_data_flow_mermaid.md`

It starts from the completed Phase 1 cache foundation and turns the repo into a
working multimodal RAG demo with a real serving path.

## Phase 2 Goal

Deliver a small but real end-to-end path:

1. accept a text query and optional image input,
2. retrieve top-k text context,
3. assemble a multimodal prompt,
4. run a stable image-text model through vLLM,
5. return a generated answer, and
6. log prompt composition and token counts by modality and segment role.

## Current baseline

Already present:

- typed config in `src/utils/config.py`
- logging, metrics, and profiling helpers in `src/utils`
- KV block identity and tier movement in `src/cache`
- cache tests and cross-tier smoke tests in `tests`

Missing for Phase 2:

- model-serving adapter
- multimodal request schema
- corpus loading and retrieval path
- prompt builder
- demo entrypoint or API route
- prompt/token instrumentation around the multimodal prefill path

## Recommended scope

Keep Phase 2 intentionally small:

- one vLLM-supported image-text model
- one local text corpus
- optional image input from a file path
- one synchronous generation path
- one demo interface, CLI or minimal FastAPI

Do not include yet:

- custom vLLM KV tagging internals
- MAST scoring beyond current cache behavior
- benchmark harness
- prefix reuse optimization
- SSD stress experiments

## Proposed code additions

Add these modules in `src`:

```text
src/rag/
  __init__.py
  corpus.py           # load corpus and chunk plain-text sources
  retriever.py        # embedding + FAISS top-k lookup
  prompt_builder.py   # assemble system/user/retrieved_text/image segments
  schemas.py          # request/response payload models

src/serving/
  __init__.py
  model_runner.py     # thin vLLM wrapper for multimodal generation
  service.py          # orchestrate retrieval -> prompt build -> model call
  api.py              # optional FastAPI endpoints

src/utils/
  prompt_logging.py   # segment-role logging and token-count summaries
```

Add test files:

```text
tests/test_rag_retriever.py
tests/test_prompt_builder.py
tests/test_model_runner.py
tests/test_service_flow.py
```

## Concrete implementation steps

### Step 1 - Freeze the Phase 2 model path

Pick one model path and encode it in config and docs.

Recommended default:

- `Qwen2.5-Omni-3B`
- `llava-hf/llava-v1.6-mistral-7b-hf`

Implementation:

- confirm `model.name` in `configs/default.yaml`
- keep `dtype` and memory settings in `src/utils/config.py`
- document one known-good run command in the README or logs

Exit criteria:

- one model is the explicit Phase 2 default
- one local environment can load it consistently

### Step 2 - Add a minimal RAG data path

Build a tiny local corpus path first, without overengineering.

Implementation:

- `src/rag/corpus.py`
  - read plain-text files from `config.rag.corpus_dir`
  - split into text chunks using `chunk_size_tokens`
  - attach chunk ids and source metadata
- `src/rag/retriever.py`
  - embed chunks with the configured embedding model
  - build a FAISS index in memory
  - return top-k chunks for a query

Exit criteria:

- given a text query, the retriever returns stable top-k text chunks
- the retriever can be exercised independently in tests

### Step 3 - Define multimodal request and response schemas

Introduce one narrow interface before wiring serving.

Implementation:

- `src/rag/schemas.py`
  - `GenerateRequest`
    - `query: str`
    - `image_path: str | None`
    - `top_k: int | None`
  - `RetrievedChunk`
  - `PromptSegment`
  - `GenerateResponse`

Suggested segment roles for Phase 2:

- `system`
- `user`
- `retrieved_text`
- `visual_context`

Exit criteria:

- one request object can represent the full Phase 2 input
- one response object captures answer text and trace metadata

### Step 4 - Build the prompt assembly layer

This is the bridge between retrieval and model execution.

Implementation:

- `src/rag/prompt_builder.py`
  - take system instruction, user query, retrieved chunks, and optional image
  - return:
    - ordered prompt segments
    - a model-ready prompt payload
    - per-segment metadata for logging

Required metadata per segment:

- `role`
- `modality`
- `source`
- `text_length`
- `retrieval_rank` when applicable

Exit criteria:

- prompt assembly is deterministic
- tests verify ordering and role assignment
- image/no-image paths both work

### Step 5 - Add a thin vLLM model runner

Do not mix retrieval logic with model startup.

Implementation:

- `src/serving/model_runner.py`
  - own vLLM initialization
  - expose `generate(request_payload)` for text-only and multimodal inputs
  - keep a narrow abstraction so the rest of the code does not depend on raw vLLM objects

Suggested responsibilities:

- initialize engine once
- convert prompt builder output into the vLLM request format
- return generated text plus timing metadata

Exit criteria:

- one direct call can generate from a text query
- one direct call can generate from a query plus image path

### Step 6 - Create a service orchestrator

This is where the current cache foundation starts to connect to the serving path.

Implementation:

- `src/serving/service.py`
  - receive `GenerateRequest`
  - call retriever
  - call prompt builder
  - call model runner
  - emit structured logs and counters

Service output should include:

- answer text
- retrieved chunk ids
- prompt segment summary
- token-count summary
- timing summary

Exit criteria:

- one in-process service method handles the full flow end to end
- integration tests can mock retriever and model runner separately

### Step 7 - Add prompt and token instrumentation

Phase 2 explicitly requires prompt-composition and token-count logging.

Implementation:

- `src/utils/prompt_logging.py`
  - summarize prompt segments by role and modality
  - estimate or record token counts by segment
  - emit structured logging payloads compatible with `src/utils/logging.py`

Minimum logs per request:

- total prompt tokens
- retrieved-text token count
- visual input present or absent
- number of retrieved chunks
- ordered segment roles

Stretch logging for Phase 2:

- per-segment token counts
- generation latency
- memory snapshot from `src/utils/profiling.py`

Exit criteria:

- every request emits a compact prompt summary
- logs are human-checkable and machine-parseable

### Step 8 - Expose one runnable demo path

Choose one of these as the official Phase 2 entrypoint:

- minimal FastAPI app in `src/serving/api.py`, or
- a CLI demo script under `scripts/`

Recommended default:

- minimal FastAPI with:
  - `GET /health`
  - `POST /generate`
  - `GET /metrics`

Exit criteria:

- one command starts the demo
- one sample request with an image path returns an answer

### Step 9 - Add Phase 2 tests

Focus on stable tests first and isolate model-heavy tests.

Unit tests:

- retriever returns top-k chunks
- prompt builder preserves expected segment order
- token logging summaries are correct
- service orchestration calls the right components

Integration tests:

- mocked model runner end-to-end service flow
- optional smoke test behind an env flag for real vLLM model invocation

Suggested split:

- default CI-safe tests mock the model
- one manual smoke test exercises the real runtime

Exit criteria:

- new unit tests pass without needing the full model
- one documented manual smoke path exists for the real model

## How Phase 2 should connect to the current cache foundation

For Phase 2, the cache system does not need deep vLLM internals yet. The first
connection should be observational and architectural:

- use the existing config and logging stack
- keep cache-manager wiring isolated behind a future adapter
- structure prompt segments so they can later map onto KV block roles
- preserve `modality` and `role` metadata now, even if the current cache layer is not yet fed by live vLLM KV blocks

This keeps Phase 2 compatible with Phase 3, where actual modality-aware block
tagging is introduced.

## Recommended order of execution

Build in this order:

1. `src/rag/schemas.py`
2. `src/rag/corpus.py`
3. `src/rag/retriever.py`
4. `src/rag/prompt_builder.py`
5. `src/serving/model_runner.py`
6. `src/serving/service.py`
7. `src/utils/prompt_logging.py`
8. `src/serving/api.py`
9. tests for each layer

## Definition of done for Phase 2

Phase 2 is complete when all of the following are true:

- a query with optional image input can be submitted through one official entrypoint
- top-k text retrieval is live
- prompt assembly produces explicit role/modality segments
- a vLLM-backed image-text model returns an answer
- prompt composition and token-count logs are emitted per request
- the new unit and service-flow tests pass
- one documented manual demo path works end to end

## What should wait until Phase 3

Do not pull these into Phase 2 unless they become necessary blockers:

- vLLM prefill-time KV block tagging
- role-aware KV metadata inside live engine blocks
- modality-aware split-tiering policy changes
- block-level semantic priority scoring
- reuse-aware routing or scheduling

Those belong in the next phase after the serving path is real.
