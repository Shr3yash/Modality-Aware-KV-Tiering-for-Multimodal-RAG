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

- `Qwen/Qwen2.5-Omni-3B`
- `llava-hf/llava-v1.6-mistral-7b-hf` (Not for now)

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

## Implemented code walkthrough by step

This section explains the global variables, top-level functions, aliases, and
main types that were implemented for Phase 2. The goal is to make each step
easy to read directly from the codebase.

### Step 1 code walkthrough

Files:

- `configs/default.yaml`
- `src/utils/config.py`

`configs/default.yaml`

- `model.name`: freezes the Phase 2 default model path to `Qwen/Qwen2.5-Omni-3B`.
- `model.dtype`, `model.max_model_len`, `model.gpu_memory_utilization`: default
  runtime settings later consumed by the model runner.
- `rag.*`: default corpus, chunking, retrieval, and embedding settings.
- `cache.*`: Phase 1 cache tiering parameters reused by later phases.
- `serving.*`: default host, port, and concurrency settings for the API layer.
- `eval.*`: evaluation defaults kept in typed config for later benchmarks.

`src/utils/config.py`

- `_gpu_device()`: picks the default execution device in priority order `mps`,
  then `cuda`, then `cpu`.
- `ModelConfig`: Pydantic model for model-serving settings.
- `RagConfig`: Pydantic model for corpus, chunking, top-k, and embedding
  settings.
- `CacheConfig`: Pydantic model for the GPU/CPU/SSD cache tier parameters.
- `ServingConfig`: Pydantic model for API server settings.
- `EvalConfig`: Pydantic model for later evaluation settings.
- `Config`: top-level Pydantic container that groups all config sections into
  one typed object.
- `load_config(path)`: reads a YAML file and returns a fully typed `Config`
  instance with defaults filled in for any missing sections.

### Step 2 code walkthrough

Files:

- `src/rag/corpus.py`
- `src/rag/retriever.py`

`src/rag/corpus.py`

- `DEFAULT_VIDORE_DATASET`: default Hugging Face dataset name for optional
  corpus download.
- `DEFAULT_VIDORE_CONFIG`: default dataset config name for that corpus.
- `DEFAULT_VIDORE_SPLIT`: default split name for that corpus.
- `_HF_DATASET_MARKERS`: file names used to detect whether a directory already
  looks like a saved Hugging Face dataset.
- `CorpusChunk`: dataclass representing one text chunk plus source metadata and
  token ids.
- `FallbackTokenizer`: small offline tokenizer used when a pretrained tokenizer
  cannot be loaded.
- `download_vidore_corpus(...)`: downloads the default ViDoRe corpus and saves
  it locally with `datasets.save_to_disk`.
- `load_corpus(...)`: main Step 2 entrypoint; loads either a text directory or a
  saved Hugging Face dataset, chunks the text, and returns `CorpusChunk`
  objects.
- `_resolve_tokenizer(...)`: chooses an injected tokenizer, a pretrained
  tokenizer, or the fallback tokenizer.
- `_looks_like_hf_dataset_dir(path)`: detects whether a directory is a saved
  Hugging Face dataset.
- `_iter_dataset_records(dataset)`: flattens a `Dataset` or `DatasetDict` into
  plain Python records.
- `_iter_text_records(corpus_path)`: reads `.txt` files from the corpus
  directory into record dictionaries.
- `_extract_record_text(record)`: pulls the first meaningful text field from a
  record.
- `_extract_record_source(record)`: resolves a stable human-readable source
  string for a record.
- `_extract_doc_id(record)`: extracts the document id from common field names.
- `_extract_page_number(record)`: extracts page number metadata when present.
- `_extract_metadata(record)`: keeps non-core record fields as metadata on the
  resulting chunk.
- `_chunk_text(...)`: tokenizes a document and slices it into chunk-sized text
  spans.
- `_build_chunk_id(source, chunk_index, token_ids)`: creates a stable hashed id
  for each chunk.

`src/rag/retriever.py`

- `OpenCLIPTextEmbedder`: lazy embedding wrapper around `open_clip` text
  encoders.
- `Retriever`: in-memory FAISS retriever that embeds the corpus, builds the
  similarity index, and returns ranked chunks.
- `_OPEN_CLIP_MODEL_ALIASES`: maps friendly or repo model names to the
  `open_clip` naming convention.
- `_resolve_model_spec(model_name)`: converts a configured embedding model name
  into the format expected by `open_clip`.
- `_normalize_embeddings(embeddings)`: L2-normalizes vectors for inner-product
  similarity search.
- `_as_float32_matrix(value)`: coerces embeddings into a 2D `float32` matrix.
- `_as_float32_vector(value)`: coerces a query embedding into a 1D `float32`
  vector.

### Step 3 code walkthrough

File:

- `src/rag/schemas.py`

`src/rag/schemas.py`

- `GenerateRequest`: Pydantic request model for Phase 2 input with `query`,
  optional `image_path`, and optional `top_k`.
- `RetrievedChunk`: Pydantic model for retriever output, including rank, score,
  and free-form metadata.
- `PromptSegment`: Pydantic model for one prompt segment, including its role,
  modality, source, and text statistics.
- `GenerateResponse`: Pydantic response model returned by the service and API,
  including answer text, chunk ids, prompt summary, token summary, timing
  summary, and prompt segments.

### Step 4 code walkthrough

File:

- `src/rag/prompt_builder.py`

`src/rag/prompt_builder.py`

- `DEFAULT_SYSTEM_INSTRUCTION`: default system message used when no custom
  instruction is passed in.
- `build_prompt(...)`: main Step 4 entrypoint; converts a request, retrieved
  chunks, and optional image into ordered `PromptSegment`s, a vLLM-like chat
  payload, and logging metadata.
- `assemble_prompt`: alias for `build_prompt` so downstream code can use either
  name.
- `_normalize_chunks(chunks)`: converts mixed chunk inputs into validated
  `RetrievedChunk` objects and sorts them by rank.
- `_make_text_segment(...)`: helper that creates a text-mode `PromptSegment`
  with consistent metadata.
- `_build_vllm_chat_payload(...)`: creates the model-ready chat payload used by
  the model runner.
- `_build_user_text_block(...)`: formats the user query plus retrieved context
  into the text block placed in the user message.

### Step 5 code walkthrough

File:

- `src/serving/model_runner.py`

`src/serving/model_runner.py`

- `DEFAULT_MAX_TOKENS`: default maximum generation length used by the runner.
- `DEFAULT_TEMPERATURE`: default sampling temperature.
- `DEFAULT_TOP_P`: default nucleus sampling parameter.
- `ModelRunnerResponse`: Pydantic response model returned by the runner with
  generated text, timing, and raw backend output.
- `ModelRunner`: thin wrapper around vLLM engine construction and generation.
  Its important methods are:
  - `generate(...)`: normalizes payloads, runs generation once, and returns a
    `ModelRunnerResponse`.
  - `_get_engine()`: lazily builds and caches a single engine instance.
  - `_default_engine_factory(...)`: creates the default `vllm.LLM(...)`
    instance.
  - `_build_generation_kwargs(...)`: builds default sampling parameters for the
    backend call.
  - `_normalize_request_payload(...)`: accepts both loose prompt payloads and
    chat-style payloads, then normalizes them into one backend-facing format.
  - `_invoke_engine(...)`: decides whether to call `engine.chat(...)` or
    `engine.generate(...)`.
  - `_extract_generated_text(...)`: extracts text from dict-style or vLLM-style
    responses.
  - `_extract_timing(...)`: extracts timing metadata or falls back to measured
    wall-clock time.

### Step 6 code walkthrough

File:

- `src/serving/service.py`

`src/serving/service.py`

- `DEFAULT_CONFIG_PATH`: default path used when the service builds itself from
  the repository config.
- `GenerationService`: main orchestration class that wires retrieval, prompt
  building, generation, prompt summary, and structured logging into one
  in-process flow. Its important methods are:
  - `generate(request)`: end-to-end Phase 2 request handler returning
    `GenerateResponse`.
  - `handle_generate`: alias to `generate`.
  - `run`: alias to `generate`.
  - `_retrieve_chunks(...)`: calls the retriever and normalizes the results.
  - `_call_prompt_builder(...)`: accepts either a builder object or a builder
    function and normalizes its return shape.
  - `_summarize_prompt(...)`: calls the prompt logging helper and falls back to
    a local summary if needed.
  - `_get_retriever()`: lazily constructs the default retriever from config and
    corpus data.
  - `_get_model_runner()`: lazily constructs the default model runner from
    config.
  - `_log_response(...)`: emits the structured prompt summary log and the final
    request-completion log.
- `RAGService`: alias for `GenerationService`.
- `Service`: alias for `GenerationService`.
- `_load_default_config()`: loads `configs/default.yaml` when present.
- `_load_prompt_summary_helper()`: currently returns the shared prompt summary
  helper from `src.utils.prompt_logging`.
- `_normalize_retrieved_chunk(chunk)`: coerces raw retriever results into
  `RetrievedChunk` models.
- `_normalize_prompt_segments(segments)`: coerces raw prompt-builder output into
  `PromptSegment` models.
- `_fallback_prompt_summary(segments)`: small local prompt-summary fallback used
  if the logging helper fails or is unavailable.
- `_call_with_supported_args(func, **kwargs)`: calls a function using only the
  keyword arguments it accepts.
- `_estimate_text_tokens(text)`: simple whitespace token estimate used by the
  fallback summary.
- `_to_mapping(value)`: normalizes dicts, dataclasses, or Pydantic models into
  a Python mapping.
- `_MISSING`: private sentinel used by `_get_value(...)` to detect whether a
  default was supplied.
- `_get_value(value, *names, default=...)`: reads one of several possible field
  names from objects or dicts.

### Step 7 code walkthrough

File:

- `src/utils/prompt_logging.py`

`src/utils/prompt_logging.py`

- `SegmentTokenCount`: Pydantic model for per-segment token statistics.
- `PromptSummary`: Pydantic model for the full prompt/token summary returned by
  the logger.
- `summarize_prompt(...)`: main Step 7 entrypoint; normalizes segments, chooses
  a token counter, computes totals and per-segment counts, and returns a
  `PromptSummary`.
- `summarize_prompt_segments`: alias for `summarize_prompt`.
- `log_prompt_summary(logger, summary, **context)`: emits one structured
  `prompt_summary` log event.
- `_normalize_segment(segment)`: converts dicts or models into one normalized
  segment shape for counting.
- `_resolve_token_counter(...)`: chooses an injected token counter, a tokenizer
  from config, or the whitespace fallback.
- `_whitespace_token_count(text)`: simple token counter used when no tokenizer
  is available.
- `_load_tokenizer_from_config(config)`: loads a tokenizer based on config.
- `_resolve_tokenizer_name(config)`: picks the best tokenizer name from the
  config object.
- `_cached_tokenizer(tokenizer_name)`: memoized tokenizer loader so repeated
  requests do not reload the tokenizer.
- `_extract_generation_latency(...)`: pulls generation latency from the timing
  summary when available.

### Step 8 code walkthrough

File:

- `src/serving/api.py`

`src/serving/api.py`

- `create_app(...)`: main Step 8 entrypoint; builds the FastAPI app, wires the
  default service, and registers the three Phase 2 routes.
- `app`: module-level FastAPI application instance so the demo can be launched
  directly with Uvicorn.
- Route behavior inside `create_app(...)`:
  - `GET /health`: returns `{"status": "ok"}` for readiness checks.
  - `POST /generate`: validates a `GenerateRequest`, forwards it to the service,
    and returns a `GenerateResponse`.
  - `GET /metrics`: exposes the current Prometheus registry with the correct
    Prometheus content type.

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
