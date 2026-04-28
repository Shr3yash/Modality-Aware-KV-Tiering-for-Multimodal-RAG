from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any

from src.rag import (
    GenerateRequest,
    GenerateResponse,
    PromptSegment,
    RetrievedChunk,
    Retriever,
    build_prompt,
    load_corpus,
)
from src.rag.prompt_builder import DEFAULT_SYSTEM_INSTRUCTION
from src.serving.model_runner import ModelRunner
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.prompt_logging import log_prompt_summary, summarize_prompt


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class GenerationService:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        prompt_builder: Any | None = None,
        model_runner: Any | None = None,
        prompt_logger: Any | None = None,
        token_logger: Any | None = None,
        top_k: int | None = None,
        config: Any | None = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        logger: Any | None = None,
    ) -> None:
        self.config = config or _load_default_config()
        self.system_instruction = system_instruction
        self.top_k = int(top_k if top_k is not None else getattr(getattr(self.config, "rag", None), "top_k", 5))
        self.retriever = retriever
        self.prompt_builder = prompt_builder or build_prompt
        self.model_runner = model_runner
        self.prompt_logger = prompt_logger or summarize_prompt
        self.token_logger = token_logger or self.prompt_logger
        self.logger = logger or get_logger(__name__)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        effective_top_k = request.top_k if request.top_k is not None else self.top_k
        retrieved_chunks = self._retrieve_chunks(query=request.query, top_k=effective_top_k)

        prompt_result = self._call_prompt_builder(
            request=request,
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            system_instruction=self.system_instruction,
            image_path=request.image_path,
        )
        segments = _normalize_prompt_segments(prompt_result["segments"])
        prompt_payload = prompt_result["payload"]
        segment_metadata = dict(prompt_result.get("segment_metadata") or {})

        runner_response = self._get_model_runner().generate(request_payload=prompt_payload)
        answer_text = _get_value(runner_response, "generated_text", "text", "answer_text", default="")
        timing_summary = _get_value(runner_response, "timing", "timing_summary", default={})
        token_summary = self._summarize_prompt(segments, timing_summary=timing_summary)
        prompt_summary = {**token_summary, **segment_metadata}

        response = GenerateResponse(
            answer_text=str(answer_text),
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
            prompt_summary=prompt_summary,
            token_summary=token_summary,
            timing_summary=dict(timing_summary) if isinstance(timing_summary, dict) else {},
            segments=segments,
        )
        self._log_response(request=request, response=response)
        return response

    handle_generate = generate
    run = generate

    def _retrieve_chunks(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        retriever = self._get_retriever()
        if hasattr(retriever, "retrieve"):
            raw_chunks = retriever.retrieve(query, top_k)
        else:
            raw_chunks = retriever(query, top_k)
        return [_normalize_retrieved_chunk(chunk) for chunk in raw_chunks]

    def _call_prompt_builder(self, **kwargs: Any) -> dict[str, Any]:
        builder = self.prompt_builder
        if hasattr(builder, "build_prompt"):
            result = builder.build_prompt(**kwargs)
        else:
            result = builder(**kwargs)

        if isinstance(result, tuple) and len(result) == 3:
            segments, payload, segment_metadata = result
            return {
                "segments": segments,
                "payload": payload,
                "segment_metadata": segment_metadata,
            }

        mapping = _to_mapping(result)
        return {
            "segments": mapping.get("segments", []),
            "payload": mapping.get("payload", mapping.get("request_payload")),
            "segment_metadata": mapping.get("segment_metadata", mapping.get("metadata", {})),
        }

    def _summarize_prompt(
        self,
        segments: list[PromptSegment],
        *,
        timing_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary_helper = self.token_logger or self.prompt_logger
        if summary_helper is None:
            return _fallback_prompt_summary(segments)

        try:
            call_kwargs = {
                "segments": segments,
                "prompt_segments": segments,
                "config": self.config,
                "timing_summary": timing_summary,
                "generation_timing": timing_summary,
            }
            if hasattr(summary_helper, "summarize_prompt"):
                result = _call_with_supported_args(summary_helper.summarize_prompt, **call_kwargs)
            else:
                result = _call_with_supported_args(summary_helper, **call_kwargs)
        except Exception:
            return _fallback_prompt_summary(segments)

        try:
            return dict(_to_mapping(result))
        except Exception:
            return _fallback_prompt_summary(segments)

    def _get_retriever(self) -> Any:
        if self.retriever is None:
            rag_config = getattr(self.config, "rag", None)
            chunks = load_corpus(config=rag_config)
            self.retriever = Retriever(corpus_chunks=chunks, config=rag_config)
        return self.retriever

    def _get_model_runner(self) -> Any:
        if self.model_runner is None:
            self.model_runner = ModelRunner(model_config=getattr(self.config, "model", None))
        return self.model_runner

    def _log_response(self, *, request: GenerateRequest, response: GenerateResponse) -> None:
        if self.logger is None or not hasattr(self.logger, "info"):
            return
        log_prompt_summary(
            self.logger,
            response.token_summary,
            query=request.query,
            retrieved_chunk_ids=response.retrieved_chunk_ids,
            image_path=request.image_path,
        )
        self.logger.info(
            "generation_completed",
            query=request.query,
            retrieved_chunk_ids=response.retrieved_chunk_ids,
            prompt_summary=response.prompt_summary,
            token_summary=response.token_summary,
            timing_summary=response.timing_summary,
        )


RAGService = GenerationService
Service = GenerationService


def _load_default_config() -> Any | None:
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    return None


def _load_prompt_summary_helper() -> Any | None:
    return summarize_prompt


def _normalize_retrieved_chunk(chunk: Any) -> RetrievedChunk:
    if isinstance(chunk, RetrievedChunk):
        return chunk

    mapping = _to_mapping(chunk)
    metadata = dict(mapping.get("metadata") or {})
    if "score" not in mapping and "score" in metadata:
        mapping["score"] = metadata.get("score")
    return RetrievedChunk.model_validate(mapping)


def _normalize_prompt_segments(segments: list[Any]) -> list[PromptSegment]:
    normalized: list[PromptSegment] = []
    for segment in segments:
        if isinstance(segment, PromptSegment):
            normalized.append(segment)
        else:
            normalized.append(PromptSegment.model_validate(_to_mapping(segment)))
    return normalized


def _fallback_prompt_summary(segments: list[PromptSegment]) -> dict[str, Any]:
    role_order = [segment.role for segment in segments]
    retrieved_segments = [segment for segment in segments if segment.role.startswith("retrieved_")]
    retrieved_text_segments = [segment for segment in retrieved_segments if segment.role == "retrieved_text"]
    total_prompt_tokens = sum(_estimate_text_tokens(segment.text or "") for segment in segments)
    retrieved_tokens = sum(_estimate_text_tokens(segment.text or "") for segment in retrieved_text_segments)
    visual_input_present = any(segment.modality == "image" or segment.role == "visual_context" for segment in segments)
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "retrieved_text_token_count": retrieved_tokens,
        "visual_input_present": visual_input_present,
        "num_retrieved_chunks": len(retrieved_segments),
        "ordered_segment_roles": role_order,
    }


def _call_with_supported_args(func: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return func(**supported_kwargs)


def _estimate_text_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Could not coerce {value!r} into a mapping.")


_MISSING = object()


def _get_value(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"Expected one of {names!r} on {value!r}.")
