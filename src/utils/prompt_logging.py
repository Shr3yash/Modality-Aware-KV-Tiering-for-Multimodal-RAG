from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
from transformers import AutoTokenizer


class SegmentTokenCount(BaseModel):
    role: str
    modality: str
    source: str
    token_count: int
    text_length: int
    retrieval_rank: int | None = None


class PromptSummary(BaseModel):
    total_prompt_tokens: int
    retrieved_text_token_count: int
    visual_input_present: bool
    num_retrieved_chunks: int
    ordered_segment_roles: list[str] = Field(default_factory=list)
    ordered_modalities: list[str] = Field(default_factory=list)
    token_count_by_role: dict[str, int] = Field(default_factory=dict)
    token_count_by_modality: dict[str, int] = Field(default_factory=dict)
    segment_token_counts: list[SegmentTokenCount] = Field(default_factory=list)
    generation_latency_seconds: float | None = None


def summarize_prompt(
    *,
    segments: list[Any] | None = None,
    prompt_segments: list[Any] | None = None,
    token_counter: Callable[[str], int] | None = None,
    config: Any | None = None,
    tokenizer: Any | None = None,
    timing_summary: Mapping[str, Any] | None = None,
    generation_timing: Mapping[str, Any] | None = None,
) -> PromptSummary:
    normalized_segments = [_normalize_segment(segment) for segment in (segments or prompt_segments or [])]
    effective_token_counter = _resolve_token_counter(
        token_counter=token_counter,
        tokenizer=tokenizer,
        config=config,
    )

    segment_token_counts: list[SegmentTokenCount] = []
    token_count_by_role: Counter[str] = Counter()
    token_count_by_modality: Counter[str] = Counter()
    total_prompt_tokens = 0
    retrieved_text_token_count = 0
    visual_input_present = False
    ordered_segment_roles: list[str] = []
    ordered_modalities: list[str] = []
    num_retrieved_chunks = 0

    for segment in normalized_segments:
        token_count = effective_token_counter(segment["text"])
        segment_token_counts.append(
            SegmentTokenCount(
                role=segment["role"],
                modality=segment["modality"],
                source=segment["source"],
                token_count=token_count,
                text_length=segment["text_length"],
                retrieval_rank=segment["retrieval_rank"],
            )
        )
        total_prompt_tokens += token_count
        token_count_by_role[segment["role"]] += token_count
        token_count_by_modality[segment["modality"]] += token_count
        ordered_segment_roles.append(segment["role"])
        ordered_modalities.append(segment["modality"])
        if segment["role"].startswith("retrieved_"):
            num_retrieved_chunks += 1
        if segment["role"] == "retrieved_text":
            retrieved_text_token_count += token_count
        if segment["modality"] == "image" or segment["role"] == "visual_context":
            visual_input_present = True

    return PromptSummary(
        total_prompt_tokens=total_prompt_tokens,
        retrieved_text_token_count=retrieved_text_token_count,
        visual_input_present=visual_input_present,
        num_retrieved_chunks=num_retrieved_chunks,
        ordered_segment_roles=ordered_segment_roles,
        ordered_modalities=ordered_modalities,
        token_count_by_role=dict(token_count_by_role),
        token_count_by_modality=dict(token_count_by_modality),
        segment_token_counts=segment_token_counts,
        generation_latency_seconds=_extract_generation_latency(
            timing_summary=timing_summary,
            generation_timing=generation_timing,
        ),
    )


summarize_prompt_segments = summarize_prompt


def log_prompt_summary(
    logger: Any,
    summary: PromptSummary | Mapping[str, Any],
    **context: Any,
) -> None:
    if logger is None or not hasattr(logger, "info"):
        return

    payload = summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)
    logger.info("prompt_summary", **context, **payload)


def _normalize_segment(segment: Any) -> dict[str, Any]:
    if isinstance(segment, Mapping):
        mapping = dict(segment)
    elif hasattr(segment, "model_dump"):
        mapping = segment.model_dump()
    elif hasattr(segment, "__dict__"):
        mapping = dict(vars(segment))
    else:
        raise TypeError(f"Could not normalize prompt segment {segment!r}.")

    text = mapping.get("text")
    text_str = "" if text is None else str(text)
    text_length = mapping.get("text_length")
    if not isinstance(text_length, int):
        text_length = len(text_str)

    return {
        "role": str(mapping.get("role", "")),
        "modality": str(mapping.get("modality", "text")),
        "source": str(mapping.get("source", "")),
        "text": text_str,
        "text_length": text_length,
        "retrieval_rank": mapping.get("retrieval_rank"),
    }


def _resolve_token_counter(
    *,
    token_counter: Callable[[str], int] | None,
    tokenizer: Any | None,
    config: Any | None,
) -> Callable[[str], int]:
    if token_counter is not None:
        return token_counter

    effective_tokenizer = tokenizer or _load_tokenizer_from_config(config)
    if effective_tokenizer is None:
        return _whitespace_token_count

    def _count_with_tokenizer(text: str) -> int:
        if not text.strip():
            return 0
        try:
            token_ids = effective_tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = effective_tokenizer.encode(text)
        return len(token_ids)

    return _count_with_tokenizer


def _whitespace_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _load_tokenizer_from_config(config: Any | None) -> Any | None:
    tokenizer_name = _resolve_tokenizer_name(config)
    if not tokenizer_name:
        return None
    return _cached_tokenizer(tokenizer_name)


def _resolve_tokenizer_name(config: Any | None) -> str | None:
    if config is None:
        return None

    explicit_name = getattr(config, "tokenizer_name", None)
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name

    model_config = getattr(config, "model", None)
    model_name = getattr(model_config, "name", None)
    if isinstance(model_name, str) and model_name:
        return model_name

    rag_config = getattr(config, "rag", None)
    embedding_name = getattr(rag_config, "embedding_model", None) or getattr(config, "embedding_model", None)
    if isinstance(embedding_name, str) and embedding_name:
        return embedding_name
    return None


@lru_cache(maxsize=4)
def _cached_tokenizer(tokenizer_name: str) -> Any | None:
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        return None


def _extract_generation_latency(
    *,
    timing_summary: Mapping[str, Any] | None,
    generation_timing: Mapping[str, Any] | None,
) -> float | None:
    for candidate in (timing_summary, generation_timing):
        if candidate is None:
            continue
        for key in ("generation_seconds", "generation_latency_seconds"):
            value = candidate.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None
