from __future__ import annotations

from typing import Any, Iterable

from .schemas import GenerateRequest, PromptSegment, RetrievedChunk


DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful multimodal RAG assistant."


def build_prompt(
    *,
    request: GenerateRequest | None = None,
    query: str | None = None,
    retrieved_chunks: Iterable[RetrievedChunk | dict[str, Any]] | None = None,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    image_path: str | None = None,
    max_input_tokens: int | None = None,
    reserved_output_tokens: int = 256,
) -> dict[str, Any]:
    effective_query = query if query is not None else (request.query if request is not None else "")
    effective_image_path = image_path if image_path is not None else (request.image_path if request is not None else None)
    ordered_chunks = _normalize_chunks(retrieved_chunks)

    segments: list[PromptSegment] = [
        _make_text_segment(
            role="system",
            source="system",
            text=system_instruction,
        ),
        _make_text_segment(
            role="user",
            source="user",
            text=effective_query,
        ),
    ]

    for chunk in ordered_chunks:
        segment_metadata = {
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "image_path": chunk.image_path,
            **chunk.metadata,
        }
        if chunk.modality == "image":
            visual_text = chunk.text or f"retrieved image: {chunk.source}"
            segments.append(
                PromptSegment(
                    role="retrieved_visual",
                    modality="image",
                    source=chunk.source,
                    text=visual_text,
                    text_length=len(visual_text),
                    retrieval_rank=chunk.retrieval_rank,
                    metadata=segment_metadata,
                )
            )
        else:
            segments.append(
                _make_text_segment(
                    role="retrieved_text",
                    source=chunk.source,
                    text=chunk.text,
                    retrieval_rank=chunk.retrieval_rank,
                    metadata=segment_metadata,
                )
            )

    if effective_image_path:
        visual_text = f"image attached: {effective_image_path}"
        segments.append(
            PromptSegment(
                role="visual_context",
                modality="image",
                source=effective_image_path,
                text=visual_text,
                text_length=len(visual_text),
            )
        )

    payload, truncation_metadata = _build_vllm_chat_payload(
        system_instruction=system_instruction,
        query=effective_query,
        retrieved_chunks=ordered_chunks,
        image_path=effective_image_path,
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    retrieved_visual_count = sum(1 for chunk in ordered_chunks if getattr(chunk, "modality", "text") == "image")
    segment_metadata = {
        "ordered_segment_roles": [segment.role for segment in segments],
        "segment_count": len(segments),
        "has_visual_context": effective_image_path is not None or retrieved_visual_count > 0,
        "retrieved_visual_count": retrieved_visual_count,
        "sources": [segment.source for segment in segments],
        **truncation_metadata,
    }

    return {
        "segments": segments,
        "payload": payload,
        "segment_metadata": segment_metadata,
    }


assemble_prompt = build_prompt


def _normalize_chunks(
    chunks: Iterable[RetrievedChunk | dict[str, Any]] | None,
) -> list[RetrievedChunk]:
    normalized: list[RetrievedChunk] = []
    for chunk in chunks or []:
        if isinstance(chunk, RetrievedChunk):
            normalized.append(chunk)
        else:
            normalized.append(RetrievedChunk.model_validate(chunk))

    return sorted(
        normalized,
        key=lambda chunk: (
            chunk.retrieval_rank is None,
            chunk.retrieval_rank if chunk.retrieval_rank is not None else 0,
            chunk.chunk_id,
        ),
    )


def _make_text_segment(
    *,
    role: str,
    source: str,
    text: str,
    retrieval_rank: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> PromptSegment:
    return PromptSegment(
        role=role,
        modality="text",
        source=source,
        text=text,
        text_length=len(text),
        retrieval_rank=retrieval_rank,
        metadata=metadata or {},
    )


def _build_vllm_chat_payload(
    *,
    system_instruction: str,
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    image_path: str | None,
    max_input_tokens: int | None,
    reserved_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    retrieved_visuals = [
        chunk for chunk in retrieved_chunks if getattr(chunk, "modality", "text") == "image" and getattr(chunk, "image_path", None)
    ]
    user_text, truncation_metadata = _build_user_text_block(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=reserved_output_tokens,
        system_instruction=system_instruction,
        has_visual_context=bool(image_path or retrieved_visuals),
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_text,
        }
    ]
    if image_path:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{image_path}"},
            }
        )
    for chunk in retrieved_visuals:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{chunk.image_path}"},
            }
        )

    return (
        {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
        },
        truncation_metadata,
    )


def _build_user_text_block(
    *,
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    max_input_tokens: int | None,
    reserved_output_tokens: int,
    system_instruction: str,
    has_visual_context: bool,
) -> tuple[str, dict[str, Any]]:
    lines = [f"User query:\n{query}"]
    retrieved_text_chunks = [chunk for chunk in retrieved_chunks if getattr(chunk, "modality", "text") != "image"]
    retrieved_visual_chunks = [chunk for chunk in retrieved_chunks if getattr(chunk, "modality", "text") == "image"]
    if retrieved_text_chunks:
        lines.append("Retrieved context:")
        for chunk in retrieved_text_chunks:
            rank = chunk.retrieval_rank if chunk.retrieval_rank is not None else "unranked"
            lines.append(f"[{rank}] {chunk.source}\n{chunk.text}")
    if retrieved_visual_chunks:
        lines.append("Retrieved visual context:")
        for chunk in retrieved_visual_chunks:
            rank = chunk.retrieval_rank if chunk.retrieval_rank is not None else "unranked"
            description = chunk.text or f"Image from {chunk.source}"
            lines.append(f"[{rank}] {chunk.source}\n{description}")
    user_text = "\n\n".join(lines)
    budget_tokens = _compute_user_text_budget_tokens(
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=reserved_output_tokens,
        system_instruction=system_instruction,
        has_visual_context=has_visual_context,
    )
    if budget_tokens is None:
        return user_text, {"prompt_truncated": False}
    truncated_text, truncation_metadata = _truncate_text_to_token_budget(user_text, budget_tokens)
    return truncated_text, truncation_metadata


def _compute_user_text_budget_tokens(
    *,
    max_input_tokens: int | None,
    reserved_output_tokens: int,
    system_instruction: str,
    has_visual_context: bool,
) -> int | None:
    if max_input_tokens is None or max_input_tokens <= 0:
        return None
    # Keep room for assistant output and chat/template overhead.
    reserve = max(1, int(reserved_output_tokens))
    template_overhead = 64
    visual_overhead = 64 if has_visual_context else 0
    system_tokens = _approx_token_count(system_instruction)
    available = max_input_tokens - reserve - template_overhead - visual_overhead - system_tokens
    return max(64, available)


def _truncate_text_to_token_budget(text: str, budget_tokens: int) -> tuple[str, dict[str, Any]]:
    if _approx_token_count(text) <= budget_tokens:
        return text, {"prompt_truncated": False}

    parts = text.split("\n\n")
    removed_context_blocks = 0
    # Always preserve user query block at index 0.
    while len(parts) > 1 and _approx_token_count("\n\n".join(parts)) > budget_tokens:
        parts.pop()
        removed_context_blocks += 1

    current = "\n\n".join(parts)
    if _approx_token_count(current) <= budget_tokens:
        return current, {
            "prompt_truncated": True,
            "retrieved_context_truncated": removed_context_blocks > 0,
            "user_query_truncated": False,
            "removed_context_blocks": removed_context_blocks,
            "budget_tokens": budget_tokens,
        }

    # If still too long, truncate user query text itself.
    tokens = current.split()
    if len(tokens) <= budget_tokens:
        return current, {
            "prompt_truncated": removed_context_blocks > 0,
            "retrieved_context_truncated": removed_context_blocks > 0,
            "user_query_truncated": False,
            "removed_context_blocks": removed_context_blocks,
            "budget_tokens": budget_tokens,
        }
    return " ".join(tokens[:budget_tokens]), {
        "prompt_truncated": True,
        "retrieved_context_truncated": removed_context_blocks > 0,
        "user_query_truncated": True,
        "removed_context_blocks": removed_context_blocks,
        "budget_tokens": budget_tokens,
    }


def _approx_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    # Rough heuristic to avoid overflow; intentionally conservative.
    return max(1, len(stripped) // 4)
