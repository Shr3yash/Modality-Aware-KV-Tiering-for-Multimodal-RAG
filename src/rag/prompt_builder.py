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

    payload = _build_vllm_chat_payload(
        system_instruction=system_instruction,
        query=effective_query,
        retrieved_chunks=ordered_chunks,
        image_path=effective_image_path,
    )
    retrieved_visual_count = sum(1 for chunk in ordered_chunks if getattr(chunk, "modality", "text") == "image")
    segment_metadata = {
        "ordered_segment_roles": [segment.role for segment in segments],
        "segment_count": len(segments),
        "has_visual_context": effective_image_path is not None or retrieved_visual_count > 0,
        "retrieved_visual_count": retrieved_visual_count,
        "sources": [segment.source for segment in segments],
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
) -> dict[str, Any]:
    retrieved_visuals = [
        chunk for chunk in retrieved_chunks if getattr(chunk, "modality", "text") == "image" and getattr(chunk, "image_path", None)
    ]
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_user_text_block(query=query, retrieved_chunks=retrieved_chunks),
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

    return {
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
    }


def _build_user_text_block(
    *,
    query: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
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
    return "\n\n".join(lines)
