from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query: str
    image_path: str | None = None
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    modality: str = "text"
    image_path: str | None = None
    retrieval_rank: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptSegment(BaseModel):
    role: str
    modality: str
    source: str
    text: str | None = None
    text_length: int
    retrieval_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    answer_text: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    prompt_summary: dict[str, Any] = Field(default_factory=dict)
    token_summary: dict[str, Any] = Field(default_factory=dict)
    timing_summary: dict[str, Any] = Field(default_factory=dict)
    segments: list[PromptSegment] = Field(default_factory=list)
