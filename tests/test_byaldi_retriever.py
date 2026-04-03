"""Tests for ByaldiRetriever using injected fakes — no model download required."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.rag.byaldi_retriever import ByaldiRetriever
from src.rag.corpus import CorpusChunk
from src.rag.schemas import GenerateRequest, GenerateResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeByaldiResult:
    def __init__(
        self,
        doc_id: int,
        page_num: int,
        score: float,
        metadata: dict | None = None,
        base64: str | None = None,
    ) -> None:
        self.doc_id = doc_id
        self.page_num = page_num
        self.score = score
        self.metadata = metadata or {}
        self.base64 = base64


class FakeRAGMultiModalModel:
    """Stand-in for byaldi.RAGMultiModalModel — no weights, no downloads."""

    _last_index_kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FakeRAGMultiModalModel":
        return cls()

    @classmethod
    def from_index(cls, index_path: str) -> "FakeRAGMultiModalModel":
        return cls()

    def index(self, **kwargs: Any) -> None:
        FakeRAGMultiModalModel._last_index_kwargs = kwargs

    def search(self, query: str, k: int) -> list[FakeByaldiResult]:
        all_results = [
            FakeByaldiResult(doc_id=0, page_num=1, score=0.95),
            FakeByaldiResult(doc_id=1, page_num=2, score=0.82),
            FakeByaldiResult(doc_id=2, page_num=1, score=0.70),
        ]
        return all_results[:k]


class FakeModelRunner:
    def generate(self, request_payload: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(generated_text="fake answer", timing={}, raw_response=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_byaldi_config(tmp_path: Any, top_k: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        corpus_dir=str(tmp_path / "corpus"),
        chunk_size_tokens=4,
        top_k=top_k,
        embedding_model="test-model",
        retriever_backend="byaldi",
        index_dir=str(tmp_path / "byaldi_index"),
        colpali_model="vidore/colqwen2-v1.0",
    )


def make_chunk_map(tmp_path: Any) -> dict[int, dict]:
    """Write a chunk_map.json and return the in-memory dict."""
    index_dir = tmp_path / "byaldi_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_map = {
        "0": {
            "source": "doc-a#page=1",
            "doc_id": "doc-a",
            "page_number": 1,
            "text": "Architecture overview diagram",
            "image_path": str(tmp_path / "img0.png"),
            "metadata": {"corpus_id": "sample-0"},
        },
        "1": {
            "source": "doc-b#page=2",
            "doc_id": "doc-b",
            "page_number": 2,
            "text": "Cache tiering illustration",
            "image_path": str(tmp_path / "img1.png"),
            "metadata": {"corpus_id": "sample-1"},
        },
        "2": {
            "source": "doc-c#page=1",
            "doc_id": "doc-c",
            "page_number": 1,
            "text": "",
            "image_path": str(tmp_path / "img2.png"),
            "metadata": {},
        },
    }
    (index_dir / "chunk_map.json").write_text(
        json.dumps(chunk_map), encoding="utf-8"
    )
    return {int(k): v for k, v in chunk_map.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_corpus_chunks(tmp_path):
    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    results = retriever.retrieve("What does the architecture diagram show?", top_k=2)

    assert len(results) == 2
    for chunk in results:
        assert isinstance(chunk, CorpusChunk)
        assert chunk.modality == "image"
        assert chunk.score is not None
        assert chunk.retrieval_rank is not None
        assert chunk.chunk_id


def test_retrieve_respects_top_k(tmp_path):
    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path, top_k=3),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    assert len(retriever.retrieve("query", top_k=1)) == 1
    assert len(retriever.retrieve("query", top_k=2)) == 2


def test_config_top_k_used_when_none_passed(tmp_path):
    chunk_map = make_chunk_map(tmp_path)

    class SpyModel(FakeRAGMultiModalModel):
        last_k: int = -1

        def search(self, query: str, k: int) -> list[FakeByaldiResult]:
            SpyModel.last_k = k
            return super().search(query, k)

    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path, top_k=3),
        _rag_model=SpyModel(),
    )
    retriever._chunk_map = chunk_map

    retriever.retrieve("query")  # no explicit top_k
    assert SpyModel.last_k == 3


def test_loads_existing_index_not_rebuilds(tmp_path, monkeypatch):
    """When index_dir is non-empty, _index_exists returns True and from_index path is taken."""
    make_chunk_map(tmp_path)  # writes chunk_map.json, making index_dir non-empty

    calls: dict[str, int] = {"from_pretrained": 0, "from_index": 0}

    class TrackingModel(FakeRAGMultiModalModel):
        @classmethod
        def from_pretrained(cls, model_name: str) -> "TrackingModel":
            calls["from_pretrained"] += 1
            return cls()

        @classmethod
        def from_index(cls, index_path: str) -> "TrackingModel":
            calls["from_index"] += 1
            return cls()

    # Patch the deferred byaldi import inside _ensure_model using a fake module
    import types
    fake_byaldi = types.ModuleType("byaldi")
    fake_byaldi.RAGMultiModalModel = TrackingModel  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "byaldi", fake_byaldi)

    retriever = ByaldiRetriever(config=make_byaldi_config(tmp_path))
    assert retriever._index_exists(), "Index dir should be non-empty after make_chunk_map"

    retriever._ensure_model()

    assert calls["from_index"] == 1, "Should load existing index, not rebuild"
    assert calls["from_pretrained"] == 0, "Should not call from_pretrained when index exists"


def test_result_to_corpus_chunk_maps_fields_correctly(tmp_path):
    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    result = FakeByaldiResult(doc_id=0, page_num=1, score=0.95)
    chunk = retriever._result_to_corpus_chunk(result, rank=1)

    assert chunk.modality == "image"
    assert chunk.source == "doc-a#page=1"
    assert chunk.doc_id == "doc-a"
    assert chunk.page_number == 1
    assert chunk.retrieval_rank == 1
    assert abs(chunk.score - 0.95) < 1e-6
    assert chunk.text == "Architecture overview diagram"
    assert chunk.metadata["byaldi_doc_id"] == 0


def test_result_to_corpus_chunk_fallback_text(tmp_path):
    """Image chunk with empty text should get a synthesized description."""
    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    result = FakeByaldiResult(doc_id=2, page_num=1, score=0.70)
    chunk = retriever._result_to_corpus_chunk(result, rank=1)

    assert chunk.text.startswith("Retrieved image from")


def test_result_to_corpus_chunk_missing_from_map(tmp_path):
    """Unknown doc_id should not crash — fallback fields used."""
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = {}

    result = FakeByaldiResult(doc_id=99, page_num=3, score=0.50)
    chunk = retriever._result_to_corpus_chunk(result, rank=2)

    assert isinstance(chunk, CorpusChunk)
    assert chunk.modality == "image"
    assert chunk.retrieval_rank == 2
    assert "99" in chunk.source or "doc99" in chunk.source


def test_search_and_call_aliases_work(tmp_path):
    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    via_retrieve = retriever.retrieve("q", top_k=1)
    via_search = retriever.search("q", top_k=1)
    via_call = retriever("q", top_k=1)

    assert len(via_retrieve) == len(via_search) == len(via_call) == 1


def test_compatible_with_generation_service(tmp_path):
    """ByaldiRetriever should wire into GenerationService without errors."""
    from src.serving.service import GenerationService

    chunk_map = make_chunk_map(tmp_path)
    retriever = ByaldiRetriever(
        config=make_byaldi_config(tmp_path),
        _rag_model=FakeRAGMultiModalModel(),
    )
    retriever._chunk_map = chunk_map

    service = GenerationService(
        retriever=retriever,
        model_runner=FakeModelRunner(),
    )

    response = service.generate(GenerateRequest(query="test query", top_k=2))

    assert isinstance(response, GenerateResponse)
    assert isinstance(response.answer_text, str)
    assert isinstance(response.retrieved_chunk_ids, list)
    assert len(response.retrieved_chunk_ids) == 2
