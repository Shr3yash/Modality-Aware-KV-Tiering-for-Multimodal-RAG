from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import (
    CorpusChunk,
    DEFAULT_VIDORE_CONFIG,
    DEFAULT_VIDORE_DATASET,
    DEFAULT_VIDORE_SPLIT,
    _materialize_record_image,
    download_vidore_corpus,
)


def build_byaldi_image_corpus(
    image_dir: str | Path,
    *,
    dataset_name: str = DEFAULT_VIDORE_DATASET,
    dataset_config: str = DEFAULT_VIDORE_CONFIG,
    dataset_split: str = DEFAULT_VIDORE_SPLIT,
    staging_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, list[dict]]:
    """Download ViDoRe dataset and materialize all images to image_dir.

    Returns (image_dir_path, list_of_record_metadata_dicts_in_index_order).
    Each dict has keys: source, doc_id, page_number, text, image_path, metadata.
    """
    from datasets import load_from_disk

    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    effective_staging = Path(staging_dir) if staging_dir else image_dir_path.parent / "corpus_staging"
    download_vidore_corpus(
        effective_staging,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        force=force,
    )

    dataset = load_from_disk(str(effective_staging))

    records_in_order: list[dict] = []
    for record in dataset:
        image = record.get("image")
        if image is None:
            continue

        doc_id = _extract_doc_id(record)
        page_number = _extract_page_number(record)
        source = _extract_source(record, doc_id, page_number)
        text = _extract_text(record)
        metadata = _extract_metadata(record)

        image_path = _materialize_record_image(
            image=image,
            image_cache_dir=image_dir_path,
            source=source,
            doc_id=doc_id,
            page_number=page_number,
        )

        records_in_order.append({
            "source": source,
            "doc_id": doc_id,
            "page_number": page_number,
            "text": text,
            "image_path": str(image_path),
            "metadata": metadata,
        })

    return image_dir_path, records_in_order


class ByaldiRetriever:
    """Multimodal retriever backed by Byaldi + ColQwen2.

    Wraps byaldi.RAGMultiModalModel and adapts its search results to
    CorpusChunk objects so it can serve as a drop-in replacement for
    the FAISS-based Retriever.

    Source: https://github.com/AnswerDotAI/byaldi (Apache 2.0)
    Model: vidore/colqwen2-v1.0 — ColPali paper: arXiv:2407.01449
    """

    def __init__(
        self,
        *,
        config: Any | None = None,
        index_dir: str | Path | None = None,
        image_dir: str | Path | None = None,
        model_name: str = "vidore/colqwen2-v1.0",
        top_k: int = 5,
        dataset_name: str = DEFAULT_VIDORE_DATASET,
        dataset_config: str = DEFAULT_VIDORE_CONFIG,
        dataset_split: str = DEFAULT_VIDORE_SPLIT,
        _rag_model: Any | None = None,
    ) -> None:
        self.top_k = int(top_k if config is None else getattr(config, "top_k", top_k))
        self.model_name = getattr(config, "colpali_model", model_name) if config else model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split

        _index_dir = index_dir or (getattr(config, "index_dir", "./data/byaldi_index") if config else "./data/byaldi_index")
        self.index_dir = Path(_index_dir)

        if image_dir is not None:
            self.image_dir = Path(image_dir)
        elif config is not None:
            corpus_dir = getattr(config, "corpus_dir", "./data/corpus")
            self.image_dir = Path(corpus_dir) / ".image_cache"
        else:
            self.image_dir = Path("./data/corpus/.image_cache")

        self._rag_model: Any | None = _rag_model
        self._chunk_map: dict[int, dict] = {}

        # If a model was injected (test path), also try to load the chunk map
        # if the index dir already has one; otherwise _chunk_map stays empty.
        if self._rag_model is not None and self._index_exists():
            self._load_chunk_map()

    # ------------------------------------------------------------------
    # Public retrieval interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[CorpusChunk]:
        """Search the indexed corpus and return CorpusChunk results."""
        self._ensure_model()
        effective_k = max(1, top_k if top_k is not None else self.top_k)
        results = self._rag_model.search(query, k=effective_k)
        corpus_chunks: list[CorpusChunk] = []
        for rank, result in enumerate(results, start=1):
            corpus_chunks.append(self._result_to_corpus_chunk(result, rank=rank))
        return corpus_chunks

    search = retrieve
    __call__ = retrieve

    # ------------------------------------------------------------------
    # Lazy model initialization
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._rag_model is not None:
            return
        from byaldi import RAGMultiModalModel  # deferred — byaldi may not be installed everywhere

        if self._index_exists():
            self._rag_model = RAGMultiModalModel.from_index(str(self.index_dir))
        else:
            self._rag_model = RAGMultiModalModel.from_pretrained(self.model_name)
            self._build_index()
        self._load_chunk_map()

    def _index_exists(self) -> bool:
        return self.index_dir.exists() and any(self.index_dir.iterdir())

    # ------------------------------------------------------------------
    # Index construction (first run only)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Download ViDoRe, materialize images, build and persist byaldi index."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = self.index_dir.parent / "corpus_staging"

        _image_dir, records = build_byaldi_image_corpus(
            self.image_dir,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            staging_dir=staging_dir,
        )

        # Build the chunk map: byaldi assigns integer doc_ids in the order
        # images are indexed (which matches the order files appear in image_dir).
        # We write a sidecar JSON so the mapping survives restarts.
        chunk_map: dict[str, dict] = {
            str(idx): record for idx, record in enumerate(records)
        }

        self._rag_model.index(
            input_path=str(self.image_dir),
            index_name=self.index_dir.name,
            store_collection_with_index=True,
            index_root=str(self.index_dir.parent),
        )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunk_map_path = self.index_dir / "chunk_map.json"
        chunk_map_path.write_text(json.dumps(chunk_map, ensure_ascii=False), encoding="utf-8")

        # Populate in-memory map
        self._chunk_map = {int(k): v for k, v in chunk_map.items()}

    # ------------------------------------------------------------------
    # Chunk map (persisted sidecar file)
    # ------------------------------------------------------------------

    def _load_chunk_map(self) -> None:
        chunk_map_path = self.index_dir / "chunk_map.json"
        if not chunk_map_path.exists():
            return
        raw: dict[str, dict] = json.loads(chunk_map_path.read_text(encoding="utf-8"))
        self._chunk_map = {int(k): v for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Result → CorpusChunk adapter
    # ------------------------------------------------------------------

    def _result_to_corpus_chunk(self, result: Any, *, rank: int) -> CorpusChunk:
        """Convert a byaldi Result object to a CorpusChunk."""
        raw = self._chunk_map.get(int(result.doc_id), {})

        source = raw.get("source") or f"doc{result.doc_id}#page={result.page_num}"
        doc_id = raw.get("doc_id")
        page_number = raw.get("page_number", result.page_num)
        image_path = raw.get("image_path")
        text = raw.get("text") or f"Retrieved image from {source}"

        chunk_id = hashlib.sha256(
            f"byaldi:{result.doc_id}:{result.page_num}".encode("utf-8")
        ).hexdigest()[:32]

        score = float(result.score)
        metadata: dict = {
            **(raw.get("metadata") or {}),
            "score": score,
            "byaldi_doc_id": int(result.doc_id),
        }

        return CorpusChunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            modality="image",
            image_path=image_path,
            doc_id=doc_id,
            page_number=page_number,
            retrieval_rank=rank,
            score=score,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Inline field-extraction helpers (avoid importing private corpus.py functions)
# ---------------------------------------------------------------------------

def _extract_text(record: dict) -> str:
    for key in ("markdown", "text", "content", "query", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_doc_id(record: dict) -> str | None:
    for key in ("doc_id", "document_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_page_number(record: dict) -> int | None:
    value = record.get("page_number_in_doc", record.get("page_number"))
    return value if isinstance(value, int) else None


def _extract_source(record: dict, doc_id: str | None, page_number: int | None) -> str:
    source = record.get("source")
    if isinstance(source, str) and source:
        return source
    if doc_id and page_number is not None:
        return f"{doc_id}#page={page_number}"
    if doc_id:
        return doc_id
    return "unknown"


def _extract_metadata(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"markdown", "text", "content", "image"}
    }
