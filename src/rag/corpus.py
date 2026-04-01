from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_VIDORE_DATASET = "vidore/vidore_v3_hr"
DEFAULT_VIDORE_CONFIG = "corpus"
DEFAULT_VIDORE_SPLIT = "test"
_HF_DATASET_MARKERS = (
    "dataset_info.json",
    "state.json",
    "dataset_dict.json",
)


@dataclass(slots=True)
class CorpusChunk:
    chunk_id: str
    text: str
    source: str
    modality: str = "text"
    image_path: str | None = None
    token_ids: tuple[int, ...] = field(default_factory=tuple)
    doc_id: str | None = None
    page_number: int | None = None
    chunk_index: int = 0
    retrieval_rank: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FallbackTokenizer:
    """Tiny encode/decode adapter for offline tests when no model tokenizer is available."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inverse_vocab: dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False) -> list[int]:
        _ = (add_special_tokens, truncation)
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        token_ids: list[int] = []
        for piece in pieces:
            if piece not in self._vocab:
                token_id = self._next_id
                self._vocab[piece] = token_id
                self._inverse_vocab[token_id] = piece
                self._next_id += 1
            token_ids.append(self._vocab[piece])
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(self._inverse_vocab[token_id] for token_id in token_ids if token_id in self._inverse_vocab)


def download_vidore_corpus(
    target_dir: str | Path,
    *,
    dataset_name: str = DEFAULT_VIDORE_DATASET,
    dataset_config: str = DEFAULT_VIDORE_CONFIG,
    split: str = DEFAULT_VIDORE_SPLIT,
    force: bool = False,
) -> Path:
    """Download the ViDoRe corpus split and persist it with datasets.save_to_disk."""
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()) and not force:
        return target_path

    target_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset.save_to_disk(str(target_path))
    return target_path


def load_corpus(
    corpus_dir: str | Path | None = None,
    *,
    chunk_size_tokens: int | None = None,
    config: Any | None = None,
    tokenizer: PreTrainedTokenizerBase | Any | None = None,
    tokenizer_name: str | None = None,
    dataset_name: str = DEFAULT_VIDORE_DATASET,
    dataset_config: str = DEFAULT_VIDORE_CONFIG,
    dataset_split: str = DEFAULT_VIDORE_SPLIT,
    auto_download: bool = False,
) -> list[CorpusChunk]:
    """Load either a saved Hugging Face dataset directory or plain-text files into chunks."""
    corpus_path = Path(corpus_dir or getattr(config, "corpus_dir", "./data/corpus"))
    effective_chunk_size = int(chunk_size_tokens or getattr(config, "chunk_size_tokens", 512))
    effective_tokenizer = _resolve_tokenizer(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name or getattr(config, "embedding_model", None),
    )

    if auto_download and (not corpus_path.exists() or not any(corpus_path.iterdir())):
        download_vidore_corpus(
            corpus_path,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=dataset_split,
        )

    if _looks_like_hf_dataset_dir(corpus_path):
        dataset = load_from_disk(str(corpus_path))
        records = _iter_dataset_records(dataset)
    else:
        records = _iter_text_records(corpus_path)

    chunks: list[CorpusChunk] = []
    image_cache_dir = corpus_path / ".image_cache"
    for record in records:
        text = _extract_record_text(record)
        source = _extract_record_source(record)
        metadata = _extract_metadata(record)
        doc_id = _extract_doc_id(record)
        page_number = _extract_page_number(record)

        if text:
            chunked_texts, chunk_token_ids = _chunk_text(
                text,
                chunk_size_tokens=effective_chunk_size,
                tokenizer=effective_tokenizer,
            )
            for chunk_index, (chunk_text, token_ids) in enumerate(zip(chunked_texts, chunk_token_ids, strict=True)):
                if not chunk_text.strip():
                    continue
                chunks.append(
                    CorpusChunk(
                        chunk_id=_build_chunk_id(source, chunk_index, token_ids),
                        text=chunk_text,
                        source=source,
                        modality="text",
                        token_ids=tuple(token_ids),
                        doc_id=doc_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )

        image = _extract_record_image(record)
        if image is not None:
            image_path = _materialize_record_image(
                image=image,
                image_cache_dir=image_cache_dir,
                source=source,
                doc_id=doc_id,
                page_number=page_number,
            )
            visual_text = text.strip() if text else f"Retrieved image from {source}"
            chunks.append(
                CorpusChunk(
                    chunk_id=_build_image_chunk_id(source, image_path),
                    text=visual_text,
                    source=source,
                    modality="image",
                    image_path=str(image_path),
                    doc_id=doc_id,
                    page_number=page_number,
                    metadata=metadata,
                )
            )

    return chunks


def _resolve_tokenizer(
    *,
    tokenizer: PreTrainedTokenizerBase | Any | None,
    tokenizer_name: str | None,
) -> PreTrainedTokenizerBase | Any:
    if tokenizer is not None:
        return tokenizer

    model_name = tokenizer_name or "openai/clip-vit-large-patch14"
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return FallbackTokenizer()


def _looks_like_hf_dataset_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _HF_DATASET_MARKERS)


def _iter_dataset_records(dataset: Dataset | DatasetDict) -> list[dict[str, Any]]:
    if isinstance(dataset, DatasetDict):
        records: list[dict[str, Any]] = []
        for split_name in sorted(dataset.keys()):
            records.extend(_iter_dataset_records(dataset[split_name]))
        return records

    return list(dataset)


def _iter_text_records(corpus_path: Path) -> list[dict[str, Any]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    text_files = sorted(path for path in corpus_path.rglob("*.txt") if path.is_file())
    records: list[dict[str, Any]] = []
    for text_file in text_files:
        records.append(
            {
                "source": str(text_file),
                "doc_id": text_file.stem,
                "markdown": text_file.read_text(encoding="utf-8"),
            }
        )
    return records


def _extract_record_text(record: dict[str, Any]) -> str:
    for field_name in ("markdown", "text", "content", "query", "answer"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_record_source(record: dict[str, Any]) -> str:
    source = record.get("source")
    if isinstance(source, str) and source:
        return source

    doc_id = _extract_doc_id(record)
    page_number = _extract_page_number(record)
    if doc_id and page_number is not None:
        return f"{doc_id}#page={page_number}"
    if doc_id:
        return doc_id
    return "unknown"


def _extract_doc_id(record: dict[str, Any]) -> str | None:
    for field_name in ("doc_id", "document_id", "id"):
        value = record.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_page_number(record: dict[str, Any]) -> int | None:
    value = record.get("page_number_in_doc", record.get("page_number"))
    return value if isinstance(value, int) else None


def _extract_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"markdown", "text", "content", "image"}
    }
    return metadata


def _extract_record_image(record: dict[str, Any]) -> Any | None:
    image = record.get("image")
    return image if image is not None else None


def _chunk_text(
    text: str,
    *,
    chunk_size_tokens: int,
    tokenizer: PreTrainedTokenizerBase | Any,
) -> tuple[list[str], list[list[int]]]:
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
            token_ids = list(encoded["input_ids"])
        except Exception:
            token_ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    else:
        token_ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    if not token_ids:
        return [], []

    chunk_texts: list[str] = []
    chunk_token_ids: list[list[int]] = []
    for start in range(0, len(token_ids), max(chunk_size_tokens, 1)):
        token_window = token_ids[start : start + max(chunk_size_tokens, 1)]
        decoded = tokenizer.decode(token_window, skip_special_tokens=True).strip()
        if decoded:
            chunk_texts.append(decoded)
            chunk_token_ids.append(token_window)
    return chunk_texts, chunk_token_ids


def _build_chunk_id(source: str, chunk_index: int, token_ids: list[int]) -> str:
    raw = f"{source}:{chunk_index}:{','.join(str(token_id) for token_id in token_ids)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _build_image_chunk_id(source: str, image_path: Path) -> str:
    raw = f"image:{source}:{image_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _materialize_record_image(
    *,
    image: Any,
    image_cache_dir: Path,
    source: str,
    doc_id: str | None,
    page_number: int | None,
) -> Path:
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    stem = doc_id or source
    page_suffix = f"-page-{page_number}" if page_number is not None else ""
    filename = hashlib.sha256(f"{stem}{page_suffix}".encode("utf-8")).hexdigest()[:24]
    image_path = image_cache_dir / f"{filename}.png"
    if not image_path.exists():
        image.save(image_path)
    return image_path
