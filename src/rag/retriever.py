from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import open_clip
from PIL import Image
import torch

from .corpus import CorpusChunk


class OpenCLIPTextEmbedder:
    """Lazy text embedder backed by open_clip text encoders."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._open_clip_model_name, self._open_clip_pretrained = _resolve_model_spec(model_name)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._image_preprocess: Any | None = None

    def encode(self, inputs: str | Sequence[str]) -> np.ndarray:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            return np.empty((0, 0), dtype="float32")

        self._ensure_model()
        assert self._model is not None
        assert self._tokenizer is not None

        tokens = self._tokenizer(texts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)

        with torch.inference_mode():
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return text_features.detach().cpu().numpy().astype("float32")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        encoded = self.encode(text)
        return encoded[0]

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        image_list = list(images)
        if not image_list:
            return np.empty((0, 0), dtype="float32")

        self._ensure_model()
        assert self._model is not None
        assert self._image_preprocess is not None

        tensors = [self._image_preprocess(image.convert("RGB")) for image in image_list]
        image_batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode():
            image_features = self._model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return image_features.detach().cpu().numpy().astype("float32")

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._image_preprocess is not None:
            return

        model, image_preprocess = open_clip.create_model_from_pretrained(
            self._open_clip_model_name,
            pretrained=self._open_clip_pretrained,
            device=self.device,
            cache_dir=self.cache_dir,
            return_transform=True,
        )
        self._model = model.eval()
        self._image_preprocess = image_preprocess
        self._tokenizer = open_clip.get_tokenizer(self._open_clip_model_name)


class Retriever:
    """Minimal in-memory FAISS retriever with injectable embeddings for tests."""

    def __init__(
        self,
        chunks: Sequence[CorpusChunk] | None = None,
        *,
        corpus_chunks: Sequence[CorpusChunk] | None = None,
        embedder: Any | None = None,
        embedding_backend: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self.chunks = list(chunks or corpus_chunks or [])
        self.config = config
        self.top_k = int(getattr(config, "top_k", 5))
        self.embedder = embedder or embedding_backend or OpenCLIPTextEmbedder(
            getattr(config, "embedding_model", "openai/clip-vit-large-patch14")
        )
        self.text_chunks = [chunk for chunk in self.chunks if getattr(chunk, "modality", "text") != "image"]
        self.image_chunks = [chunk for chunk in self.chunks if getattr(chunk, "modality", "text") == "image"]
        self.text_index: faiss.IndexFlatIP | None = None
        self.image_index: faiss.IndexFlatIP | None = None
        self._text_embeddings: np.ndarray | None = None
        self._image_embeddings: np.ndarray | None = None

        if self.chunks:
            self._build_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[CorpusChunk]:
        if not self.chunks:
            return []
        if self.text_index is None and self.image_index is None:
            self._build_index()

        effective_top_k = max(1, top_k or self.top_k)
        query_embedding = _normalize_embeddings(self._embed_query(query).reshape(1, -1))
        text_results = self._search_index(
            index=self.text_index,
            query_embedding=query_embedding,
            chunks=self.text_chunks,
            top_k=effective_top_k,
        )
        image_results = self._search_index(
            index=self.image_index,
            query_embedding=query_embedding,
            chunks=self.image_chunks,
            top_k=effective_top_k,
        )

        scored_results = text_results + image_results
        scored_results.sort(key=lambda item: (-item[0], item[1].chunk_id, item[1].modality))
        ranked_results: list[CorpusChunk] = []
        for rank, (_, chunk) in enumerate(scored_results, start=1):
            ranked_results.append(
                replace(
                    chunk,
                    retrieval_rank=rank,
                    score=scored_results[rank - 1][0],
                    metadata={**chunk.metadata, "score": scored_results[rank - 1][0]},
                )
            )
        return ranked_results

    search = retrieve
    __call__ = retrieve

    def _build_index(self) -> None:
        if self.text_chunks:
            text_embeddings = _normalize_embeddings(self._embed_documents([chunk.text for chunk in self.text_chunks]))
            self._text_embeddings = text_embeddings
            self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
            self.text_index.add(text_embeddings)

        if self.image_chunks:
            images = [self._load_image(chunk) for chunk in self.image_chunks]
            image_embeddings = _normalize_embeddings(self._embed_images(images))
            self._image_embeddings = image_embeddings
            self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
            self.image_index.add(image_embeddings)

    def _embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.embedder, "embed_documents"):
            embeddings = self.embedder.embed_documents(list(texts))
        else:
            embeddings = self.embedder.encode(list(texts))
        return _as_float32_matrix(embeddings)

    def _embed_query(self, text: str) -> np.ndarray:
        if hasattr(self.embedder, "embed_query"):
            embedding = self.embedder.embed_query(text)
        else:
            embedding = self.embedder.encode(text)
        return _as_float32_vector(embedding)

    def _embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if hasattr(self.embedder, "embed_images"):
            embeddings = self.embedder.embed_images(list(images))
            return _as_float32_matrix(embeddings)
        raise AttributeError("Embedding backend must expose 'embed_images' for image retrieval.")

    def _load_image(self, chunk: CorpusChunk) -> Image.Image:
        if not chunk.image_path:
            raise ValueError(f"Image chunk {chunk.chunk_id} is missing image_path.")
        return Image.open(chunk.image_path).convert("RGB")

    def _search_index(
        self,
        *,
        index: faiss.IndexFlatIP | None,
        query_embedding: np.ndarray,
        chunks: Sequence[CorpusChunk],
        top_k: int,
    ) -> list[tuple[float, CorpusChunk]]:
        if index is None or not chunks:
            return []

        candidate_count = min(len(chunks), max(top_k, top_k * 4))
        scores, indices = index.search(query_embedding, candidate_count)

        scored_results: list[tuple[float, CorpusChunk]] = []
        for score, index_value in zip(scores[0], indices[0], strict=True):
            if index_value < 0:
                continue
            scored_results.append((float(score), chunks[int(index_value)]))
        return scored_results


_OPEN_CLIP_MODEL_ALIASES: dict[str, tuple[str, str | None]] = {
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
}


def _resolve_model_spec(model_name: str) -> tuple[str, str | None]:
    if model_name in _OPEN_CLIP_MODEL_ALIASES:
        return _OPEN_CLIP_MODEL_ALIASES[model_name]
    if model_name.startswith("hf-hub:"):
        return model_name, None
    if "/" in model_name:
        return f"hf-hub:{model_name}", None
    return model_name, None


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (embeddings / norms).astype("float32")


def _as_float32_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _as_float32_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype="float32")
    if array.ndim != 1:
        array = array.reshape(-1)
    return array
