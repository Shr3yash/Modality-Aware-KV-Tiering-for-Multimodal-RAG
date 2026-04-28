from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tests.phase2_spec_helpers import (
    call_with_supported_args,
    get_value,
    instantiate_with_supported_args,
    make_rag_config,
    require_any_attr,
    require_phase2_module,
)


class FakeEmbedder:
    vocabulary = ("cache", "image", "text")

    def _embed_one(self, text: str) -> np.ndarray:
        lowered = text.lower()
        return np.asarray([lowered.count(token) for token in self.vocabulary], dtype="float32")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def encode(self, inputs):
        if isinstance(inputs, str):
            return self._embed_one(inputs)
        return self.embed_documents(list(inputs))

    def embed_images(self, images):
        vectors = []
        for image in images:
            red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            vectors.append(np.asarray([red / 255.0, blue / 255.0, green / 255.0], dtype="float32"))
        return np.stack(vectors)


def test_corpus_loader_chunks_plaintext_and_attaches_ids_and_source(tmp_path: Path) -> None:
    corpus_module = require_phase2_module("src.rag.corpus")
    load_corpus = require_any_attr(corpus_module, ["load_corpus"])

    (tmp_path / "doc_a.txt").write_text(
        "cache text cache text image",
        encoding="utf-8",
    )
    (tmp_path / "doc_b.txt").write_text(
        "image context text cache",
        encoding="utf-8",
    )

    chunks = call_with_supported_args(
        load_corpus,
        corpus_dir=str(tmp_path),
        chunk_size_tokens=2,
        config=make_rag_config(str(tmp_path), chunk_size_tokens=2),
    )

    assert chunks, "Expected the corpus loader to return at least one chunk."
    first_chunk = chunks[0]
    assert get_value(first_chunk, "chunk_id", "id")
    assert get_value(first_chunk, "text", "content")
    assert get_value(first_chunk, "source", "source_path", "path")


def test_retriever_returns_stable_top_k_results_with_injected_embedder(tmp_path: Path) -> None:
    corpus_module = require_phase2_module("src.rag.corpus")
    retriever_module = require_phase2_module("src.rag.retriever")

    corpus_chunk = require_any_attr(corpus_module, ["CorpusChunk"])
    retriever_cls = require_any_attr(retriever_module, ["Retriever", "InMemoryRetriever"])

    image_path = tmp_path / "vision.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    chunks = [
        corpus_chunk(
            chunk_id="text-cache",
            text="cache cache text",
            source="cache.txt",
            modality="text",
        ),
        corpus_chunk(
            chunk_id="text-mixed",
            text="cache image text",
            source="mixed.txt",
            modality="text",
        ),
        corpus_chunk(
            chunk_id="image-red",
            text="image evidence from the document",
            source="vision.txt#page=1",
            modality="image",
            image_path=str(image_path),
        ),
    ]
    embedder = FakeEmbedder()
    retriever = instantiate_with_supported_args(
        retriever_cls,
        chunks=chunks,
        corpus_chunks=chunks,
        embedder=embedder,
        embedding_backend=embedder,
        config=make_rag_config(str(tmp_path), chunk_size_tokens=8, top_k=2),
    )

    retrieve = require_any_attr(retriever, ["retrieve", "search", "__call__"])
    results_a = call_with_supported_args(retrieve, query="image question", top_k=2)
    results_b = call_with_supported_args(retrieve, query="image question", top_k=2)

    assert len(results_a) == 3
    assert len(results_b) == 3
    assert [
        get_value(result, "chunk_id", "id") for result in results_a
    ] == [
        get_value(result, "chunk_id", "id") for result in results_b
    ]
    assert {get_value(result, "modality", default="text") for result in results_a} == {"text", "image"}


def test_corpus_loader_preserves_dataset_images_as_retrievable_image_chunks(tmp_path: Path) -> None:
    corpus_module = require_phase2_module("src.rag.corpus")
    load_corpus = require_any_attr(corpus_module, ["load_corpus"])

    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "corpus_id": ["sample-1"],
            "doc_id": ["doc-1"],
            "markdown": ["cache architecture diagram"],
            "page_number_in_doc": [0],
            "image": [Image.new("RGB", (4, 4), color=(255, 0, 0))],
        }
    )
    dataset.save_to_disk(str(tmp_path))

    chunks = call_with_supported_args(
        load_corpus,
        corpus_dir=str(tmp_path),
        chunk_size_tokens=8,
        config=make_rag_config(str(tmp_path), chunk_size_tokens=8),
    )

    modalities = [get_value(chunk, "modality", default="text") for chunk in chunks]
    assert "text" in modalities
    assert "image" in modalities

    image_chunks = [chunk for chunk in chunks if get_value(chunk, "modality", default="text") == "image"]
    assert image_chunks
    assert Path(get_value(image_chunks[0], "image_path")).is_file()


def test_open_clip_embedder_maps_config_alias_to_open_clip_model_pair() -> None:
    retriever_module = require_phase2_module("src.rag.retriever")
    embedder_cls = require_any_attr(retriever_module, ["OpenCLIPTextEmbedder"])

    embedder = embedder_cls("openai/clip-vit-large-patch14")

    assert getattr(embedder, "_open_clip_model_name") == "ViT-L-14"
    assert getattr(embedder, "_open_clip_pretrained") == "openai"
