"""RAG retrieval pipeline supporting text and image chunks.

Indexes a corpus directory with FAISS + CLIP embeddings, retrieves top-k
chunks per query, and applies visual_retrieval_boost scoring for modality
balance.  Falls back to deterministic hash-based embeddings when CLIP is
unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import structlog

from src.cache.kv_block import Modality
from src.utils.config import Config

logger = structlog.get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}

_EMBED_DIM = 768
_DEFAULT_IMAGE_TOKENS = 576  # 24x24 patch grid typical for VLMs


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    modality: Modality
    token_ids: List[int]
    score: float
    source_doc: str
    image_path: Optional[str] = None


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline with FAISS indexing."""

    def __init__(self, config: Config) -> None:
        self.rag_config = config.rag
        self.model_config = config.model
        self._index = None
        self._chunks: List[RetrievedChunk] = []
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load embedding model and index the corpus."""
        self._load_embedder()
        corpus_dir = Path(self.rag_config.corpus_dir)
        if corpus_dir.exists():
            self._index_corpus(corpus_dir)
        else:
            logger.warning("corpus_dir_missing", path=str(corpus_dir))
        self._initialized = True

    def _load_embedder(self) -> None:
        try:
            import open_clip

            model_name = self.rag_config.embedding_model.split("/")[-1]
            self._clip_model, _, self._clip_preprocess = (
                open_clip.create_model_and_transforms(model_name, pretrained="openai")
            )
            self._clip_tokenizer = open_clip.get_tokenizer(model_name)
            self._clip_model.eval()
            logger.info("clip_model_loaded", model=model_name)
        except Exception as e:
            logger.warning("clip_load_failed_using_fallback", error=str(e))
            self._clip_model = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_corpus(self, corpus_dir: Path) -> None:
        import faiss

        chunks: List[RetrievedChunk] = []
        for fpath in sorted(corpus_dir.rglob("*")):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if ext in TEXT_EXTENSIONS:
                chunks.extend(self._chunk_text_file(fpath))
            elif ext in IMAGE_EXTENSIONS:
                chunks.append(self._create_image_chunk(fpath))

        if not chunks:
            logger.warning("no_documents_found", corpus_dir=str(corpus_dir))
            return

        embeddings = [self._embed_chunk(c) for c in chunks]
        emb_matrix = np.vstack(embeddings).astype("float32")
        faiss.normalize_L2(emb_matrix)

        dim = emb_matrix.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb_matrix)
        self._chunks = chunks
        logger.info("corpus_indexed", num_chunks=len(chunks), dim=dim)

    def _chunk_text_file(self, path: Path) -> List[RetrievedChunk]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        words = text.split()
        cs = self.rag_config.chunk_size_tokens
        chunks: List[RetrievedChunk] = []
        for i in range(0, len(words), cs):
            segment = " ".join(words[i : i + cs])
            token_ids = list(range(i, min(i + cs, len(words))))
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{path.stem}_chunk{i // cs}",
                    content=segment,
                    modality=Modality.TEXT,
                    token_ids=token_ids,
                    score=0.0,
                    source_doc=str(path),
                )
            )
        return chunks

    def _create_image_chunk(self, path: Path) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"img_{path.stem}",
            content=f"[Image: {path.name}]",
            modality=Modality.VISUAL,
            token_ids=list(range(_DEFAULT_IMAGE_TOKENS)),
            score=0.0,
            source_doc=str(path),
            image_path=str(path),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_chunk(self, chunk: RetrievedChunk) -> np.ndarray:
        if self._clip_model is not None:
            return self._embed_with_clip(chunk)
        return self._embed_with_hash(chunk.content)

    def _embed_with_clip(self, chunk: RetrievedChunk) -> np.ndarray:
        import torch

        with torch.no_grad():
            if chunk.modality == Modality.VISUAL and chunk.image_path:
                from PIL import Image

                img = Image.open(chunk.image_path).convert("RGB")
                img_tensor = self._clip_preprocess(img).unsqueeze(0)
                features = self._clip_model.encode_image(img_tensor)
            else:
                tokens = self._clip_tokenizer([chunk.content])
                features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()

    def _embed_with_hash(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding fallback (no model needed)."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        emb = rng.randn(_EMBED_DIM).astype("float32")
        emb /= np.linalg.norm(emb) + 1e-8
        return emb.reshape(1, -1)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._clip_model is not None:
            import torch

            with torch.no_grad():
                tokens = self._clip_tokenizer([query])
                features = self._clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype("float32")
        return self._embed_with_hash(query)

    # ------------------------------------------------------------------
    # Retrieval (Task 5: visual_retrieval_boost + force_visual)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        image_path: Optional[str] = None,
        top_k: int = 5,
        force_visual: bool = False,
    ) -> List[RetrievedChunk]:
        if self._index is None or not self._chunks:
            logger.warning("index_not_available")
            return []

        import faiss

        q_emb = self._embed_query(query).astype("float32")
        faiss.normalize_L2(q_emb)

        # Over-retrieve so we have room for re-ranking and visual injection.
        k = min(top_k * 3, len(self._chunks))
        scores, indices = self._index.search(q_emb, k)

        candidates: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            adjusted_score = float(score)
            if chunk.modality == Modality.VISUAL:
                adjusted_score *= self.rag_config.visual_retrieval_boost
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    modality=chunk.modality,
                    token_ids=chunk.token_ids,
                    score=adjusted_score,
                    source_doc=chunk.source_doc,
                    image_path=chunk.image_path,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        selected = candidates[:top_k]

        # --force-visual: guarantee at least one image chunk in top-k
        if force_visual:
            has_visual = any(c.modality == Modality.VISUAL for c in selected)
            if not has_visual:
                visual_candidates = [
                    c for c in candidates if c.modality == Modality.VISUAL
                ]
                if visual_candidates:
                    selected[-1] = visual_candidates[0]
                    logger.info(
                        "force_visual_injected",
                        chunk_id=visual_candidates[0].chunk_id,
                    )

        visual_count = sum(1 for c in selected if c.modality == Modality.VISUAL)
        logger.info(
            "retrieval_complete",
            total=len(selected),
            text=len(selected) - visual_count,
            visual=visual_count,
        )
        return selected

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        image_path: Optional[str] = None,
    ) -> str:
        context_parts: List[str] = []
        for i, chunk in enumerate(chunks):
            tag = "image" if chunk.modality == Modality.VISUAL else "text"
            context_parts.append(f"[Context {i + 1} ({tag})]: {chunk.content}")

        context = "\n\n".join(context_parts)
        return (
            "Use the following context to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    # ------------------------------------------------------------------
    # Memory management (Task 4)
    # ------------------------------------------------------------------

    def offload_encoders_to_cpu(self) -> None:
        """Move CLIP/ColPali encoders to CPU to free GPU memory."""
        if self._clip_model is not None:
            self._clip_model = self._clip_model.cpu()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("clip_model_offloaded_to_cpu")

    # ------------------------------------------------------------------
    # Synthetic data loading (for benchmarks)
    # ------------------------------------------------------------------

    def load_synthetic_chunks(self, chunks: List[RetrievedChunk]) -> None:
        """Load pre-built chunks directly (skips corpus indexing)."""
        import faiss

        self._chunks = chunks
        embeddings = [self._embed_chunk(c) for c in chunks]
        if not embeddings:
            return
        emb_matrix = np.vstack(embeddings).astype("float32")
        faiss.normalize_L2(emb_matrix)
        dim = emb_matrix.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb_matrix)
        self._initialized = True
        logger.info("synthetic_chunks_loaded", num_chunks=len(chunks))
