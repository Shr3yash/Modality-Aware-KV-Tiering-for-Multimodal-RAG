import re
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised only when dep missing
    BM25Okapi = None


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _rrf_fuse(rank_lists: List[List[int]], weights: List[float], k: int, top_k: int) -> List[int]:
    """Reciprocal Rank Fusion over multiple ranked id-lists.

    rank_lists[i] is a list of candidate indices ordered best-first.
    Returns the top_k fused indices."""
    scores: Dict[int, float] = {}
    for ranks, w in zip(rank_lists, weights):
        if w == 0.0:
            continue
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + w / (k + r + 1)
    if not scores:
        return []
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [idx for idx, _ in fused[:top_k]]


def _clip_features_to_tensor(raw: object) -> torch.Tensor:
    """Transformers 5.x returns BaseModelOutputWithPooling from get_*_features; older stacks returned a tensor."""
    if isinstance(raw, torch.Tensor):
        return raw
    po = getattr(raw, "pooler_output", None)
    if po is not None:
        return po
    raise TypeError(f"Unexpected CLIP feature output type: {type(raw)}")


class QuoteRetriever:
    def __init__(
        self,
        text_model_name: str,
        image_model_name: str,
        device: str = "cuda",
        retrieval_mode: str = "dense",
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.text_model = SentenceTransformer(text_model_name)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.clip_model.eval()

        if retrieval_mode not in ("dense", "hybrid"):
            raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")
        if retrieval_mode == "hybrid" and BM25Okapi is None:
            raise ImportError(
                "retrieval_mode='hybrid' requires the 'rank_bm25' package. "
                "Install via: pip install rank_bm25"
            )
        self.retrieval_mode = retrieval_mode
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def _encode_text_quotes(self, texts: List[str]) -> np.ndarray:
        return self.text_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def _encode_clip_text(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.clip_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            raw_feats = self.clip_model.get_text_features(**inputs)
            feats = _clip_features_to_tensor(raw_feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def _encode_clip_images(self, image_paths: List[str]) -> np.ndarray:
        images = []
        valid_idx = []

        for i, p in enumerate(image_paths):
            path = Path(p)
            if not path.exists():
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_idx.append(i)
            except Exception:
                continue

        if not images:
            return np.zeros((0, self.clip_model.config.projection_dim), dtype="float32"), []

        with torch.no_grad():
            inputs = self.clip_processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            raw_feats = self.clip_model.get_image_features(**inputs)
            feats = _clip_features_to_tensor(raw_feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype("float32"), valid_idx

    def _topk(self, q_emb: np.ndarray, c_embs: np.ndarray, k: int) -> List[int]:
        if len(c_embs) == 0:
            return []
        sims = c_embs @ q_emb
        order = np.argsort(-sims)
        return order[:k].tolist()

    def _bm25_rank(self, query: str, corpus: List[str]) -> List[int]:
        """Return candidate indices ranked by BM25 score, best-first.

        Returns empty list if the dep is missing or corpus is empty so callers
        can fall back gracefully."""
        if BM25Okapi is None or not corpus:
            return []
        tokenized_corpus = [_bm25_tokenize(t) for t in corpus]
        if not any(tokenized_corpus):
            return []
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_bm25_tokenize(query))
        return np.argsort(-scores).tolist()

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def _image_conditioned_clip_tag(
        self,
        query_emb: np.ndarray,
        img_emb: np.ndarray,
    ) -> List[float]:
        image_context = self._normalize_vector(img_emb)
        tag = self._normalize_vector(query_emb * image_context)
        if np.linalg.norm(tag) == 0.0:
            tag = image_context
        return tag.astype(float).tolist()

    def retrieve(
        self,
        example: Dict,
        text_top_k: int = 4,
        image_top_k: int = 2,
        variants: Optional[List[str]] = None,
    ) -> Dict:
        query = example["question"]
        # variants[0] is the principal query; preserves single-query semantics
        # when no rewriter is configured.
        if not variants:
            variants = [query]

        text_quotes = example["text_quotes"]
        img_quotes = example["img_quotes"]

        # 1) text-to-text retrieval (multi-variant + optional BM25, fused via RRF)
        selected_texts = []
        if text_quotes:
            text_corpus = [q.get("text", "") for q in text_quotes]
            text_embs = self._encode_text_quotes(text_corpus)
            k_text = min(text_top_k, len(text_quotes))

            rank_lists: List[List[int]] = []
            weights: List[float] = []
            for v in variants:
                v_emb = self._encode_text_quotes([v])[0]
                rank_lists.append(self._topk(v_emb, text_embs, len(text_quotes)))
                weights.append(1.0)
            if self.retrieval_mode == "hybrid":
                for v in variants:
                    bm25_full = self._bm25_rank(v, text_corpus)
                    if bm25_full:
                        rank_lists.append(bm25_full)
                        weights.append(self.bm25_weight)

            if len(rank_lists) == 1:
                # Identical to legacy dense top-k: avoid RRF round-off for the default path.
                text_idx = rank_lists[0][:k_text]
            else:
                text_idx = _rrf_fuse(rank_lists, weights, self.rrf_k, k_text)
            selected_texts = [text_quotes[i] for i in text_idx]

        # 2) text-to-image retrieval with CLIP (multi-variant fused via RRF)
        selected_images = []
        if img_quotes:
            # Principal query embedding drives the per-image conditioning tag so the
            # downstream image-prune cache keys stay stable across rewriter modes.
            clip_query_embs = self._encode_clip_text(variants)
            principal_clip_emb = clip_query_embs[0]
            image_paths = [q.get("local_img_path", "") for q in img_quotes]
            img_embs, valid_idx = self._encode_clip_images(image_paths)

            if len(valid_idx) > 0 and len(img_embs) > 0:
                k_img = min(image_top_k, len(valid_idx))
                if len(clip_query_embs) == 1:
                    local_topk = self._topk(principal_clip_emb, img_embs, k_img)
                else:
                    img_rank_lists = [
                        self._topk(q_emb, img_embs, len(img_embs))
                        for q_emb in clip_query_embs
                    ]
                    local_topk = _rrf_fuse(
                        img_rank_lists,
                        [1.0] * len(img_rank_lists),
                        self.rrf_k,
                        k_img,
                    )
                for i in local_topk:
                    item = dict(img_quotes[valid_idx[i]])
                    item["tag"] = self._image_conditioned_clip_tag(
                        principal_clip_emb, img_embs[i]
                    )
                    selected_images.append(item)

        return {
            "selected_text_quotes": selected_texts,
            "selected_img_quotes": selected_images,
        }
