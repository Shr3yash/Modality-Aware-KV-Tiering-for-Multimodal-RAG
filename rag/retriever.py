from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

class QuoteRetriever:
    def __init__(
        self,
        text_model_name: str,
        image_model_name: str,
        device: str = "cuda",
    ):
        self.text_model = SentenceTransformer(text_model_name)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.clip_model.eval()

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
            feats = self.clip_model.get_text_features(**inputs)
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
            feats = self.clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype("float32"), valid_idx

    def _topk(self, q_emb: np.ndarray, c_embs: np.ndarray, k: int) -> List[int]:
        if len(c_embs) == 0:
            return []
        sims = c_embs @ q_emb
        order = np.argsort(-sims)
        return order[:k].tolist()

    def retrieve(self, example: Dict, text_top_k: int = 4, image_top_k: int = 2) -> Dict:
        query = example["question"]

        text_quotes = example["text_quotes"]
        img_quotes = example["img_quotes"]

        # 1) text-to-text retrieval
        selected_texts = []
        if text_quotes:
            text_corpus = [q.get("text", "") for q in text_quotes]
            text_embs = self._encode_text_quotes(text_corpus)
            query_text_emb = self._encode_text_quotes([query])[0]
            text_idx = self._topk(query_text_emb, text_embs, min(text_top_k, len(text_quotes)))
            selected_texts = [text_quotes[i] for i in text_idx]

        # Text-to-image retrieval with CLIP (image paths)
        selected_images = []
        if img_quotes:
            image_paths = [q.get("local_img_path", "") for q in img_quotes]
            img_embs, valid_idx = self._encode_clip_images(image_paths)

            if len(valid_idx) > 0 and len(img_embs) > 0:
                query_img_emb = self._encode_clip_text([query])[0]
                local_topk = self._topk(query_img_emb, img_embs, min(image_top_k, len(valid_idx)))
                selected_images = [img_quotes[valid_idx[i]] for i in local_topk]

        return {
            "selected_text_quotes": selected_texts,
            "selected_img_quotes": selected_images,
        }