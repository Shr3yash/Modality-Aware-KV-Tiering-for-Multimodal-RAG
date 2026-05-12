import base64
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from rag.config import RAGConfig
from rag.prompt_builder import build_prompt
from rag.pruner import RetrievalPruner
from rag.retriever import QuoteRetriever

def iter_jpgs(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        # deterministic order: 1.jpg, 2.jpg, 10.jpg, ...
        def key(x: Path):
            try:
                return (0, int(x.stem))
            except ValueError:
                return (1, x.name)
        return sorted(p.glob("*.jpg"), key=key)
    return []

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@dataclass(init=False)
class CachedPrunedImage:
    tag: List[float]
    tag_hash: str
    pruned_img_path: str

    def __init__(
        self,
        tag: List[float],
        tag_hash: str,
        pruned_img_path: str,
        legacy_probe_text: str | None = None,
        **_ignored: object,
    ):
        _ = legacy_probe_text
        self.tag = tag
        self.tag_hash = tag_hash
        self.pruned_img_path = pruned_img_path

class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.image_prune_cache_path = self._image_prune_cache_path()
        self.image_prune_cache: Dict[str, List[CachedPrunedImage]] = (
            self._load_image_prune_cache()
        )
        self.retriever = QuoteRetriever(
            text_model_name=cfg.text_embedding_model,
            image_model_name=cfg.image_embedding_model,
            device=cfg.retrieval_device,
            retrieval_mode=getattr(cfg, "retrieval_mode", "dense"),
            bm25_weight=getattr(cfg, "bm25_weight", 1.0),
            rrf_k=getattr(cfg, "rrf_k", 60),
        )
        self.pruner = RetrievalPruner(
            mode=cfg.pruning_mode,
            keep_ratio=cfg.pruning_keep_ratio,
            percentile_ratio = cfg.pruning_percentile_ratio,
            image_model_name=cfg.image_embedding_model,
            device=cfg.retrieval_device,
            patch_grid_rows=cfg.patch_grid_rows,
            patch_grid_cols=cfg.patch_grid_cols,
            min_visual_tokens=cfg.min_visual_tokens,
            montage_tile_size=cfg.montage_tile_size,
            output_dir=cfg.pruned_image_dir,
        )
        self.client = OpenAI(base_url=cfg.vlm_api_base, api_key="EMPTY")
    
    def _tag_hash(self, tag: List[float]) -> str:
        quantized = ",".join(f"{float(v):.6f}" for v in tag)
        return hashlib.sha256(quantized.encode("utf-8")).hexdigest()[:12]

    def _is_vector_tag(self, tag: object) -> bool:
        return isinstance(tag, list) and bool(tag) and all(
            isinstance(value, (float, int)) for value in tag
        )

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return -1.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return -1.0
        return dot / (left_norm * right_norm)

    def _is_image_prune_cache_hit(self, q: Dict) -> bool:
        """This is also a fuck needed to be handled"""
        cache_meta = q.get("image_prune_cache")
        return isinstance(cache_meta, dict) and bool(cache_meta.get("hit"))

    def _image_cache_id(self, q: Dict) -> str:
        existing = q.get("image_cache_id")
        if existing:
            return str(existing)

        img_path = q.get("local_img_path")
        if img_path:
            stem = Path(str(img_path)).stem
            if stem:
                return stem

        quote_id = q.get("quote_id")
        return str(quote_id or "")

    def _with_tag_hash(self, retrieval: Dict) -> Dict:
        selected_images = []
        tag_hashes = []
        for idx, q in enumerate(retrieval.get("selected_img_quotes", [])):
            tag = q["tag"] if self._is_vector_tag(q["tag"]) else []
            tag_hash = self._tag_hash(tag) if tag else ""
            tag_hashes.append(tag_hash)
            selected_images.append(
                {
                    **q,
                    "tag_hash": tag_hash,
                    "image_cache_id": self._image_cache_id(q),
                }
            )
        return {
            **retrieval,
            "selected_img_quotes": selected_images,
        }

    def _image_prune_cache_path(self) -> Path:
        configured = getattr(self.cfg, "image_prune_cache_path", None)
        if configured:
            return Path(configured)
        return Path(self.cfg.output_dir) / "image_prune_cache.json"

    def _load_image_prune_cache(self) -> Dict[str, List[CachedPrunedImage]]:
        if not getattr(self.cfg, "image_prune_cache_enabled", True):
            return {}
        if not self.image_prune_cache_path.exists():
            return {}
        with self.image_prune_cache_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw_entries = raw.get("entries", raw) if isinstance(raw, dict) else {}
        if not isinstance(raw_entries, dict):
            return {}
        cache: Dict[str, List[CachedPrunedImage]] = {}
        for image_id, entries in raw_entries.items():
            if not isinstance(image_id, str) or not isinstance(entries, list):
                continue
            parsed_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                tag = entry.get("tag")
                tag_hash = entry.get("tag_hash")
                pruned_img_path = entry.get("pruned_img_path")
                if (
                    not self._is_vector_tag(tag)
                    or not isinstance(tag_hash, str)
                    or not isinstance(pruned_img_path, str)
                ):
                    continue
                parsed_entries.append(
                    CachedPrunedImage(
                        tag=[float(value) for value in tag],
                        tag_hash=tag_hash,
                        pruned_img_path=pruned_img_path,
                    )
                )
            if parsed_entries:
                cache[image_id] = parsed_entries
        return cache

    def _save_image_prune_cache(self) -> None:
        if not getattr(self.cfg, "image_prune_cache_enabled", True):
            return
        self.image_prune_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": {
                image_id: [asdict(entry) for entry in entries]
                for image_id, entries in self.image_prune_cache.items()
            },
        }
        self.image_prune_cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _split_cached_images(self, retrieval: Dict) -> tuple[List[Dict], Dict[str, int]]:
        stats = {
            "image_prune_cache_hits": 0,
            "image_prune_cache_misses": 0,
        }
        if not getattr(self.cfg, "image_prune_cache_enabled", True):
            return [], stats
        remaining_images = []
        cached_images = []
        for q in retrieval.get("selected_img_quotes", []):
            image_id = q.get("image_cache_id") or self._image_cache_id(q)
            tag = q.get("tag") or []
            entries = self.image_prune_cache.get(image_id, [])
            if not tag or not image_id or not entries:
                stats["image_prune_cache_misses"] += 1
                remaining_images.append(q)
                continue
            scored = [
                (self._cosine_similarity(tag, entry.tag), entry)
                for entry in entries
            ]
            best_similarity, best_entry = max(scored, key=lambda item: item[0]) # what if there are multiple combined?
            if best_similarity < self.cfg.image_prune_cache_similarity_threshold:
                stats["image_prune_cache_misses"] += 1
                remaining_images.append(q)
                continue
            if not Path(best_entry.pruned_img_path).exists():
                stats["image_prune_cache_misses"] += 1
                remaining_images.append(q)
                continue
            cached_q = {
                **q,
                "local_img_path": best_entry.pruned_img_path,
                "tag_hash": best_entry.tag_hash,
                "cache_similarity": float(best_similarity),
            }
            cached_images.append(cached_q)
            stats["image_prune_cache_hits"] += 1
        retrieval["selected_img_quotes"] = remaining_images
        return cached_images, stats

    def _store_pruned_images_in_cache(
        self,
        pruned_img_quotes: List[Dict],
    ) -> None:
        if not getattr(self.cfg, "image_prune_cache_enabled", True):
            return
        changed = False
        for q in pruned_img_quotes:
            image_id = q.get("image_cache_id") or self._image_cache_id(q)
            pruned_img_path = q.get("local_img_path")
            tag = q.get("tag") or []
            tag_hash = q.get("tag_hash") or ""
            if (not image_id or not pruned_img_path
                or not Path(pruned_img_path).exists()
                or not tag or not tag_hash
            ):
                continue
            entry = CachedPrunedImage(
                tag=list(tag),
                tag_hash=tag_hash,
                pruned_img_path=str(pruned_img_path),
            )
            entries = self.image_prune_cache.setdefault(image_id, [])
            for i, old_entry in enumerate(entries):
                if old_entry.tag_hash == tag_hash:
                    entries[i] = entry
                    changed = True
                    break
            else:
                entries.append(entry)
                changed = True
        if changed:
            self._save_image_prune_cache()

    def run_one(self, example: Dict) -> Dict:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            example,
            text_top_k=self.cfg.text_top_k,
            image_top_k=self.cfg.image_top_k,
        )
        retrieval = self._with_tag_hash(retrieval)
        cached_img_quotes, cache_stats = self._split_cached_images(retrieval)
        t1 = time.perf_counter()

        pruned_retrieval = self.pruner.apply(example["question"], retrieval)

        prompt = build_prompt(example["question"], pruned_retrieval, cached_img_quotes)

        content = [{"type": "text", "text": prompt}]

        for q in pruned_retrieval["selected_img_quotes"]:
            path = q.get("local_img_path")
            if ".jpg" in path:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"},
                })
            else:
                for img_path in iter_jpgs(path):
                    if img_path.exists():
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                        })

        for q in cached_img_quotes:
            path = q.get("local_img_path")
            if ".jpg" in path:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"},
                })
            else:
                for img_path in iter_jpgs(path):
                    if img_path.exists():
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                        })

        t2 = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.cfg.vlm_model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.cfg.temperature,
            stream=True,
        )
        first_token_time = None
        pieces = []

        for chunk in stream:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                pieces.append(text)

        t3 = time.perf_counter()

        pred = "".join(pieces)

        retrieved_ids = [
            q["quote_id"] for q in pruned_retrieval["selected_text_quotes"]
        ] + [
            q["quote_id"] for q in pruned_retrieval["selected_img_quotes"]
        ] + [
            q["quote_id"] for q in cached_img_quotes
        ]

        self._store_pruned_images_in_cache(
            pruned_img_quotes=[q for q in pruned_retrieval["selected_img_quotes"]],
        )

        ttft = None if first_token_time is None else (first_token_time - t2)

        visual_pruning_meta = [
            {
                "quote_id": q.get("quote_id"),
                "visual_pruning": q.get("visual_pruning"),
            }
            for q in pruned_retrieval["selected_img_quotes"]
            if q.get("visual_pruning") is not None
        ]
        returned_selected_img_quotes = []
        for q in pruned_retrieval["selected_img_quotes"]:
            new_q = dict()
            new_q["image_cache_id"] = q["image_cache_id"]
            new_q["tag_hash"] = q["tag_hash"]
            q_pruning = q.get("visual_pruning")
            new_q["pruning_mode"] = self.cfg.pruning_mode
            new_q["tokens_before"] = q_pruning.get("tokens_before", -1) if q_pruning else -1
            new_q["tokens_after"] = q_pruning.get("tokens_after", -1) if q_pruning else -1
            returned_selected_img_quotes.append(new_q)
        for q in cached_img_quotes:
            new_q = dict()
            new_q["image_cache_id"] = q["image_cache_id"]
            new_q["tag_hash"] = q["tag_hash"]
            new_q["pruning_mode"] = "from_cache"
            new_q["tokens_before"] = -1
            new_q["tokens_after"] = -1
            returned_selected_img_quotes.append(new_q)

        return {
            "q_id": example["q_id"], # query id
            "question": example["question"],
            "gold_answer": example["answer"],
            "pred_answer": pred,
            "gold_quotes": example["gold_quotes"],
            "retrieved_quote_ids": retrieved_ids,
            "selected_text_quotes": pruned_retrieval["selected_text_quotes"],
            "selected_img_quotes": returned_selected_img_quotes,
            # **cache_stats,
            "pruning": pruned_retrieval["pruning"],
            "timing": {
                "retrieval_sec": t1 - t0,
                "request_build_sec": t2 - t1,
                "ttft_sec": ttft,
                "generation_sec": t3 - t2,
                "total_sec": t3 - t0,
            }
        }
