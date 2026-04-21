from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class PruningStats:
    mode: str
    text_before: int
    text_after: int
    images_before: int
    images_after: int
    visual_tokens_before: int
    visual_tokens_after: int

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "mode": self.mode,
            "text_before": self.text_before,
            "text_after": self.text_after,
            "images_before": self.images_before,
            "images_after": self.images_after,
            "visual_tokens_before": self.visual_tokens_before,
            "visual_tokens_after": self.visual_tokens_after,
        }


class RetrievalPruner:
    """
    Retrieval-side and image-patch-side pruning.

    Modes:
      - no_pruning
      - uniform_pruning
      - visual_only_pruning
      - visual_patch_pruning
      - model_internal_visual_pruning

    Notes:
      * visual_patch_pruning is server-compatible: it rewrites each selected image into
        a smaller montage of kept patches so the served model sees fewer visual patches.
      * model_internal_visual_pruning cannot directly alter a remote served model from
        query_pipeline.py. Instead, it computes and returns the keep-mask / keep-indices
        that a model-side hook could consume later.
    """

    SUPPORTED_MODES = {
        "no_pruning",
        "uniform_pruning",
        "visual_only_pruning",
        "visual_patch_pruning",
        "model_internal_visual_pruning",
    }

    def __init__(
        self,
        mode: str = "no_pruning",
        keep_ratio: float = 0.5,
        image_model_name: str | None = None,
        device: str = "cuda",
        patch_grid_rows: int = 4,
        patch_grid_cols: int = 4,
        min_visual_tokens: int = 4,
        montage_tile_size: int = 224,
        output_dir: str | Path = "data/mmdocrag/outputs/pruned_images",
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported pruning mode: {mode}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in the range (0, 1].")
        if patch_grid_rows <= 0 or patch_grid_cols <= 0:
            raise ValueError("patch_grid_rows and patch_grid_cols must be positive.")

        self.mode = mode
        self.keep_ratio = keep_ratio
        self.patch_grid_rows = patch_grid_rows
        self.patch_grid_cols = patch_grid_cols
        self.min_visual_tokens = min_visual_tokens
        self.montage_tile_size = montage_tile_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_model_name = image_model_name
        self.clip_processor = None
        self.clip_model = None
        if mode in {"visual_patch_pruning", "model_internal_visual_pruning"}:
            if not image_model_name:
                raise ValueError(
                    "image_model_name is required for visual patch pruning modes."
                )
            self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
            self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
            self.clip_model.eval()

    def apply(self, example: Dict, retrieval: Dict) -> Dict:
        text_quotes = list(retrieval.get("selected_text_quotes", []))
        img_quotes = [dict(q) for q in retrieval.get("selected_img_quotes", [])]

        pruned_texts = text_quotes
        pruned_images = img_quotes
        visual_before = sum(self._estimate_visual_tokens(q) for q in img_quotes)
        visual_after = visual_before

        if self.mode == "uniform_pruning":
            pruned_texts = self._prune_list(text_quotes)
            pruned_images = self._prune_list(img_quotes)
            visual_after = sum(self._estimate_visual_tokens(q) for q in pruned_images)
        elif self.mode == "visual_only_pruning":
            pruned_images = self._prune_list(img_quotes)
            visual_after = sum(self._estimate_visual_tokens(q) for q in pruned_images)
        elif self.mode in {"visual_patch_pruning", "model_internal_visual_pruning"}:
            processed = []
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i = self._patch_prune_image(example, q)
                processed.append(new_q)
                visual_after += after_i
            pruned_images = processed

        stats = PruningStats(
            mode=self.mode,
            text_before=len(text_quotes),
            text_after=len(pruned_texts),
            images_before=len(img_quotes),
            images_after=len(pruned_images),
            visual_tokens_before=visual_before,
            visual_tokens_after=visual_after,
        )

        return {
            "selected_text_quotes": pruned_texts,
            "selected_img_quotes": pruned_images,
            "pruning": stats.to_dict(),
        }

    def _prune_list(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        keep_n = max(1, int(len(items) * self.keep_ratio))
        keep_n = min(keep_n, len(items))
        return items[:keep_n]

    def _estimate_visual_tokens(self, q: Dict) -> int:
        meta = q.get("visual_pruning")
        if isinstance(meta, dict) and "tokens_after" in meta:
            return int(meta["tokens_after"])
        return self.patch_grid_rows * self.patch_grid_cols

    def _patch_prune_image(self, example: Dict, q: Dict) -> Tuple[Dict, int, int]:
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        after = before
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "mode": self.mode,
                "skipped": True,
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": after,
            }
            return q, before, after

        image = Image.open(img_path).convert("RGB")
        tiles, boxes = self._extract_grid_tiles(image)
        scores = self._score_tiles(example["question"], tiles)

        keep_n = max(self.min_visual_tokens, int(len(tiles) * self.keep_ratio))
        keep_n = min(max(1, keep_n), len(tiles))
        keep_idx = np.argsort(-scores)[:keep_n].tolist()
        keep_idx.sort()
        after = len(keep_idx)

        q["visual_pruning"] = {
            "mode": self.mode,
            "tokens_before": before,
            "tokens_after": after,
            "grid_rows": self.patch_grid_rows,
            "grid_cols": self.patch_grid_cols,
            "keep_indices": keep_idx,
            "keep_boxes": [boxes[i] for i in keep_idx],
            "scores": [float(scores[i]) for i in keep_idx],
        }

        if self.mode == "visual_patch_pruning":
            kept_tiles = [tiles[i] for i in keep_idx]
            pruned_path = self._save_montage(image_path=Path(img_path), kept_tiles=kept_tiles)
            q["local_img_path"] = str(pruned_path)
            q["visual_pruning"]["rendered_image_path"] = str(pruned_path)
        else:
            q["visual_pruning"]["hook_required"] = True
            q["visual_pruning"]["hook_point"] = (
                "after vision encoder patch embeddings, before projector / multimodal merge"
            )

        return q, before, after

    def _extract_grid_tiles(self, image: Image.Image) -> Tuple[List[Image.Image], List[List[int]]]:
        width, height = image.size
        xs = np.linspace(0, width, self.patch_grid_cols + 1, dtype=int)
        ys = np.linspace(0, height, self.patch_grid_rows + 1, dtype=int)

        tiles: List[Image.Image] = []
        boxes: List[List[int]] = []
        for r in range(self.patch_grid_rows):
            for c in range(self.patch_grid_cols):
                left, right = int(xs[c]), int(xs[c + 1])
                top, bottom = int(ys[r]), int(ys[r + 1])
                box = [left, top, right, bottom]
                tile = image.crop(box)
                tiles.append(tile)
                boxes.append(box)
        return tiles, boxes

    def _score_tiles(self, query: str, tiles: List[Image.Image]) -> np.ndarray:
        assert self.clip_model is not None and self.clip_processor is not None
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=[query], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_feats = self.clip_model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            image_inputs = self.clip_processor(images=tiles, return_tensors="pt", padding=True).to(self.device)
            img_feats = self.clip_model.get_image_features(**image_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            sims = (img_feats @ text_feats.T).squeeze(-1)
        return sims.detach().float().cpu().numpy()

    def _save_montage(self, image_path: Path, kept_tiles: List[Image.Image]) -> Path:
        if not kept_tiles:
            raise ValueError("kept_tiles must not be empty.")
        n = len(kept_tiles)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        tile_size = self.montage_tile_size
        canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(255, 255, 255))

        for idx, tile in enumerate(kept_tiles):
            thumb = ImageOps.contain(tile, (tile_size, tile_size))
            x = (idx % cols) * tile_size + (tile_size - thumb.width) // 2
            y = (idx // cols) * tile_size + (tile_size - thumb.height) // 2
            canvas.paste(thumb, (x, y))

        out_name = f"{image_path.stem}_pruned_{self.mode}_{int(self.keep_ratio * 100)}{image_path.suffix}"
        out_path = self.output_dir / out_name
        canvas.save(out_path)
        return out_path
