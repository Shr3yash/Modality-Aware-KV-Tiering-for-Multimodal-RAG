"""Foreground-aware visual token pruning using MobileSAM or center-crop heuristic.

Background image patches identified by SAM segmentation are excluded from the
KV cache at prefill time, reducing the number of cached visual tokens by an
expected 40-60%.  Falls back to a simple center-crop heuristic when MobileSAM
is not available (GPU-memory-constrained environments).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import structlog

from src.cache.kv_block import KVBlock

logger = structlog.get_logger(__name__)


class VisualTokenPruner:
    """Prunes background visual tokens to reduce cached KV entries."""

    def __init__(
        self,
        backend: str = "mobilesam",
        foreground_threshold: float = 0.3,
    ) -> None:
        self.backend = backend
        self.foreground_threshold = foreground_threshold
        self._sam_model = None
        self._mask_generator = None

        if backend == "mobilesam":
            self._try_load_mobilesam()

    # ------------------------------------------------------------------
    # MobileSAM bootstrap
    # ------------------------------------------------------------------

    def _try_load_mobilesam(self) -> None:
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            import torch

            checkpoint = self._find_mobilesam_checkpoint()
            if checkpoint is None:
                logger.warning("mobilesam_checkpoint_not_found_falling_back")
                self.backend = "center_crop"
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
            sam.to(device)
            sam.eval()
            self._mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.8,
                min_mask_region_area=100,
            )
            self._sam_model = sam
            logger.info("mobilesam_loaded", device=device)
        except ImportError:
            logger.warning("mobile_sam_not_installed_falling_back")
            self.backend = "center_crop"
        except Exception as e:
            logger.warning("mobilesam_init_failed", error=str(e))
            self.backend = "center_crop"

    @staticmethod
    def _find_mobilesam_checkpoint() -> Optional[str]:
        candidates = [
            os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt"),
            "weights/mobile_sam.pt",
            "/tmp/mobile_sam.pt",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_foreground_mask(
        self,
        image,  # PIL.Image
        patch_grid: Tuple[int, int],
    ) -> np.ndarray:
        """Compute per-patch foreground mask.

        Args:
            image: PIL Image to analyze.
            patch_grid: (rows, cols) of VLM image patch grid.

        Returns:
            Boolean array of shape (rows * cols,) — True = foreground = keep.
        """
        if self.backend == "mobilesam" and self._mask_generator is not None:
            return self._mobilesam_mask(image, patch_grid)
        return self._center_crop_mask(image, patch_grid)

    def apply_mask_to_kv_block(
        self,
        kv_block: KVBlock,
        mask: np.ndarray,
    ) -> KVBlock:
        """Remove background patch KV entries from a KVBlock in place."""
        import torch

        keep_indices = np.where(mask)[0]
        if len(keep_indices) == 0 or len(keep_indices) == kv_block.num_tokens:
            return kv_block

        idx = torch.tensor(keep_indices, dtype=torch.long)

        if kv_block.k_cache is not None and kv_block.k_cache.shape[2] == len(mask):
            kv_block.k_cache = kv_block.k_cache[:, :, idx, :]
        if kv_block.v_cache is not None and kv_block.v_cache.shape[2] == len(mask):
            kv_block.v_cache = kv_block.v_cache[:, :, idx, :]

        kv_block.token_ids = tuple(
            tid
            for i, tid in enumerate(kv_block.token_ids)
            if i < len(mask) and mask[i]
        )
        kv_block.num_tokens = len(keep_indices)
        return kv_block

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _mobilesam_mask(
        self,
        image,
        patch_grid: Tuple[int, int],
    ) -> np.ndarray:
        rows, cols = patch_grid
        num_patches = rows * cols

        img_np = np.array(image)
        masks = self._mask_generator.generate(img_np)

        if not masks:
            return np.ones(num_patches, dtype=bool)

        h, w = img_np.shape[:2]
        total_area = h * w
        combined = np.zeros((h, w), dtype=np.float64)

        for m in sorted(masks, key=lambda m: m["predicted_iou"], reverse=True):
            if m["area"] / total_area >= 0.8:
                continue  # skip whole-image masks
            seg = m["segmentation"].astype(np.float64)
            combined = np.maximum(combined, seg * m["predicted_iou"])

        if combined.max() < 1e-6:
            return np.ones(num_patches, dtype=bool)

        patch_h, patch_w = h / rows, w / cols
        patch_mask = np.zeros(num_patches, dtype=bool)
        for r in range(rows):
            for c in range(cols):
                y0, y1 = int(r * patch_h), int((r + 1) * patch_h)
                x0, x1 = int(c * patch_w), int((c + 1) * patch_w)
                iou = combined[y0:y1, x0:x1].mean()
                patch_mask[r * cols + c] = iou >= self.foreground_threshold

        kept = int(patch_mask.sum())
        logger.debug(
            "mobilesam_mask",
            total=num_patches,
            kept=kept,
            ratio=f"{kept / num_patches:.2f}",
        )
        return patch_mask

    def _center_crop_mask(
        self,
        image,  # unused but kept for API uniformity
        patch_grid: Tuple[int, int],
    ) -> np.ndarray:
        """Keep patches near the center of the image."""
        rows, cols = patch_grid
        num_patches = rows * cols
        mask = np.zeros(num_patches, dtype=bool)

        center_r, center_c = rows / 2, cols / 2
        max_dist = ((rows / 2) ** 2 + (cols / 2) ** 2) ** 0.5

        for r in range(rows):
            for c in range(cols):
                dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                if dist / max_dist <= (1.0 - self.foreground_threshold):
                    mask[r * cols + c] = True

        kept = int(mask.sum())
        logger.debug(
            "center_crop_mask",
            total=num_patches,
            kept=kept,
            ratio=f"{kept / num_patches:.2f}",
        )
        return mask
