import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image, ImageDraw
import base64
import io
import os
from typing import Any, Dict, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

IS_TOPK = True # False: Bbox Bbox works worse than topk
IS_CLUSTER = True # False: Safe Crop

# Below Qwen2-VL default chat budget (28*28*1280): CATP uses eager attention plus full
# attentions/hidden_states, so vision patch count must stay small to avoid OOM.
_DEFAULT_CATP_MAX_PIXELS = 28 * 28 * 1280


def _normalize_map(attn_map: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    attn_map = attn_map.astype(np.float32)
    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + eps)

def _patch_importance_to_spatial_map(
    patch_importance: torch.Tensor,
    active_token_indices: torch.Tensor,
    merged_grid_h: int,
    merged_grid_w: int,
) -> np.ndarray:
    """
    Convert 1D visual-token importance into a 2D spatial attention map.

    patch_importance:
        shape = (num_active_tokens,)

    active_token_indices:
        original visual-token indices among all image tokens.
        shape = (num_active_tokens,)
    """
    spatial_map = torch.zeros(
        merged_grid_h * merged_grid_w,
        device=patch_importance.device,
        dtype=patch_importance.dtype,
    )

    spatial_map[active_token_indices] = patch_importance
    spatial_map = spatial_map.reshape(merged_grid_h, merged_grid_w)

    spatial_map = spatial_map.detach().float().cpu().numpy()
    spatial_map = _normalize_map(spatial_map)

    return spatial_map

def collect_spatial_attention_maps(
    outputs,
    input_ids,
    image_token_indices,
    current_active_indices,
    query_start_idx,
    query_end_idx,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    layer_start: int = 6,
    layer_end: int = 14,
):
    """
    Collect query-to-image spatial attention maps from layer_start to layer_end.

    Returns:
        spatial_maps: dict[layer_idx] -> 2D numpy array
    """
    merged_grid_h = grid_h // merge_size
    merged_grid_w = grid_w // merge_size

    spatial_maps = {}

    for layer_idx in range(layer_start, layer_end + 1):
        patch_importance = get_fused_patch_importance(outputs, image_token_indices, current_active_indices,
                                                      query_start_idx, query_end_idx,
                                                      merged_grid_h, merged_grid_w, [layer_idx], "max", "sum")

        spatial_map = _patch_importance_to_spatial_map(
            patch_importance=patch_importance,
            active_token_indices=current_active_indices,
            merged_grid_h=merged_grid_h,
            merged_grid_w=merged_grid_w,
        )

        spatial_maps[layer_idx] = spatial_map

    return spatial_maps

def save_layer_spatial_attention_grid(
    image,
    spatial_maps,
    save_path: str,
    alpha: float = 0.45,
):
    """
    Save one figure containing spatial attention maps from multiple layers.

    image:
        PIL.Image

    spatial_maps:
        dict[layer_idx] -> 2D numpy array
    """
    layer_items = sorted(spatial_maps.items(), key=lambda x: x[0])

    num_maps = len(layer_items)
    cols = 3
    rows = math.ceil(num_maps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    image_np = np.array(image.convert("RGB"))

    for ax, (layer_idx, spatial_map) in zip(axes, layer_items):
        h, w = image_np.shape[:2]

        spatial_tensor = torch.tensor(spatial_map)[None, None, :, :]
        spatial_resized = F.interpolate(
            spatial_tensor,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        spatial_resized = _normalize_map(spatial_resized)

        ax.imshow(image_np)
        ax.imshow(spatial_resized, alpha=alpha)
        ax.set_title(f"Layer {layer_idx}")
        ax.axis("off")

    for ax in axes[num_maps:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def get_fused_patch_importance(
    outputs,
    image_token_indices,
    current_active_indices,
    query_start_idx,
    query_end_idx,
    merged_grid_h: int,
    merged_grid_w: int,
    selected_layers=(10, 11, 12, 13),
    head_reduce="max",   # "max" or "mean"
    layer_reduce="sum", # "mean", "max", or "sum"
):
    active_global_indices = image_token_indices[current_active_indices]
    layer_importances = []

    for layer_idx in selected_layers:
        layer_attn = outputs.attentions[layer_idx].squeeze(0)
        query_to_image_attn = layer_attn[:,query_start_idx:query_end_idx,:,][:, :, active_global_indices]
        head_patch_importance = query_to_image_attn.mean(dim=1)

        if head_reduce == "max":
            patch_importance = head_patch_importance.max(dim=0).values
        elif head_reduce == "mean":
            patch_importance = head_patch_importance.mean(dim=0)
        else:
            raise ValueError(f"Unknown head_reduce: {head_reduce}")

        layer_importances.append(patch_importance.float())

    stacked = torch.stack(layer_importances, dim=0)
    # shape: (num_selected_layers, active_image_tokens)

    if layer_reduce == "mean":
        fused_importance = stacked.mean(dim=0)
    elif layer_reduce == "max":
        fused_importance = stacked.max(dim=0).values
    elif layer_reduce == "sum":
        fused_importance = stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown layer_reduce: {layer_reduce}")
    importance_2d = fused_importance.view(1, 1, merged_grid_h, merged_grid_w)
    # Apply a 3x3 or 5x5 Average Pool with stride 1 to smear the attention into blobs.
    # This connects nearby text tokens and dilutes isolated noise spikes.
    smoothed_2d = F.avg_pool2d(importance_2d, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    # Create a center-weighted mask (1.0 in the center, decaying towards 0.5 at the edges)
    y_coords = torch.linspace(-1, 1, merged_grid_h, device=smoothed_2d.device)
    x_coords = torch.linspace(-1, 1, merged_grid_w, device=smoothed_2d.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    # Calculate distance from center, normalize, and invert
    distance_from_center = torch.sqrt(x_grid**2 + y_grid**2)
    # Tweak 0.5 to be more/less aggressive on edge penalization
    edge_penalty_mask = 1.0 - (distance_from_center * 0.5) 
    edge_penalty_mask = torch.clamp(edge_penalty_mask, min=0.1) # Don't zero out edges completely
    smoothed_2d = smoothed_2d * edge_penalty_mask.view(1, 1, merged_grid_h, merged_grid_w)
    final_fused_importance = smoothed_2d.flatten()

    return final_fused_importance

class Qwen2VLCATPBoundingBoxCropper:
    def __init__(
        self,
        model_id="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        device="cuda",
        catp_max_pixels: int | None = None,
    ):
        self.device = device
        if catp_max_pixels is not None:
            self.catp_max_pixels = int(catp_max_pixels)
        else:
            env_v = os.environ.get("MRAG_CATP_MAX_PIXELS")
            self.catp_max_pixels = int(env_v) if env_v else _DEFAULT_CATP_MAX_PIXELS
        if "gptq" in model_id.lower():
            from optimum.utils import is_gptqmodel_available
            if not is_gptqmodel_available():
                raise RuntimeError(
                    "Loading this GPTQ model requires gptqmodel (used by optimum with transformers). "
                    "Install with: pip install 'gptqmodel>=1.6.0'"
                )
        # Load the quantized Qwen2-VL model and its specific processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        self.model.eval()

    def _diversity_pre_filter(self, hidden_states, similarity_threshold=0.98):
        """
        CATP Stage 1: Unconditional diversity-based pruning.
        Removes visually identical patches before attention scaling.
        """
        norm_states = F.normalize(hidden_states.float(), p=2, dim=1)
        sim_matrix = torch.matmul(norm_states, norm_states.T)

        unique_mask = torch.ones(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device,)

        for i in range(hidden_states.shape[0]):
            if unique_mask[i]:
                duplicates = sim_matrix[i] > similarity_threshold
                duplicates[i] = False
                unique_mask[duplicates] = False

        return unique_mask

    def _attention_entropy(self, scores: torch.Tensor, eps: float = 1e-12) -> float:
        """
        scores: shape (num_image_tokens,)
        """
        probs = scores.float()
        probs = probs / (probs.sum() + eps)
        entropy = -(probs * torch.log(probs + eps)).sum()
        return float(entropy.detach().cpu())

    def _pick_tokens(self, masked_importance, current_active_indices, keep_ratio = 0.5, percentile_ratio = 0.7):
        if IS_TOPK:
            keep_n = max(4, int(masked_importance.numel() * keep_ratio))
            top_idx_local = torch.topk(masked_importance, keep_n).indices
            selected_active_indices = current_active_indices[top_idx_local]
        else:
            threshold = torch.quantile(masked_importance.float(), percentile_ratio,)
            top_idx_local = masked_importance > threshold
            if top_idx_local.sum() == 0: top_idx_local[torch.argmax(masked_importance)] = True
            selected_active_indices = current_active_indices[top_idx_local]
        
        return selected_active_indices

    def _safe_crop(
        self,
        current_active_indices: torch.Tensor,
        selected_active_indices: torch.Tensor,
        merged_grid_h: int,
        merged_grid_w: int,
        image_width: int,
        image_height: int,
    ):
        px_per_grid_x = image_width / merged_grid_w
        px_per_grid_y = image_height / merged_grid_h

        selected_y = selected_active_indices // merged_grid_w
        selected_x = selected_active_indices % merged_grid_w
        min_x, max_x = selected_x.min(), selected_x.max()
        min_y, max_y = selected_y.min(), selected_y.max()
        min_x, max_x = int(min_x.detach().cpu()), int(max_x.detach().cpu())
        min_y, max_y = int(min_y.detach().cpu()), int(max_y.detach().cpu())
        all_active_y = current_active_indices // merged_grid_w
        all_active_x = current_active_indices % merged_grid_w
        bbox_mask = ((all_active_x >= min_x) & (all_active_x <= max_x) & (all_active_y >= min_y) & (all_active_y <= max_y))

        pruned_active_indices = current_active_indices[bbox_mask]
        # Safety fallback
        if pruned_active_indices.numel() == 0:
            pruned_active_indices = selected_active_indices

        x_min_px, x_max_px = int(min_x * px_per_grid_x), int((max_x + 1) * px_per_grid_x)
        y_min_px, y_max_px = int(min_y * px_per_grid_y), int((max_y + 1) * px_per_grid_y)

        x_min_px, x_max_px = max(0, x_min_px), min(image_width, x_max_px)
        y_min_px, y_max_px = max(0, y_min_px), min(image_height, y_max_px)
        return {
            "active_visual_token_indices_after_pruning": pruned_active_indices,
            "clusters": [{
                "cluster_id": 0,
                "grid_bbox": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
                "pixel_bbox": {"x_min": x_min_px, "x_max": x_max_px, "y_min": y_min_px, "y_max": y_max_px},
                "annotation": "Fallback: Part of the original image."
            }],
            "tokens_after_bbox": int(pruned_active_indices.numel()),
        }

    def _cluster_active_patches(self, active_patch_mask: torch.Tensor, max_gap: int = 1,):
        """
        Clusters active patches but prevents "spiderwebbing" by enforcing a minimum
        bounding box density.
        active_patch_mask: bool tensor, shape = (merged_grid_h, merged_grid_w)
        max_gap:
            Allows small gaps between active patches.
            max_gap=1 means patches within a 3x3 neighborhood are connected.
            max_gap=2 means patches within a 5x5 neighborhood are connected.

        Returns:
            list of clusters, each cluster is list of (y, x)
        """
        h, w = active_patch_mask.shape

        visited = torch.zeros_like(active_patch_mask, dtype=torch.bool)
        clusters: list[list[tuple[int, int]]] = []
        active_coords = active_patch_mask.nonzero(as_tuple=False)
        total_cells = h * w
        dominance_cutoff = 0.7 * total_cells

        for coord in active_coords:
            start_y, start_x = int(coord[0]), int(coord[1])

            if visited[start_y, start_x]: continue

            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            cluster: list[tuple[int, int]] = []

            c_min_y, c_max_y = start_y, start_y
            c_min_x, c_max_x = start_x, start_x

            while queue:
                y, x = queue.popleft()
                cluster.append((y, x))

                y0 = max(0, y - max_gap)
                y1 = min(h - 1, y + max_gap)
                x0 = max(0, x - max_gap)
                x1 = min(w - 1, x + max_gap)

                for ny in range(y0, y1 + 1):
                    for nx in range(x0, x1 + 1):
                        if active_patch_mask[ny, nx] and not visited[ny, nx]:
                            new_min_y, new_max_y = min(c_min_y, ny), max(c_max_y, ny)
                            new_min_x, new_max_x = min(c_min_x, nx), max(c_max_x, nx)

                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            c_min_y, c_max_y = new_min_y, new_max_y
                            c_min_x, c_max_x = new_min_x, new_max_x

            has_visited_cells = visited[c_min_y:c_max_y + 1, c_min_x:c_max_x + 1].sum()
            cluster_area = (c_max_y - c_min_y + 1) * (c_max_x - c_min_x + 1)
            cluster_density = len(cluster) / max(1, cluster_area)
            if cluster_density < 0.15:
                continue
            if cluster_area - has_visited_cells < 16 and has_visited_cells / cluster_area > 0.5:
                continue
            visited[c_min_y:c_max_y + 1, c_min_x:c_max_x + 1] = True
            if cluster_area >= dominance_cutoff:
                return [cluster]

            clusters.append(cluster)

        return clusters

    def _cluster_crop(
        self,
        selected_active_indices: torch.Tensor,
        merged_grid_h: int,
        merged_grid_w: int,
        image_width: int,
        image_height: int,
        max_gap: int = 1,
    ):
        device = selected_active_indices.device
        px_per_grid_x = image_width / merged_grid_w
        px_per_grid_y = image_height / merged_grid_h

        active_patch_mask = torch.zeros(merged_grid_h * merged_grid_w, dtype=torch.bool, device=device,)
        active_patch_mask[selected_active_indices] = True
        active_patch_mask = active_patch_mask.reshape(merged_grid_h, merged_grid_w)

        clusters = self._cluster_active_patches(active_patch_mask=active_patch_mask, max_gap=max_gap,)

        cluster_infos = []
        final_active_indices = []
        true_tokens_after = 0

        for cluster_id, cluster in enumerate(clusters, start=1):
            ys, xs = [p[0] for p in cluster], [p[1] for p in cluster]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            x_min_px, x_max_px = int(min_x * px_per_grid_x), int((max_x + 1) * px_per_grid_x)
            y_min_px, y_max_px = int(min_y * px_per_grid_y), int((max_y + 1) * px_per_grid_y)

            x_min_px, x_max_px = max(0, x_min_px), min(image_width, x_max_px)
            y_min_px, y_max_px = max(0, y_min_px), min(image_height, y_max_px)

            cluster_token_indices = []
            cluster_scores = []

            for y, x in cluster:
                token_idx = y * merged_grid_w + x
                cluster_token_indices.append(token_idx)
                final_active_indices.append(token_idx)

            cluster_infos.append({
                "cluster_id": cluster_id,
                # Grid position in visual-token coordinate space
                "grid_bbox": {
                    "min_x": int(min_x), "max_x": int(max_x),
                    "min_y": int(min_y), "max_y": int(max_y),
                },
                "pixel_bbox": {
                    "x_min": int(x_min_px), "x_max": int(x_max_px),
                    "y_min": int(y_min_px), "y_max": int(y_max_px),
                },
                "annotation": (
                    f"Cluster {cluster_id}: position in original image "
                    f"(x_min={x_min_px}, x_max={x_max_px}, "
                    f"y_min={y_min_px}, y_max={y_max_px})"
                ),
            })
            true_tokens_after += int((max_x - min_x + 1) * (max_y - min_y + 1))

        final_active_indices = sorted(set(final_active_indices))
        original_total_tokens = merged_grid_h * merged_grid_w

        if true_tokens_after >= original_total_tokens:
            print(f"[Warning] Sparse Patch Problem detected. Bbox tokens ({true_tokens_after}) >= Original ({original_total_tokens}). Falling back to full image.")
            return {
                "active_visual_token_indices_after_pruning": list(range(original_total_tokens)),
                "clusters": [{
                    "cluster_id": 0,
                    "grid_bbox": {"min_x": 0, "max_x": merged_grid_w - 1, "min_y": 0, "max_y": merged_grid_h - 1},
                    "pixel_bbox": {"x_min": 0, "x_max": image_width, "y_min": 0, "y_max": image_height},
                    "annotation": "Fallback: Full Image due to sparse scattering."
                }],
                "tokens_after_bbox": original_total_tokens,
            }

        return {
            "active_visual_token_indices_after_pruning": final_active_indices,
            "clusters": cluster_infos,
            "tokens_after_bbox": true_tokens_after,
        }

    def _prune_by_attention(
        self,
        patch_importance: torch.Tensor,
        current_active_indices: torch.Tensor,
        hidden_states: torch.Tensor,
        merged_grid_h: int,
        merged_grid_w: int,
        image_width: int,
        image_height: int,
        keep_ratio: float = 0.3,
        percentile_ratio: float = 0.7,
        similarity_threshold: float = 0.98,
        max_gap: int = 1,
    ):

        unique_mask = self._diversity_pre_filter(hidden_states, similarity_threshold)
        masked_importance = patch_importance.clone()
        masked_importance[~unique_mask] = -float('inf')

        selected_active_indices = self._pick_tokens(masked_importance, current_active_indices, keep_ratio, percentile_ratio)

        res = None
        if IS_CLUSTER:
            res = self._cluster_crop(selected_active_indices, merged_grid_h, merged_grid_w, image_width, image_height, max_gap)
        else:
            res = self._safe_crop(current_active_indices, selected_active_indices, merged_grid_h, merged_grid_w, image_width, image_height)
        return res

    def get_pruned_image(
        self,
        image: Image.Image,
        query: str,
        keep_ratio: float = 0.3,
        percentile_ratio: float = 0.5,
        image_cache_id: str = "example",
        tag_hash: str = "123",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes CATP Attention-Based Pruning using Qwen2-VL's dynamic spatial grid.
        Returns the cropped image and pruning metadata.
        """
        width, height = image.size

        messages = [{"role": "user",
                    "content": [{"type": "image", "image": image},{"type": "text", "text": query}]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Qwen style prompt
        # CATP budget: cap effective `max_pixels` via `image_processor.size["longest_edge"]` for this call only.
        # `processor(..., images_kwargs={"max_pixels": ...})` did not shrink the grid in our Transformers/Optimum stack.
        ip = self.processor.image_processor
        _prev_longest = ip.size["longest_edge"]
        try:
            ip.size["longest_edge"] = self.catp_max_pixels
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        finally:
            ip.size["longest_edge"] = _prev_longest

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True,)

        # Map Qwen2-VL's Dynamic Grid
        input_ids = inputs.input_ids.squeeze(0)
        
        # Qwen2-VL stores the dynamic shape of the image in image_grid_thw (Temporal, Height, Width)
        # For a static image, Temporal is 1. Height and Width are the patch grids.
        grid_t, grid_h, grid_w = inputs.image_grid_thw[0].tolist()
        merge_size = self.processor.image_processor.merge_size
        num_image_tokens = int(grid_t * grid_h * grid_w // (merge_size ** 2))
        
        # Qwen2-VL uses a specific token for image patches (usually <|image_pad|>)
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        if image_token_indices.numel() == 0:
            raise ValueError("Could not find image tokens in input_ids.")

        early_image_features = outputs.hidden_states[0].squeeze(0)[image_token_indices]
        active_mask = torch.ones(early_image_features.shape[0], dtype=torch.bool, device=early_image_features.device,)
        active_image_token_indices = image_token_indices[active_mask]
        current_active_indices = torch.arange(image_token_indices.numel(),device=input_ids.device,)[active_mask]
        
        # Find query token idx
        query_ids = self.processor.tokenizer(query, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(input_ids.device)
        def find_subsequence(seq, subseq):
            n, m = seq.numel(), subseq.numel()
            for i in range(n - m + 1):
                if torch.equal(seq[i:i + m], subseq): return i
            return -1
        query_start_idx = find_subsequence(input_ids, query_ids)
        if query_start_idx == -1:
            raise ValueError("Could not find query tokens in input_ids.")
        query_end_idx = query_start_idx + query_ids.numel()
        merged_grid_h = grid_h // merge_size
        merged_grid_w = grid_w // merge_size

        # # Layer-wise measurement: analyze query -> image attention for every layer (dev)
        layer_metrics = []
        # prev_topk_set = None
        # analysis_topk_ratio = keep_ratio

        # for layer_idx in range(len(outputs.attentions)):
        #     patch_importance = get_fused_patch_importance(outputs, image_token_indices, current_active_indices,
        #                                                   query_start_idx, query_end_idx,
        #                                                   merged_grid_h, merged_grid_w, [layer_idx], "mean", "sum")

        #     num_active = int(current_active_indices.numel())
        #     analysis_k = max(1, int(num_active * analysis_topk_ratio))
        #     analysis_k = min(analysis_k, num_active)

        #     topk = torch.topk(patch_importance, analysis_k)
        #     topk_indices_local = topk.indices.detach().cpu().tolist()
        #     topk_set = set(int(i) for i in topk_indices_local)

        #     total_mass = patch_importance.sum().item()
        #     topk_mass = topk.values.sum().item()
        #     topk_mass_ratio = topk_mass / total_mass if total_mass > 0 else 0.0

        #     entropy = self._attention_entropy(patch_importance)

        #     if prev_topk_set is None:
        #         topk_overlap = None
        #     else:
        #         topk_overlap = len(topk_set & prev_topk_set) / max(1, len(topk_set))

        #     layer_metrics.append({
        #         "layer_idx": int(layer_idx),
        #         "num_active_tokens": num_active,
        #         "query_to_image_attention_mean": float(patch_importance.mean().detach().cpu()),
        #         "query_to_image_attention_max": float(patch_importance.max().detach().cpu()),
        #         "topk_mass_ratio": float(topk_mass_ratio),
        #         "attention_entropy": float(entropy),
        #         "topk_overlap_with_prev_layer": topk_overlap,
        #     })
        #     prev_topk_set = topk_set

        # Spatial Heat Map (dev)
        # spatial_maps = collect_spatial_attention_maps(
        #     outputs=outputs,
        #     input_ids=input_ids,
        #     image_token_indices=image_token_indices,
        #     current_active_indices=current_active_indices,
        #     query_start_idx=query_start_idx,
        #     query_end_idx=query_end_idx,
        #     grid_h=grid_h,
        #     grid_w=grid_w,
        #     merge_size=merge_size,
        #     layer_start=8,
        #     layer_end=16,
        # )
        # save_layer_spatial_attention_grid(
        #     image=image,
        #     spatial_maps=spatial_maps,
        #     save_path=f"data/mmdocrag/analysis/{image_cache_id}_{tag_hash}_spatial_attention_layers_6_to_14.png",
        # )

        patch_importance = get_fused_patch_importance(
            outputs=outputs,
            image_token_indices=image_token_indices,
            current_active_indices=current_active_indices,
            query_start_idx=query_start_idx,
            query_end_idx=query_end_idx,
            merged_grid_h=merged_grid_h,
            merged_grid_w=merged_grid_w,
            selected_layers=(10, 11, 12, 13),
            head_reduce="max",   # "max" or "mean"
            layer_reduce="sum",  # "mean", "max", or "sum"
        )
        bbox_info = self._prune_by_attention(
            patch_importance=patch_importance,
            current_active_indices=current_active_indices,
            hidden_states=early_image_features,
            merged_grid_h=merged_grid_h,
            merged_grid_w=merged_grid_w,
            image_width=width,
            image_height=height,
            keep_ratio=keep_ratio,
            percentile_ratio=percentile_ratio,
        )
        true_tokens_after = bbox_info["tokens_after_bbox"]

        metadata = {
            "tokens_before": num_image_tokens,
            "tokens_after": true_tokens_after,
            "tokens_before_diversity": int(image_token_indices.numel()),
            "tokens_after_diversity": int(active_image_token_indices.numel()),
            "layer_metrics": layer_metrics if len(layer_metrics) != 0 else None,
        }

        return bbox_info["clusters"], metadata

    def get_pruned_image_base64(self, image: Image.Image, query: str, keep_ratio: float = 0.3) -> str:
        """
        Executes CATP Attention-Based Pruning and returns a base64 string.
        Kept for compatibility; pruner.py should use get_pruned_image().
        """
        cropped_image, _ = self.get_pruned_image(image, query, keep_ratio)

        buffered = io.BytesIO()
        cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_str}"
