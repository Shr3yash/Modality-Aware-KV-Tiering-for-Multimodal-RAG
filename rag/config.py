from dataclasses import dataclass
from pathlib import Path

@dataclass
class RAGConfig:
    data_root: Path = Path("data/mmdocrag")
    ann_file: Path = Path("data/mmdocrag/evaluation_15.jsonl")
    images_root: Path = Path("data/mmdocrag/images")
    output_dir: Path = Path("data/mmdocrag/outputs")

    # retrieval
    text_top_k: int = 4
    image_top_k: int = 2
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_embedding_model: str = "openai/clip-vit-base-patch32"
    retrieval_device: str = "cuda"

    # hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion)
    # mode: "dense" (current behavior) | "hybrid" (BM25 + dense via RRF)
    retrieval_mode: str = "dense"
    bm25_weight: float = 1.0
    rrf_k: int = 60

    # query rewriting / expansion (multi-query retrieval)
    # mode: "none" (passthrough) | "rule_based" | "llm"
    query_rewrite_mode: str = "none"
    query_rewrite_max_variants: int = 3
    query_rewrite_cache_path: Path = Path("data/mmdocrag/outputs/query_rewrite_cache.json")
    # LLM rewriter reuses the VLM endpoint by default; override for a separate model.
    query_rewrite_api_base: str | None = None
    query_rewrite_model_name: str | None = None

    # pruning baselines
    pruning_keep_ratio: float = 0.30
    pruning_percentile_ratio: float = 0.70
    # {"no_pruning", "visual_patch_pruning", "safecrop_pruning", "cluster_pruning"}
    pruning_mode: str = "safecrop_pruning"
    patch_grid_rows: int = 4
    patch_grid_cols: int = 4
    min_visual_tokens: int = 4
    montage_tile_size: int = 224
    pruned_image_dir: Path = Path("data/mmdocrag/outputs/pruned_images")
    image_prune_cache_enabled: bool = True
    image_prune_cache_similarity_threshold: float = 0.85
    image_prune_cache_path: Path = Path("data/mmdocrag/outputs/image_prune_cache.json")

    # generation
    vlm_api_base: str = "http://127.0.0.1:8000/v1"
    vlm_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    temperature: float = 0.0

    # offline judge
    judge_api_base: str = "http://127.0.0.1:8000/v1"
    judge_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # eval (None = use entire split / JSONL / official VQA file)
    max_examples: int | None = 50
    # MMDocRAG JSONL line indices (0-based); slice_stop is exclusive (Python slice).
    # Rows must fall in [slice_start, slice_stop) before max_examples caps length.
    eval_slice_start: int = 0
    eval_slice_stop: int | None = None

    # benchmark: "mmdocrag" (JSONL) or "vqav2" (Hugging Face)
    benchmark: str = "mmdocrag"
    # Full VQAv2 val (~214k Q): set all three to local paths from visualqa.org + COCO val2014 images.
    vqa_questions_json: Path | None = None
    vqa_annotations_json: Path | None = None
    vqa_coco_images_dir: Path | None = None
    vqa_hf_dataset: str = "HuggingFaceM4/VQAv2"
    vqa_hf_split: str = "validation"
    vqa_hf_revision: str | None = None
    vqa_hf_auto_fallback: bool = True
    vqa_hf_fallback_dataset: str = "Multimodal-Fatima/VQAv2_sample_validation"
    vqa_hf_fallback_split: str = "validation"
    vqa_image_cache_dir: Path = Path("data/vqav2/image_cache")
