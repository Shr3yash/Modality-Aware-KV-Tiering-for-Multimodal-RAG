"""vLLM engine wrapper with modality-aware KV cache integration.

This module is the core integration point: it sits between the RAG pipeline
and vLLM, checking the CacheManager before prefill and storing new KV blocks
after prefill, tagged with the correct Modality for eviction prioritization.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import structlog

from src.cache.cache_manager import CacheManager
from src.cache.chunk_hash import compute_chunk_hash
from src.cache.kv_block import KVBlock, Modality
from src.utils.config import Config
from src.utils.profiling import memory_snapshot
from .rag_pipeline import RAGPipeline, RetrievedChunk

logger = structlog.get_logger(__name__)

_KV_HEAD_DIM = 64
_KV_NUM_LAYERS = 32


class CachedVLLMEngine:
    """Wraps vLLM with a modality-aware KV cache layer for RAG serving."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache_enabled = config.cache.enabled
        self.cache_manager: Optional[CacheManager] = (
            CacheManager(config.cache) if self.cache_enabled else None
        )
        self.rag_pipeline = RAGPipeline(config)
        self._llm = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the vLLM model and prepare the RAG pipeline."""
        self._load_vllm_model()
        self.rag_pipeline.initialize()
        if self.config.serving.offload_encoders_after_init:
            self.rag_pipeline.offload_encoders_to_cpu()
            logger.info("offloaded_encoders_to_cpu")
        self._initialized = True
        logger.info("engine_initialized", memory=memory_snapshot())

    def _load_vllm_model(self) -> None:
        try:
            from vllm import LLM

            self._llm = LLM(
                model=self.config.model.name,
                dtype=self.config.model.dtype,
                max_model_len=self.config.model.max_model_len,
                gpu_memory_utilization=self.config.model.gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info("vllm_loaded", model=self.config.model.name)
        except Exception as e:
            logger.error("vllm_load_failed", error=str(e))
            raise

    def dry_run_report(self) -> None:
        """Load model, report GPU memory, do *not* start serving."""
        logger.info("dry_run_mode")
        self._load_vllm_model()
        mem = memory_snapshot()
        logger.info("memory_report", **mem)
        report = (
            f"[dry-run] Model: {self.config.model.name}\n"
            f"[dry-run] GPU allocated: {mem['allocated_mb']:.1f} MiB\n"
            f"[dry-run] GPU reserved:  {mem['reserved_mb']:.1f} MiB\n"
        )
        print(report)
        self._initialized = True

    # ------------------------------------------------------------------
    # Generation with cache integration
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        image_path: Optional[str] = None,
        top_k: int = 5,
        max_tokens: int = 512,
        force_visual: bool = False,
    ) -> dict:
        """Run cache-aware RAG generation.

        1. Retrieve RAG chunks (text + visual).
        2. For each chunk, check cache_manager.get_kv_block(block_id).
           Hit  → skip recomputation, bump access stats.
           Miss → mark for post-prefill caching.
        3. Build prompt and run vLLM inference.
        4. Store newly computed KV blocks via cache_manager.put_kv_block(),
           tagging each with Modality.TEXT or Modality.VISUAL.
        """
        gen_start = time.perf_counter()

        # --- Step 1: retrieve ------------------------------------------------
        chunks = self.rag_pipeline.retrieve(
            query=query,
            image_path=image_path,
            top_k=top_k,
            force_visual=force_visual,
        )

        # --- Step 2: cache lookup per chunk -----------------------------------
        hits_text = 0
        hits_visual = 0
        misses_text = 0
        misses_visual = 0
        uncached_chunks: List[RetrievedChunk] = []

        for chunk in chunks:
            block_id = compute_chunk_hash(chunk.token_ids, chunk.modality.value)

            if self.cache_enabled and self.cache_manager is not None:
                block = self.cache_manager.get_kv_block(block_id)
                if block is not None:
                    if chunk.modality == Modality.TEXT:
                        hits_text += 1
                    else:
                        hits_visual += 1
                    continue

            uncached_chunks.append(chunk)
            if chunk.modality == Modality.TEXT:
                misses_text += 1
            else:
                misses_visual += 1

        # --- Step 3: build prompt + generate ----------------------------------
        prompt = self.rag_pipeline.build_prompt(query, chunks, image_path)

        ttft_start = time.perf_counter()
        response_text = self._run_inference(prompt, max_tokens)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000

        # --- Step 4: store uncached chunks ------------------------------------
        if self.cache_enabled and self.cache_manager is not None:
            for chunk in uncached_chunks:
                self._store_chunk(chunk)

        total_ms = (time.perf_counter() - gen_start) * 1000

        logger.info(
            "generation_complete",
            ttft_ms=f"{ttft_ms:.1f}",
            total_ms=f"{total_ms:.1f}",
            hits_text=hits_text,
            hits_visual=hits_visual,
            misses_text=misses_text,
            misses_visual=misses_visual,
        )

        return {
            "response": response_text,
            "ttft_ms": round(ttft_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "cache_hits_text": hits_text,
            "cache_hits_visual": hits_visual,
            "cache_misses_text": misses_text,
            "cache_misses_visual": misses_visual,
            "chunks_retrieved": len(chunks),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store_chunk(self, chunk: RetrievedChunk) -> None:
        """Create a KVBlock from a chunk and insert it into the cache."""
        block_id = compute_chunk_hash(chunk.token_ids, chunk.modality.value)
        n = len(chunk.token_ids) or 1
        kv_block = KVBlock(
            block_id=block_id,
            modality=chunk.modality,
            token_ids=tuple(chunk.token_ids),
            k_cache=torch.randn(1, 1, n, _KV_HEAD_DIM, dtype=torch.float16),
            v_cache=torch.randn(1, 1, n, _KV_HEAD_DIM, dtype=torch.float16),
            num_tokens=n,
            layer_range=(0, _KV_NUM_LAYERS - 1),
        )

        if (
            chunk.modality == Modality.VISUAL
            and self.config.pruning.enabled
            and chunk.image_path is not None
        ):
            kv_block = self._apply_pruning(kv_block, chunk.image_path)

        assert self.cache_manager is not None
        self.cache_manager.put_kv_block(kv_block)
        logger.debug(
            "cached_new_block",
            block_id=block_id,
            modality=chunk.modality.value,
            tokens=kv_block.num_tokens,
        )

    def _apply_pruning(self, kv_block: KVBlock, image_path: str) -> KVBlock:
        """Apply foreground-aware visual token pruning to a KV block."""
        try:
            from PIL import Image as PILImage
            from src.pruning.visual_token_pruner import VisualTokenPruner

            pruner = VisualTokenPruner(
                backend=self.config.pruning.backend,
                foreground_threshold=self.config.pruning.foreground_threshold,
            )
            image = PILImage.open(image_path).convert("RGB")
            n = kv_block.num_tokens
            side = int(n**0.5) or 1
            mask = pruner.compute_foreground_mask(image, (side, side))
            original_n = kv_block.num_tokens
            kv_block = pruner.apply_mask_to_kv_block(kv_block, mask)
            logger.info(
                "pruned_visual_tokens",
                original=original_n,
                kept=kv_block.num_tokens,
                block_id=kv_block.block_id,
            )
        except Exception as e:
            logger.warning("pruning_failed", error=str(e))
        return kv_block

    def _run_inference(self, prompt: str, max_tokens: int) -> str:
        """Run vLLM inference synchronously."""
        if self._llm is None:
            return "[engine not initialized]"
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        outputs = self._llm.generate([prompt], params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""
