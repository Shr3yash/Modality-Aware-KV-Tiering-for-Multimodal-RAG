"""FastAPI serving layer with cache-integrated RAG generation."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.utils.config import load_config
from src.utils.logging import configure_logging
from .engine import CachedVLLMEngine

logger = structlog.get_logger(__name__)

_engine: Optional[CachedVLLMEngine] = None


class GenerateRequest(BaseModel):
    query: str
    image_path: Optional[str] = None
    top_k: int = 5
    max_tokens: int = 512
    force_visual: bool = False


class GenerateResponse(BaseModel):
    response: str
    ttft_ms: float
    total_latency_ms: float
    cache_hits_text: int
    cache_hits_visual: int
    cache_misses_text: int
    cache_misses_visual: int
    chunks_retrieved: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    configure_logging()
    config = load_config(app.state.config_path)
    logger.info("initializing_engine", model=config.model.name)
    _engine = CachedVLLMEngine(config)
    if app.state.dry_run:
        _engine.dry_run_report()
    else:
        _engine.initialize()
        logger.info("engine_ready")
    yield
    logger.info("shutting_down")
    _engine = None


app = FastAPI(title="Modality-Aware KV Cache Server", lifespan=lifespan)
app.state.config_path = "configs/default.yaml"
app.state.dry_run = False


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_loaded": _engine is not None and _engine._initialized,
    }


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if _engine is None or not _engine._initialized:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    result = await _engine.generate(
        query=request.query,
        image_path=request.image_path,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        force_visual=request.force_visual,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Modality-Aware KV Cache Server")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model, report memory usage, then exit",
    )
    args = parser.parse_args()

    app.state.config_path = args.config
    app.state.dry_run = args.dry_run

    config = load_config(args.config)
    host = args.host or config.serving.host
    port = args.port or config.serving.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
