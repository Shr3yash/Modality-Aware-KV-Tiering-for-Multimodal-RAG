from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.rag import GenerateRequest, GenerateResponse
from src.serving.service import GenerationService


def create_app(
    *,
    service: Any | None = None,
    generation_service: Any | None = None,
    config: Any | None = None,
) -> FastAPI:
    resolved_service = generation_service or service or GenerationService(config=config)

    app = FastAPI(
        title="Modality-Aware KV Tiering Demo API",
        version="0.1.0",
    )
    app.state.generation_service = resolved_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> Any:
        return app.state.generation_service.generate(request)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
