from fastapi.testclient import TestClient

from tests.phase2_spec_helpers import call_with_supported_args, require_any_attr, require_phase2_module


class FakeService:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return {
            "answer_text": "api answer",
            "retrieved_chunk_ids": ["c1"],
            "prompt_summary": {"ordered_segment_roles": ["system", "user"]},
            "token_summary": {"total_prompt_tokens": 4},
            "timing_summary": {"generation_seconds": 0.01},
        }


def test_generate_endpoint_accepts_phase2_request_and_returns_service_response() -> None:
    api_module = require_phase2_module("src.serving.api")
    create_app = require_any_attr(api_module, ["create_app"])

    service = FakeService()
    client = TestClient(
        call_with_supported_args(
            create_app,
            service=service,
            generation_service=service,
        )
    )

    response = client.post(
        "/generate",
        json={
            "query": "What does the image show?",
            "image_path": "/tmp/example.png",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer_text"] == "api answer"
    assert len(service.calls) == 1


def test_health_endpoint_reports_ready() -> None:
    api_module = require_phase2_module("src.serving.api")
    create_app = require_any_attr(api_module, ["create_app"])

    service = FakeService()
    client = TestClient(
        call_with_supported_args(
            create_app,
            service=service,
            generation_service=service,
        )
    )
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
