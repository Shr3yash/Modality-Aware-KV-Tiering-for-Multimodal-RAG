from __future__ import annotations

from types import SimpleNamespace

from tests.phase2_spec_helpers import (
    call_with_supported_args,
    get_value,
    instantiate_with_supported_args,
    require_any_attr,
    require_phase2_module,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return [
            {
                "chunk_id": "c1",
                "text": "cache context",
                "source": "doc1.txt",
                "retrieval_rank": 1,
                "modality": "text",
            },
            {
                "chunk_id": "c2",
                "text": "diagram of cache tiers",
                "source": "doc2.txt#page=2",
                "retrieval_rank": 2,
                "modality": "image",
                "image_path": "/tmp/retrieved.png",
            },
        ]


class FakePromptBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "segments": [
                {
                    "role": "system",
                    "modality": "text",
                    "source": "system",
                    "text": "You are helpful.",
                    "text_length": 16,
                },
                {
                    "role": "user",
                    "modality": "text",
                    "source": "user",
                    "text": kwargs["query"],
                    "text_length": len(str(kwargs["query"])),
                },
                {
                    "role": "retrieved_text",
                    "modality": "text",
                    "source": "doc1.txt",
                    "text": "cache context",
                    "text_length": 13,
                    "retrieval_rank": 1,
                },
                {
                    "role": "retrieved_visual",
                    "modality": "image",
                    "source": "doc2.txt#page=2",
                    "text": "diagram of cache tiers",
                    "text_length": 22,
                    "retrieval_rank": 2,
                },
            ],
            "payload": {"messages": [{"role": "user", "content": []}]},
            "segment_metadata": {"ordered_segment_roles": ["system", "user", "retrieved_text", "retrieved_visual"]},
        }

    __call__ = build_prompt


class FakeModelRunner:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def generate(self, request_payload):
        self.calls.append(request_payload)
        return {
            "generated_text": "service answer",
            "timing": {"generation_seconds": 0.05},
        }


class FakePromptLogger:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def summarize_prompt(self, segments, **kwargs):
        self.calls.append({"segments": segments, "kwargs": kwargs})
        return {
            "total_prompt_tokens": 7,
            "retrieved_text_token_count": 2,
            "visual_input_present": True,
            "num_retrieved_chunks": 2,
            "ordered_segment_roles": ["system", "user", "retrieved_text", "retrieved_visual"],
        }

    __call__ = summarize_prompt


def _make_request() -> object:
    schemas = require_phase2_module("src.rag.schemas")
    generate_request = require_any_attr(schemas, ["GenerateRequest"])
    return generate_request(query="What does the cache reuse?", image_path=None, top_k=2)


def test_service_flow_orchestrates_retrieval_prompting_generation_and_summaries() -> None:
    service_module = require_phase2_module("src.serving.service")
    service_cls = require_any_attr(service_module, ["GenerationService", "RAGService", "Service"])

    retriever = FakeRetriever()
    prompt_builder = FakePromptBuilder()
    model_runner = FakeModelRunner()
    prompt_logger = FakePromptLogger()

    service = instantiate_with_supported_args(
        service_cls,
        retriever=retriever,
        prompt_builder=prompt_builder,
        model_runner=model_runner,
        prompt_logger=prompt_logger,
        token_logger=prompt_logger,
        top_k=4,
        config=SimpleNamespace(rag=SimpleNamespace(top_k=4)),
    )
    request = _make_request()
    generate = require_any_attr(service, ["generate", "handle_generate", "run"])

    response = call_with_supported_args(generate, request=request)

    assert retriever.calls == [("What does the cache reuse?", 2)]
    assert len(prompt_builder.calls) == 1
    assert model_runner.calls == [{"messages": [{"role": "user", "content": []}]}]
    assert get_value(response, "answer_text", "generated_text", "text") == "service answer"
    assert get_value(response, "retrieved_chunk_ids") == ["c1", "c2"]
    assert get_value(response, "prompt_summary")["ordered_segment_roles"] == [
        "system",
        "user",
        "retrieved_text",
        "retrieved_visual",
    ]
    assert get_value(response, "token_summary")["total_prompt_tokens"] == 7
    assert get_value(response, "timing_summary", "timing")["generation_seconds"] == 0.05


def test_service_flow_uses_default_top_k_when_request_does_not_override_it() -> None:
    service_module = require_phase2_module("src.serving.service")
    service_cls = require_any_attr(service_module, ["GenerationService", "RAGService", "Service"])
    schemas = require_phase2_module("src.rag.schemas")
    generate_request = require_any_attr(schemas, ["GenerateRequest"])

    retriever = FakeRetriever()
    service = instantiate_with_supported_args(
        service_cls,
        retriever=retriever,
        prompt_builder=FakePromptBuilder(),
        model_runner=FakeModelRunner(),
        prompt_logger=FakePromptLogger(),
        token_logger=FakePromptLogger(),
        top_k=4,
        config=SimpleNamespace(rag=SimpleNamespace(top_k=4)),
    )
    generate = require_any_attr(service, ["generate", "handle_generate", "run"])

    response = call_with_supported_args(
        generate,
        request=generate_request(query="default top k", image_path=None, top_k=None),
    )

    assert retriever.calls == [("default top k", 4)]
    assert get_value(response, "answer_text", "generated_text", "text") == "service answer"
