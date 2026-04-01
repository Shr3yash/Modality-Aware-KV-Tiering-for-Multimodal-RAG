from __future__ import annotations

from types import SimpleNamespace

from tests.phase2_spec_helpers import (
    call_with_supported_args,
    get_value,
    instantiate_with_supported_args,
    require_any_attr,
    require_phase2_module,
)


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def generate(self, payload, **kwargs):
        self.calls.append({"payload": payload, "kwargs": kwargs})
        return {
            "generated_text": "offline answer",
            "generation_seconds": 0.05,
        }


def test_model_runner_initializes_engine_once_and_reuses_it() -> None:
    model_runner_module = require_phase2_module("src.serving.model_runner")
    runner_cls = require_any_attr(model_runner_module, ["ModelRunner"])

    engine = FakeEngine()
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def engine_factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return engine

    runner = instantiate_with_supported_args(
        runner_cls,
        model_config=SimpleNamespace(
            name="Qwen/Qwen2.5-Omni-3B",
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        ),
        model_name="Qwen/Qwen2.5-Omni-3B",
        engine_factory=engine_factory,
    )
    generate = require_any_attr(runner, ["generate"])

    first = call_with_supported_args(generate, request_payload={"prompt": "hello"})
    second = call_with_supported_args(
        generate,
        request_payload={"prompt": "describe image", "image_path": "/tmp/example.png"},
    )

    assert len(factory_calls) == 1
    assert factory_calls[0][1]["allowed_local_media_path"] == "/"
    assert len(engine.calls) == 2
    assert get_value(first, "generated_text", "text", "answer_text") == "offline answer"
    assert get_value(second, "generated_text", "text", "answer_text") == "offline answer"


def test_model_runner_returns_generated_text_and_timing_metadata() -> None:
    model_runner_module = require_phase2_module("src.serving.model_runner")
    runner_cls = require_any_attr(model_runner_module, ["ModelRunner"])

    engine = FakeEngine()
    runner = instantiate_with_supported_args(
        runner_cls,
        model_config=SimpleNamespace(name="Qwen/Qwen2.5-Omni-3B"),
        model_name="Qwen/Qwen2.5-Omni-3B",
        engine_factory=lambda *args, **kwargs: engine,
    )
    generate = require_any_attr(runner, ["generate"])

    response = call_with_supported_args(generate, request_payload={"prompt": "hello"})

    timing = get_value(response, "timing", "timing_summary", default={})
    assert get_value(response, "generated_text", "text", "answer_text") == "offline answer"
    assert timing or "generation_seconds" in str(response)
