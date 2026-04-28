from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
import inspect
from types import SimpleNamespace
from typing import Any

import pytest


def require_phase2_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"Phase 2 module '{module_name}' is not implemented yet: {exc}",
            pytrace=False,
        )


def require_any_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    pytest.fail(
        f"Expected one of {names!r} on {obj!r}, but none were found.",
        pytrace=False,
    )


def declared_fields(obj: Any) -> set[str]:
    if hasattr(obj, "model_fields"):
        return set(obj.model_fields.keys())
    if hasattr(obj, "__annotations__"):
        return set(obj.__annotations__.keys())
    if is_dataclass(obj):
        return set(asdict(obj).keys())
    return set()


def to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    pytest.fail(f"Could not coerce {value!r} into a mapping.", pytrace=False)


_MISSING = object()


def get_value(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    pytest.fail(
        f"Expected one of {names!r} on {value!r}, but none were found.",
        pytrace=False,
    )


def call_with_supported_args(func: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)

    supported = {name: value for name, value in kwargs.items() if name in parameters}
    return func(**supported)


def instantiate_with_supported_args(cls: type[Any], **kwargs: Any) -> Any:
    return call_with_supported_args(cls, **kwargs)


def make_rag_config(
    corpus_dir: str,
    *,
    chunk_size_tokens: int = 4,
    top_k: int = 2,
    embedding_model: str = "phase2-test-embedder",
) -> SimpleNamespace:
    return SimpleNamespace(
        corpus_dir=corpus_dir,
        chunk_size_tokens=chunk_size_tokens,
        top_k=top_k,
        embedding_model=embedding_model,
    )


def normalize_prompt_result(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 3:
        segments, payload, segment_metadata = result
        return {
            "segments": segments,
            "payload": payload,
            "segment_metadata": segment_metadata,
        }

    mapping = to_mapping(result)
    if "segments" in mapping:
        return {
            "segments": mapping["segments"],
            "payload": get_value(
                mapping,
                "payload",
                "prompt_payload",
                "request_payload",
                "model_input",
                default=None,
            ),
            "segment_metadata": get_value(
                mapping,
                "segment_metadata",
                "metadata",
                default=None,
            ),
        }

    pytest.fail(
        "Prompt builder result should expose segments, payload, and segment metadata.",
        pytrace=False,
    )


def get_roles(segments: list[Any]) -> list[str]:
    return [str(get_value(segment, "role")) for segment in segments]
