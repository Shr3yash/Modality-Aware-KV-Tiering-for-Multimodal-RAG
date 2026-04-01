from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, Field


DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95


class ModelRunnerResponse(BaseModel):
    generated_text: str
    timing: dict[str, Any] = Field(default_factory=dict)
    raw_response: Any | None = None


class ModelRunner:
    def __init__(
        self,
        *,
        model_config: Any | None = None,
        model_name: str | None = None,
        engine_factory: Callable[..., Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> None:
        self.model_config = model_config
        self.model_name = model_name or getattr(model_config, "name", None)
        self.engine_factory = engine_factory or self._default_engine_factory
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._engine: Any | None = None

    def generate(
        self,
        request_payload: Any,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> ModelRunnerResponse:
        engine = self._get_engine()
        normalized_payload = self._normalize_request_payload(request_payload)
        generation_kwargs = self._build_generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        started_at = perf_counter()
        raw_response = self._invoke_engine(
            engine,
            normalized_payload,
            generation_kwargs,
        )
        elapsed = perf_counter() - started_at

        timing = self._extract_timing(raw_response, default_seconds=elapsed)
        return ModelRunnerResponse(
            generated_text=self._extract_generated_text(raw_response),
            timing=timing,
            raw_response=raw_response,
        )

    def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = self.engine_factory(
                model=self.model_name,
                dtype=getattr(self.model_config, "dtype", None),
                max_model_len=getattr(self.model_config, "max_model_len", None),
                gpu_memory_utilization=getattr(self.model_config, "gpu_memory_utilization", None),
                enforce_eager=getattr(self.model_config, "enforce_eager", True),
                allowed_local_media_path=getattr(self.model_config, "allowed_local_media_path", "/"),
            )
        return self._engine

    def _default_engine_factory(self, **kwargs: Any) -> Any:
        from vllm import LLM

        llm_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return LLM(**llm_kwargs)

    def _build_generation_kwargs(
        self,
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> dict[str, Any]:
        params = {
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
        }
        try:
            from vllm import SamplingParams

            return {"sampling_params": SamplingParams(**params)}
        except Exception:
            return params

    def _normalize_request_payload(self, request_payload: Any) -> Any:
        if isinstance(request_payload, str):
            return {"prompt": request_payload}
        if not isinstance(request_payload, Mapping):
            return request_payload

        payload = dict(request_payload)
        if "messages" in payload:
            return payload
        if "prompt" in payload:
            prompt = str(payload["prompt"])
            image_path = payload.get("image_path")
            if image_path:
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                            ],
                        }
                    ]
                }
            return {"prompt": prompt}
        return payload

    def _invoke_engine(
        self,
        engine: Any,
        normalized_payload: Any,
        generation_kwargs: dict[str, Any],
    ) -> Any:
        if isinstance(normalized_payload, Mapping) and "messages" in normalized_payload and hasattr(engine, "chat"):
            return engine.chat(normalized_payload["messages"], **generation_kwargs)
        if hasattr(engine, "generate"):
            return engine.generate(normalized_payload, **generation_kwargs)
        raise AttributeError("Model engine must expose either 'chat' or 'generate'.")

    def _extract_generated_text(self, raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, Mapping):
            for key in ("generated_text", "text", "answer_text"):
                value = raw_response.get(key)
                if isinstance(value, str):
                    return value

        if isinstance(raw_response, list) and raw_response:
            first = raw_response[0]
            outputs = getattr(first, "outputs", None)
            if outputs:
                first_output = outputs[0]
                text = getattr(first_output, "text", None)
                if isinstance(text, str):
                    return text
            if isinstance(first, Mapping):
                return self._extract_generated_text(first)

        output = getattr(raw_response, "outputs", None)
        if output:
            first_output = output[0]
            text = getattr(first_output, "text", None)
            if isinstance(text, str):
                return text
        return ""

    def _extract_timing(
        self,
        raw_response: Any,
        *,
        default_seconds: float,
    ) -> dict[str, Any]:
        if isinstance(raw_response, Mapping):
            timing = raw_response.get("timing")
            if isinstance(timing, Mapping):
                return dict(timing)

            if "generation_seconds" in raw_response:
                return {"generation_seconds": raw_response["generation_seconds"]}

        return {"generation_seconds": default_seconds}
