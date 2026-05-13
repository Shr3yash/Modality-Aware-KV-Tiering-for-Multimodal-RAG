"""Unit tests for rag.query_rewriter (pure-Python; no torch/model deps)."""

import json
from pathlib import Path

from rag.query_rewriter import (
    LLMRewriter,
    NoopRewriter,
    RuleBasedRewriter,
    build_rewriter,
    query_hash,
)


def test_noop_returns_original_only():
    assert NoopRewriter().rewrite("what is the capital of France?") == [
        "what is the capital of France?"
    ]


def test_rule_based_adds_keyword_variant():
    r = RuleBasedRewriter(max_variants=3)
    variants = r.rewrite("What is the capital of France?")
    assert variants[0] == "What is the capital of France?"
    assert any("capital" in v and "france" in v and "what" not in v.split() for v in variants[1:])


def test_rule_based_drops_redundant_variant():
    # All-stopword query yields no useful keyword variant; should not duplicate.
    r = RuleBasedRewriter(max_variants=3)
    variants = r.rewrite("the and of")
    assert variants == ["the and of"]


def test_rule_based_respects_max_variants():
    r = RuleBasedRewriter(max_variants=1)
    variants = r.rewrite("What is the capital of France?")
    assert variants == ["What is the capital of France?"]


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeChat:
    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResp(self._content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = type("Chat", (), {"completions": _FakeChat(content)})()


def test_llm_rewriter_parses_and_caches(tmp_path: Path):
    cache = tmp_path / "rewrite.json"
    client = _FakeClient('["capital of france", "france capital city"]')
    r = LLMRewriter(
        api_base="unused",
        model_name="unused",
        max_variants=3,
        cache_path=cache,
        client=client,
    )

    v1 = r.rewrite("What is the capital of France?")
    assert v1[0] == "What is the capital of France?"
    assert "capital of france" in v1
    assert cache.exists()

    # Second call hits cache (no new LLM call).
    calls_before = client.chat.completions.calls
    v2 = r.rewrite("What is the capital of France?")
    assert v2 == v1
    assert client.chat.completions.calls == calls_before


def test_llm_rewriter_handles_bad_json(tmp_path: Path):
    client = _FakeClient("not valid json at all")
    r = LLMRewriter(
        api_base="unused",
        model_name="unused",
        max_variants=3,
        cache_path=tmp_path / "rewrite.json",
        client=client,
    )
    variants = r.rewrite("hello world")
    assert variants == ["hello world"]


def test_llm_rewriter_strips_code_fences(tmp_path: Path):
    client = _FakeClient('```json\n["a", "b"]\n```')
    r = LLMRewriter(
        api_base="unused",
        model_name="unused",
        max_variants=3,
        cache_path=tmp_path / "rewrite.json",
        client=client,
    )
    variants = r.rewrite("q")
    assert variants == ["q", "a", "b"]


class _Cfg:
    def __init__(self, mode):
        self.query_rewrite_mode = mode
        self.query_rewrite_max_variants = 3
        self.query_rewrite_cache_path = None
        self.query_rewrite_api_base = None
        self.query_rewrite_model_name = None
        self.vlm_api_base = "http://localhost"
        self.vlm_model_name = "test-model"


def test_build_rewriter_factory_returns_correct_types():
    assert isinstance(build_rewriter(_Cfg("none")), NoopRewriter)
    assert isinstance(build_rewriter(_Cfg("rule_based")), RuleBasedRewriter)


def test_query_hash_stable_and_short():
    h = query_hash("hello")
    assert len(h) == 12
    assert query_hash("hello") == h
