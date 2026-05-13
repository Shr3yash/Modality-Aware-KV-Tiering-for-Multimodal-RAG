"""Query rewriting / expansion for multi-query retrieval.

Strategies:
- NoopRewriter:        passthrough -> [original]
- RuleBasedRewriter:   adds a keyword-only variant (stopwords stripped) using
                       metrics.normalize_text as the shared text preprocessor.
- LLMRewriter:         asks an OpenAI-compatible endpoint for paraphrases and
                       caches them on disk by question hash, mirroring the
                       image_prune_cache JSON layout used in query_pipeline.

Variants always include the original query first so downstream RRF fusion can
treat it as the primary ranking signal.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol


def _normalize_text(s: str) -> str:
    # Lazy reuse of rag.metrics.normalize_text; imported inside the function so
    # this module stays importable in environments that lack the openai client
    # (rag.metrics top-level-imports it). Behavior is identical to that helper.
    try:
        from rag.metrics import normalize_text as _nt
        return _nt(s)
    except Exception:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s


# Compact English stopword list. Kept inline to avoid pulling NLTK just for this.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of in on at to for from by with about as is are was were be
    been being do does did doing have has had having i you he she it we they them his her its our
    their this that these those what which who whom how why when where there here not no nor so
    can could should would may might must will shall just also too very much more most some any
    each every all both either neither than into out over under up down off again further once
    """.split()
)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


class QueryRewriter(Protocol):
    """Returns 1..N query variants, original first."""

    def rewrite(self, query: str) -> List[str]: ...


class NoopRewriter:
    def rewrite(self, query: str) -> List[str]:
        return [query]


class RuleBasedRewriter:
    """Adds a stopword-stripped keyword variant alongside the original."""

    def __init__(self, max_variants: int = 3):
        self.max_variants = max(1, int(max_variants))

    def _keyword_variant(self, query: str) -> str:
        norm = _normalize_text(query)
        tokens = [t for t in norm.split() if t and t not in _STOPWORDS]
        return " ".join(tokens)

    def rewrite(self, query: str) -> List[str]:
        variants = [query]
        kw = self._keyword_variant(query)
        if kw and kw != _normalize_text(query):
            variants.append(kw)
        # Future room for additional deterministic variants (e.g. question -> noun phrase).
        return variants[: self.max_variants]


_LLM_PROMPT = (
    "Rewrite the following search query into {n} short, diverse paraphrases that "
    "preserve the original intent. Return ONLY a JSON array of strings.\n\n"
    "Query: {query}"
)


@dataclass
class _CachedRewrite:
    query: str
    variants: List[str]


class LLMRewriter:
    """Calls an OpenAI-compatible endpoint to produce paraphrases; caches on disk."""

    def __init__(
        self,
        api_base: str,
        model_name: str,
        max_variants: int = 3,
        cache_path: Optional[Path] = None,
        client=None,
    ):
        self.api_base = api_base
        self.model_name = model_name
        self.max_variants = max(1, int(max_variants))
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, _CachedRewrite] = self._load_cache()
        self._client = client  # injected for tests; created lazily otherwise

    def _load_cache(self) -> dict[str, _CachedRewrite]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
        out: dict[str, _CachedRewrite] = {}
        for key, value in entries.items():
            if not isinstance(value, dict):
                continue
            q = value.get("query")
            variants = value.get("variants")
            if isinstance(q, str) and isinstance(variants, list) and all(
                isinstance(v, str) for v in variants
            ):
                out[key] = _CachedRewrite(query=q, variants=variants)
        return out

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": {
                key: {"query": entry.query, "variants": entry.variants}
                for key, entry in self._cache.items()
            },
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key="EMPTY")
        return self._client

    def _parse_variants(self, text: str) -> List[str]:
        text = text.strip()
        # Strip code fences if present.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            arr = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(arr, list):
            return []
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()]

    def _call_llm(self, query: str, n: int) -> List[str]:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(n=n, query=query)}],
            temperature=0.0,
        )
        return self._parse_variants(resp.choices[0].message.content or "")

    def rewrite(self, query: str) -> List[str]:
        key = query_hash(query)
        cached = self._cache.get(key)
        if cached and cached.query == query:
            return cached.variants[: self.max_variants]

        n_extra = max(0, self.max_variants - 1)
        extra: List[str] = []
        if n_extra > 0:
            try:
                extra = self._call_llm(query, n_extra)
            except Exception:
                extra = []  # never break the pipeline on a rewrite failure
        variants = [query] + [v for v in extra if v != query][: n_extra]
        self._cache[key] = _CachedRewrite(query=query, variants=variants)
        self._save_cache()
        return variants


def build_rewriter(cfg) -> QueryRewriter:
    """Construct a rewriter from a RAGConfig-like object."""
    mode = getattr(cfg, "query_rewrite_mode", "none")
    max_variants = int(getattr(cfg, "query_rewrite_max_variants", 3))

    if mode == "none":
        return NoopRewriter()
    if mode == "rule_based":
        return RuleBasedRewriter(max_variants=max_variants)
    if mode == "llm":
        api_base = getattr(cfg, "query_rewrite_api_base", None) or cfg.vlm_api_base
        model_name = getattr(cfg, "query_rewrite_model_name", None) or cfg.vlm_model_name
        cache_path = getattr(cfg, "query_rewrite_cache_path", None)
        return LLMRewriter(
            api_base=api_base,
            model_name=model_name,
            max_variants=max_variants,
            cache_path=cache_path,
        )
    raise ValueError(f"Unknown query_rewrite_mode: {mode!r}")
