"""Unit tests for Reciprocal Rank Fusion + BM25 tokenization.

These exercise pure-Python helpers in rag.retriever and do not require torch,
CUDA, or model downloads."""

from rag.retriever import _bm25_tokenize, _rrf_fuse


def test_bm25_tokenize_lowercases_and_splits():
    assert _bm25_tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_bm25_tokenize_empty():
    assert _bm25_tokenize("") == []
    assert _bm25_tokenize(None) == []


def test_rrf_fuse_prefers_consensus():
    # Item 1 is top in both lists -> wins.
    dense = [1, 2, 3]
    bm25 = [1, 3, 2]
    fused = _rrf_fuse([dense, bm25], [1.0, 1.0], k=60, top_k=3)
    assert fused[0] == 1
    assert set(fused) == {1, 2, 3}


def test_rrf_fuse_weights_skew_ranking():
    # Dense puts 2 first; BM25 puts 5 first. Heavy BM25 weight surfaces 5.
    dense = [2, 5]
    bm25 = [5, 2]
    fused = _rrf_fuse([dense, bm25], [1.0, 10.0], k=60, top_k=2)
    assert fused[0] == 5


def test_rrf_fuse_zero_weight_disables_list():
    dense = [9, 8]
    bm25 = [7, 6]
    fused = _rrf_fuse([dense, bm25], [1.0, 0.0], k=60, top_k=2)
    assert fused == [9, 8]


def test_rrf_fuse_empty_returns_empty():
    assert _rrf_fuse([[], []], [1.0, 1.0], k=60, top_k=5) == []
