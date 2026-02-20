"""Tests for retrieval logic — search routing, RRF fusion, and reranking."""

import numpy as np
import pytest

from eros.retrieval import (
    SearchResult,
    format_results,
    format_explain,
    rrf_fuse,
)


class TestRRFFusion:
    """Test Reciprocal Rank Fusion for merging ranked result lists."""

    def test_single_list(self):
        """RRF of a single list should preserve order."""
        results = [
            SearchResult(text="a", score=0.9, collection="code", metadata={"id": "1"}),
            SearchResult(text="b", score=0.5, collection="code", metadata={"id": "2"}),
        ]
        fused = rrf_fuse([results], k=60)
        assert len(fused) == 2
        assert fused[0].text == "a"
        assert fused[1].text == "b"

    def test_two_lists_merge(self):
        """RRF should merge two lists, boosting items that appear in both."""
        list_a = [
            SearchResult(text="shared", score=0.9, collection="code", metadata={"id": "1"}),
            SearchResult(text="only_a", score=0.5, collection="code", metadata={"id": "2"}),
        ]
        list_b = [
            SearchResult(text="only_b", score=0.8, collection="docs", metadata={"id": "3"}),
            SearchResult(text="shared", score=0.7, collection="code", metadata={"id": "1"}),
        ]
        fused = rrf_fuse([list_a, list_b], k=60)
        # "shared" appears in both lists, so it should rank highest
        assert fused[0].text == "shared"

    def test_empty_lists(self):
        """RRF of empty lists should return empty."""
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []

    def test_deduplication_by_text(self):
        """Same text from different lists should be merged, not duplicated."""
        list_a = [
            SearchResult(text="same", score=0.9, collection="code", metadata={"id": "1"}),
        ]
        list_b = [
            SearchResult(text="same", score=0.8, collection="docs", metadata={"id": "1"}),
        ]
        fused = rrf_fuse([list_a, list_b], k=60)
        assert len(fused) == 1


class TestFormatResults:
    """Test formatting search results for MCP output."""

    def test_format_basic(self):
        """Should format results as readable text."""
        results = [
            SearchResult(
                text="def authenticate(user, pwd)",
                score=0.92,
                collection="code",
                metadata={
                    "symbol_name": "authenticate",
                    "kind": "function",
                    "language": "python",
                    "file_path": "src/auth.py",
                    "start_line": 10,
                },
            ),
        ]
        output = format_results(results, explain=False)
        assert "authenticate" in output
        assert "src/auth.py" in output
        assert "function" in output

    def test_format_with_explain(self):
        """Explain mode should include score details."""
        results = [
            SearchResult(
                text="some code",
                score=0.85,
                collection="code",
                metadata={"symbol_name": "x", "file_path": "x.py"},
            ),
        ]
        output = format_results(results, explain=True)
        assert "0.85" in output or "score" in output.lower()

    def test_format_empty(self):
        """Empty results should produce a meaningful message."""
        output = format_results([], explain=False)
        assert "no results" in output.lower() or "0 results" in output.lower()


class TestFormatExplain:
    """Test the explain_retrieval diagnostic output."""

    def test_explain_output(self):
        """Should show score breakdown for a result."""
        result = SearchResult(
            text="def validate(data)",
            score=0.88,
            collection="code",
            metadata={"symbol_name": "validate", "file_path": "val.py"},
        )
        output = format_explain("validate input", result)
        assert "validate" in output
        assert "0.88" in output or "score" in output.lower()
