"""Tests for parse_streaming_chunks utility."""

from __future__ import annotations

import json


class TestParseStreamingChunks:
    def test_extracts_content_from_chunks(self) -> None:
        """應從 streaming chunks 中提取 content 文字"""
        from leed_retrieval.tools.leed_advisor import parse_streaming_chunks

        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
            {"choices": [{"delta": {"role": "assistant", "content": " world"}}]},
        ]
        raw = "".join(json.dumps(c) for c in chunks)
        assert parse_streaming_chunks(raw) == "Hello world"

    def test_skips_chunks_without_content(self) -> None:
        """沒有 content 的 chunk 應被跳過"""
        from leed_retrieval.tools.leed_advisor import parse_streaming_chunks

        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "data"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        raw = "".join(json.dumps(c) for c in chunks)
        assert parse_streaming_chunks(raw) == "data"

    def test_returns_empty_string_for_empty_input(self) -> None:
        """空字串輸入應回傳空字串"""
        from leed_retrieval.tools.leed_advisor import parse_streaming_chunks

        assert parse_streaming_chunks("") == ""

    def test_handles_non_chunk_json_gracefully(self) -> None:
        """非 streaming chunk 格式的 JSON 應被跳過"""
        from leed_retrieval.tools.leed_advisor import parse_streaming_chunks

        good = json.dumps({"choices": [{"delta": {"content": "ok"}}]})
        bad = json.dumps({"error": "something"})
        raw = bad + good
        assert parse_streaming_chunks(raw) == "ok"
