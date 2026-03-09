"""Tests for leed_query FunctionTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestLeedQuery:
    def _make_mcp_response(self, text_content: str) -> MagicMock:
        """Helper: build a mock requests.Response with MCP JSON-RPC result."""
        chunks = [
            json.dumps({"choices": [{"delta": {"content": tok}}]}) for tok in text_content.split()
        ]
        raw_text = "".join(chunks)
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": raw_text}]},
        }
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_parsed_content(self) -> None:
        """leed_query 應回傳 parse 後的乾淨文字"""
        from leed_retrieval.tools.leed_advisor import leed_query

        mock_resp = self._make_mcp_response("Hello world")
        with patch("leed_retrieval.tools.leed_advisor.requests.post", return_value=mock_resp):
            result = leed_query(query="test query")
        # Helper splits by whitespace, each token is a chunk without spaces
        assert result == "Helloworld"

    def test_sends_correct_mcp_payload(self) -> None:
        """應發送正確的 MCP JSON-RPC 格式"""
        from leed_retrieval.tools.leed_advisor import leed_query

        mock_resp = self._make_mcp_response("ok")
        with patch(
            "leed_retrieval.tools.leed_advisor.requests.post", return_value=mock_resp
        ) as mock_post:
            leed_query(query="my question")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "leed_orchestrator"
        assert payload["params"]["arguments"]["query"] == "my question"

    def test_is_function_tool(self) -> None:
        """leed_advisor_tool 應為 FunctionTool 實例"""
        from google.adk.tools import FunctionTool

        from leed_retrieval.tools.leed_advisor import leed_advisor_tool

        assert isinstance(leed_advisor_tool, FunctionTool)
