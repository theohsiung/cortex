"""Tests for LEED Advisor tool configuration."""

from __future__ import annotations


class TestLeedAdvisorTool:
    def test_leed_query_requires_query_param(self) -> None:
        """leed_query 函式應接受 query 參數"""
        import inspect

        from leed_retrieval.tools.leed_advisor import leed_query

        sig = inspect.signature(leed_query)
        assert "query" in sig.parameters

    def test_leed_advisor_tool_name(self) -> None:
        """leed_advisor_tool 的函式名稱應為 leed_query"""
        from leed_retrieval.tools.leed_advisor import leed_advisor_tool

        assert leed_advisor_tool.name == "leed_query"
