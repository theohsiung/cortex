"""Tests for LEED Retrieval ToolManager."""

from __future__ import annotations


class TestToolManager:
    def test_tool_manager_is_singleton(self) -> None:
        """ToolManager 應為單例"""
        from leed_retrieval.tools._manager import ToolManager

        tm1 = ToolManager()
        tm2 = ToolManager()
        assert tm1 is tm2

    def test_tool_manager_loads_function_tools(self) -> None:
        """ToolManager 應載入 submit_final_answer 和 leed_advisor FunctionTool"""
        from leed_retrieval.tools._manager import tool_manager

        tool_names = {t.name for t in tool_manager.tools}
        assert "submit_final_answer" in tool_names
        assert "leed_query" in tool_names

    def test_tool_manager_loads_prompts(self) -> None:
        """ToolManager 應載入 prompt 說明"""
        from leed_retrieval.tools._manager import tool_manager

        prompts = tool_manager.get_all_tool_prompts()
        assert len(prompts) >= 1

    def test_get_all_tools_returns_all_function_tools(self) -> None:
        """get_all_tools 應包含所有 FunctionTool"""
        from leed_retrieval.tools._manager import tool_manager

        all_tools = tool_manager.get_all_tools()
        assert len(all_tools) >= 2  # submit_final_answer + leed_query
