"""Tests for GeneralTask ToolManager."""

from __future__ import annotations


class TestToolManager:
    def test_tool_manager_is_singleton(self) -> None:
        """ToolManager 應為單例"""
        from general_task.tools._manager import ToolManager

        tm1 = ToolManager()
        tm2 = ToolManager()
        assert tm1 is tm2

    def test_tool_manager_loads_tools(self) -> None:
        """ToolManager 應自動載入 12 個工具"""
        from general_task.tools._manager import tool_manager

        tools = tool_manager.get_all_tools()
        assert len(tools) == 12

    def test_tool_manager_loads_prompts(self) -> None:
        """ToolManager 應自動載入 12 個 prompt 說明"""
        from general_task.tools._manager import tool_manager

        prompts = tool_manager.get_all_tool_prompts()
        assert len(prompts) == 12

    def test_all_expected_tools_present(self) -> None:
        """確認 12 個預期工具都有載入"""
        from general_task.tools._manager import tool_manager

        tool_names = {t.name for t in tool_manager.get_all_tools()}
        expected = {
            "web_search",
            "web_browser",
            "download_file",
            "file_reader",
            "pdf_reader",
            "excel_reader",
            "pptx_reader",
            "zip_extractor",
            "audio_transcription",
            "python_executor",
            "calculator",
            "submit_final_answer",
        }
        assert expected == tool_names
