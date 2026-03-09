"""Tests for LEED Retrieval system prompt."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestSystemPrompt:
    def test_build_system_prompt_returns_string(self) -> None:
        """build_system_prompt 應回傳字串"""
        from leed_retrieval._prompt import build_system_prompt

        ctx = MagicMock()
        result = build_system_prompt(ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_contains_leed_context(self) -> None:
        """system prompt 應包含 LEED 相關內容"""
        from leed_retrieval._prompt import build_system_prompt

        ctx = MagicMock()
        result = build_system_prompt(ctx)
        assert "LEED" in result

    def test_prompt_contains_tool_prompts(self) -> None:
        """system prompt 應包含 tool prompts"""
        from leed_retrieval._prompt import build_system_prompt

        ctx = MagicMock()
        result = build_system_prompt(ctx)
        assert "submit_final_answer" in result
