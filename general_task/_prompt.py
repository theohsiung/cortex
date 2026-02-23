"""System Prompt builder for General Task Agent."""

from __future__ import annotations

import sys

from google.adk.agents.readonly_context import ReadonlyContext

from ._config import agent_config
from .tools import tool_manager

GENERAL_TASK_CLI_PROMPT = """You are a General Task Agent powered by an LLM, designed to handle a wide variety of tasks.

You can:
- Search the web for information
- Read and analyze files (PDF, Excel, Word, PowerPoint, ZIP, audio)
- Execute Python code for data processing and calculations
- Download files from URLs
- Submit final answers when the task is complete

Answer the user's request using the relevant tool(s) when available.
Break complex tasks into steps and execute them systematically.

CRITICAL RULES:
1. NEVER fabricate or hallucinate information. Every fact, headline, URL, number, or data point in your answer MUST come from an actual tool call result in this session. If you do not have real data from tools, say so explicitly instead of making something up.
2. You are executing ONE step of a larger multi-step plan. Focus only on completing the assigned step. Do not attempt to answer the entire task or skip ahead to future steps.
3. When the step requires searching, retrieving, or finding information, you MUST call web_search or web_browser first and base your answer on those results. Never submit information recalled from training data as if it were freshly searched.
4. Only call submit_final_answer after you have gathered real evidence from tool calls. The content of submit_final_answer must reflect only what tools actually returned.
"""


def _get_platform_info() -> str:
    platform_names = {
        "win32": "Windows",
        "darwin": "macOS",
        "linux": "Linux",
    }
    platform_name = platform_names.get(sys.platform, "Unix-like")
    return f"Operating system: {platform_name}"


class SystemPrompt:
    @property
    def cli_prompt(self) -> str:
        return GENERAL_TASK_CLI_PROMPT

    @property
    def model_name_prompt(self) -> str:
        return f"Your model name is: {agent_config.model_name}"

    @property
    def platform_prompt(self) -> str:
        return _get_platform_info()

    @property
    def tools_prompt(self) -> str:
        tool_prompts = tool_manager.get_all_tool_prompts()
        return "\n---\n".join(tool_prompts)


def build_system_prompt(readonly_context: ReadonlyContext) -> str:
    # readonly_context is required by the ADK instruction callable interface
    # but not used here since this prompt is statically constructed.
    system_prompt = SystemPrompt()
    sections = [
        system_prompt.cli_prompt,
        system_prompt.model_name_prompt,
        system_prompt.platform_prompt,
        system_prompt.tools_prompt,
    ]
    return "\n\n".join(sections)
