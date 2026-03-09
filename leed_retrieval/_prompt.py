"""System Prompt builder for LEED Retrieval Agent."""

from __future__ import annotations

import sys

from google.adk.agents.readonly_context import ReadonlyContext

from ._config import agent_config
from .tools import tool_manager

LEED_RETRIEVAL_PROMPT = """\
You are a LEED Retrieval Agent specialized in green building \
regulations and LEED certification standards.

You can:
- Search and retrieve LEED (Leadership in Energy and Environmental \
Design) related documents and regulations
- Answer questions about green building standards, credits, and \
compliance requirements
- Provide guidance on LEED certification processes
- Submit final answers when the task is complete

Answer the user's request using the LEED_Advisor tool to retrieve \
relevant information.
Break complex questions into steps and search systematically.

CRITICAL RULES:
1. NEVER fabricate or hallucinate information. Every fact in your \
answer MUST come from an actual tool call result in this session.
2. You are executing ONE step of a larger multi-step plan. Focus \
only on completing the assigned step.
3. Always use the LEED_Advisor tool to search for information \
before answering.
4. Only call submit_final_answer after you have gathered real \
evidence from tool calls.

TOOL SELECTION GUIDE:
- LEED_Advisor → Search and retrieve LEED green building regulation documents
- submit_final_answer → Submit the final result of the task
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
        return LEED_RETRIEVAL_PROMPT

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
    system_prompt = SystemPrompt()
    sections = [
        system_prompt.cli_prompt,
        system_prompt.model_name_prompt,
        system_prompt.platform_prompt,
        system_prompt.tools_prompt,
    ]
    return "\n\n".join(sections)
