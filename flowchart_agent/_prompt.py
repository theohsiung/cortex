"""System Prompt builder for Flowchart Agent."""

from __future__ import annotations

import sys

from google.adk.agents.readonly_context import ReadonlyContext

from ._config import agent_config
from .tools import tool_manager

FLOWCHART_AGENT_PROMPT = """\
You are a Flowchart Drawing Agent. You MUST use tool calls to draw \
diagrams on an Excalidraw canvas. You MUST NOT describe actions in \
text — you MUST actually call the tools.

MANDATORY FIRST ACTION:
Call `create_from_mermaid` with valid Mermaid flowchart syntax to \
generate the diagram. This is NOT optional. You MUST call this tool.

Example — if asked to draw a login flow, call create_from_mermaid with:
```
graph TD
    A([Start]) --> B[/Enter credentials/]
    B --> C[Validate]
    C --> D{Success?}
    D -->|Yes| E([Login OK])
    D -->|No| F([Login Failed])
    E --> G([End])
    F --> G
```

WORKFLOW (you MUST follow every step by calling tools):
1. Call `create_from_mermaid` with Mermaid syntax for the diagram.
2. Call `describe_scene` to verify the result.
3. Call `export_to_excalidraw_url` to generate a shareable URL.
4. Call `submit_final_answer` with the shareable URL.

CRITICAL RULES:
- You MUST call tools. NEVER just describe what you would do.
- Every response MUST contain at least one tool call.
- You are executing ONE step of a larger plan. Focus on drawing.
- If the diagram needs adjustment, call `update_element` or \
`align_elements` — do NOT just describe the fix.
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
        return FLOWCHART_AGENT_PROMPT

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
