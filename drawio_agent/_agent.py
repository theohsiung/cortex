"""Draw.io Agent implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth import auth_preprocessor
from google.adk.flows.llm_flows import (
    basic,
    contents,
    context_cache_processor,
    instructions,
    request_confirmation,
)
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.lite_llm import LiteLlm

from ._config import agent_config
from ._prompt import build_system_prompt
from .tools import tool_manager

# 載入 .env 檔案
agent_dir = Path(__file__).parent
load_dotenv(agent_dir / ".env")


class DrawioFlow(BaseLlmFlow):
    """Custom flow for Draw.io Agent."""

    def __init__(self) -> None:
        super().__init__()
        self.request_processors += [
            basic.request_processor,
            auth_preprocessor.request_processor,
            request_confirmation.request_processor,
            instructions.request_processor,
            contents.request_processor,
            context_cache_processor.request_processor,
        ]


class DrawioAgent(LlmAgent):
    """Draw.io Agent using DrawioFlow."""

    @property
    def _llm_flow(self) -> BaseLlmFlow:
        return DrawioFlow()


def create_drawio_agent(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DrawioAgent:
    """Create a Draw.io Agent.

    Args:
        model_name: 覆蓋 .env 中的 MODEL_NAME
        api_base: 覆蓋 .env 中的 API_BASE
        api_key: 覆蓋 .env 中的 API_KEY
    """
    # Initialize from env vars
    if agent_config.api_base is None:
        agent_config.api_base = os.getenv("API_BASE")
    if agent_config.api_key is None:
        agent_config.api_key = os.getenv("API_KEY")
    if agent_config.model_name is None:
        agent_config.model_name = os.getenv("MODEL_NAME")

    # Apply explicit overrides
    if model_name:
        agent_config.model_name = model_name
    if api_base:
        agent_config.api_base = api_base
    if api_key:
        agent_config.api_key = api_key

    model = LiteLlm(
        model=agent_config.model_name or "",
        api_base=agent_config.api_base,
        api_key=agent_config.api_key,
        max_tokens=4096,
    )

    agent = DrawioAgent(
        model=model,
        tools=tool_manager.get_all_tools(include_aliases=False),  # type: ignore[arg-type]
        name="DrawioAgent",
        instruction=build_system_prompt,
    )

    return agent
