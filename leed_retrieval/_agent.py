"""LEED Retrieval Agent implementation."""

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


class LeedRetrievalFlow(BaseLlmFlow):
    """Custom flow for LEED Retrieval Agent."""

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


class LeedRetrievalAgent(LlmAgent):
    """LEED Retrieval Agent using LeedRetrievalFlow."""

    @property
    def _llm_flow(self) -> BaseLlmFlow:
        return LeedRetrievalFlow()


def create_leed_retrieval_agent(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LeedRetrievalAgent:
    """Create a LEED Retrieval Agent.

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
    if agent_config.mcp_url is None:
        agent_config.mcp_url = os.getenv("LEED_MCP_URL")
    if agent_config.mcp_token is None:
        agent_config.mcp_token = os.getenv("LEED_MCP_TOKEN")

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
        # gpt-oss-20b is not in litellm's model registry, so it defaults to a
        # tiny context window and computes negative max_tokens. Setting
        # max_tokens explicitly avoids this.
        max_tokens=4096,
    )

    # Never include aliases — this agent uses MCP tools which already
    # consume significant context window, aliases would cause token overflow.
    agent = LeedRetrievalAgent(
        model=model,
        tools=tool_manager.get_all_tools(include_aliases=False),  # type: ignore[arg-type]
        name="LeedRetrievalAgent",
        instruction=build_system_prompt,
    )

    return agent
