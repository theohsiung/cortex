"""General Task Agent implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth import auth_preprocessor
from google.adk.flows.llm_flows import (
    _code_execution,
    _nl_planning,
    _output_schema_processor,
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

# 從環境變數初始化 config
agent_config.api_base = os.getenv("API_BASE")
agent_config.api_key = os.getenv("API_KEY")
agent_config.model_name = os.getenv("MODEL_NAME")
_output_dir_env = os.getenv("OUTPUT_DIR")
if _output_dir_env:
    agent_config.output_dir = _output_dir_env


class GeneralTaskFlow(BaseLlmFlow):
    """Custom flow for General Task Agent."""

    def __init__(self) -> None:
        super().__init__()
        self.request_processors += [
            basic.request_processor,
            auth_preprocessor.request_processor,
            request_confirmation.request_processor,
            instructions.request_processor,
            contents.request_processor,
            context_cache_processor.request_processor,
            _nl_planning.request_processor,
            _code_execution.request_processor,
            _output_schema_processor.request_processor,
        ]
        self.response_processors += [
            _nl_planning.response_processor,
            _code_execution.response_processor,
        ]


class GeneralTaskAgent(LlmAgent):
    """General Task Agent using GeneralTaskFlow."""

    @property
    def _llm_flow(self) -> BaseLlmFlow:
        return GeneralTaskFlow()


def create_general_task_agent(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> GeneralTaskAgent:
    """Create a General Task Agent.

    Args:
        model_name: 覆蓋 .env 中的 MODEL_NAME
        api_base: 覆蓋 .env 中的 API_BASE
        api_key: 覆蓋 .env 中的 API_KEY
        output_dir: 輸出目錄，用於儲存下載的檔案、Python 執行輸出等
    """
    if model_name:
        agent_config.model_name = model_name
    if api_base:
        agent_config.api_base = api_base
    if api_key:
        agent_config.api_key = api_key
    if output_dir:
        agent_config.output_dir = output_dir

    model = LiteLlm(
        model=agent_config.model_name or "",
        api_base=agent_config.api_base,
        api_key=agent_config.api_key,
    )

    agent = GeneralTaskAgent(
        model=model,
        tools=tool_manager.get_all_tools(),  # type: ignore[arg-type]
        name="GeneralTaskAgent",
        instruction=build_system_prompt,
    )

    return agent
