"""Shared test fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Stub classes for google.adk types that need to be subclassable / inspectable
# ---------------------------------------------------------------------------


class MockLlmAgent:
    """Stub for google.adk.agents.llm_agent.LlmAgent."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockBaseLlmFlow:
    """Stub for google.adk.flows.llm_flows.base_llm_flow.BaseLlmFlow."""

    def __init__(self):
        self.request_processors = []
        self.response_processors = []


class MockFunctionTool:
    """Stub for google.adk.tools.FunctionTool.

    Derives the tool name from the wrapped function's __name__.
    """

    def __init__(self, func, **kwargs):
        self._func = func
        self.name = func.__name__

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class MockFunctionDeclaration:
    """Mock replacement for google.genai.types.FunctionDeclaration."""

    def __init__(
        self, name: str, description: str | None = None, parameters: dict | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


# ---------------------------------------------------------------------------
# Build mock modules
# ---------------------------------------------------------------------------

mock_adk_agents_llm_agent = MagicMock()
mock_adk_agents_llm_agent.LlmAgent = MockLlmAgent

mock_adk_agents = MagicMock()
mock_adk_agents.LlmAgent = MockLlmAgent
mock_adk_agents.LoopAgent = Mock
mock_adk_agents.SequentialAgent = Mock
mock_adk_agents.ParallelAgent = Mock

mock_adk_flows_base_llm_flow = MagicMock()
mock_adk_flows_base_llm_flow.BaseLlmFlow = MockBaseLlmFlow

mock_adk_tools = MagicMock()
mock_adk_tools.FunctionTool = MockFunctionTool

mock_genai_types = MagicMock()
mock_genai_types.FunctionDeclaration = MockFunctionDeclaration

sys.modules["google.adk"] = MagicMock()
sys.modules["google.adk.agents"] = mock_adk_agents
sys.modules["google.adk.agents.llm_agent"] = mock_adk_agents_llm_agent
sys.modules["google.adk.agents.readonly_context"] = MagicMock()
sys.modules["google.adk.auth"] = MagicMock()
sys.modules["google.adk.auth.auth_preprocessor"] = MagicMock()
sys.modules["google.adk.flows"] = MagicMock()
sys.modules["google.adk.flows.llm_flows"] = MagicMock()
sys.modules["google.adk.flows.llm_flows.base_llm_flow"] = mock_adk_flows_base_llm_flow
sys.modules["google.adk.flows.llm_flows.basic"] = MagicMock()
sys.modules["google.adk.flows.llm_flows._code_execution"] = MagicMock()
sys.modules["google.adk.flows.llm_flows._nl_planning"] = MagicMock()
sys.modules["google.adk.flows.llm_flows._output_schema_processor"] = MagicMock()
sys.modules["google.adk.flows.llm_flows.contents"] = MagicMock()
sys.modules["google.adk.flows.llm_flows.context_cache_processor"] = MagicMock()
sys.modules["google.adk.flows.llm_flows.instructions"] = MagicMock()
sys.modules["google.adk.flows.llm_flows.request_confirmation"] = MagicMock()
sys.modules["google.adk.models"] = MagicMock()
sys.modules["google.adk.models.lite_llm"] = MagicMock()
sys.modules["google.adk.runners"] = MagicMock()
sys.modules["google.adk.sessions"] = MagicMock()
sys.modules["google.adk.tools"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool"] = MagicMock()
sys.modules["google.adk.tools.mcp_tool.mcp_toolset"] = MagicMock()
sys.modules["google.adk.tools.mcp_tool.mcp_session_manager"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = mock_genai_types


@pytest.fixture(autouse=True)
def _no_toml_file(monkeypatch):
    """Prevent tests from accidentally loading a real config.toml."""
    monkeypatch.setattr("app.config.get_config_file_path", lambda: None)
