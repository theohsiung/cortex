"""Shared test fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock google.adk.agents before any imports
# This prevents google.adk from trying to create directories
mock_adk_agents = MagicMock()
mock_adk_agents.LlmAgent = Mock
mock_adk_agents.LoopAgent = Mock
mock_adk_agents.SequentialAgent = Mock
mock_adk_agents.ParallelAgent = Mock


class MockFunctionDeclaration:
    """Mock replacement for google.genai.types.FunctionDeclaration."""

    def __init__(self, name: str, description: str | None = None, parameters: dict | None = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


mock_genai_types = MagicMock()
mock_genai_types.FunctionDeclaration = MockFunctionDeclaration

sys.modules['google.adk'] = MagicMock()
sys.modules['google.adk.agents'] = mock_adk_agents
sys.modules['google.adk.models'] = MagicMock()
sys.modules['google.adk.runners'] = MagicMock()
sys.modules['google.adk.sessions'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = mock_genai_types


@pytest.fixture(autouse=True)
def _no_toml_file(monkeypatch):
    """Prevent tests from accidentally loading a real config.toml."""
    monkeypatch.setattr("app.config.get_config_file_path", lambda: None)
