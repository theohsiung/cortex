"""LEED Retrieval test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _leed_env_vars(monkeypatch):
    """Set LEED MCP env vars for tests."""
    monkeypatch.setenv("LEED_MCP_URL", "http://test.example.com/mcp/LEED_Advisor/")
    monkeypatch.setenv("LEED_MCP_TOKEN", "test-token-123")
