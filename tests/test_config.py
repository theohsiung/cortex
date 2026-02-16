"""Tests for app.config Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.config import MCPStdio, MCPSse, MCPServer


# ---------------------------------------------------------------------------
# Task 2 – MCP Discriminated Union Models
# ---------------------------------------------------------------------------

class TestMCPStdio:
    """MCPStdio model tests."""

    def test_valid_config(self):
        cfg = MCPStdio(transport="stdio", command="/usr/bin/node", args=["server.js"])
        assert cfg.transport == "stdio"
        assert cfg.command == "/usr/bin/node"
        assert cfg.args == ["server.js"]

    def test_defaults(self):
        cfg = MCPStdio(transport="stdio", command="python")
        assert cfg.args == []
        assert cfg.env == {}

    def test_with_env(self):
        cfg = MCPStdio(
            transport="stdio",
            command="node",
            env={"NODE_ENV": "production"},
        )
        assert cfg.env == {"NODE_ENV": "production"}

    def test_rejects_wrong_transport(self):
        with pytest.raises(ValidationError):
            MCPStdio(transport="sse", command="python")


class TestMCPSse:
    """MCPSse model tests."""

    def test_valid_config(self):
        cfg = MCPSse(transport="sse", url="http://localhost:8080/sse")
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:8080/sse"

    def test_with_headers(self):
        cfg = MCPSse(
            transport="sse",
            url="http://localhost:8080/sse",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_rejects_wrong_transport(self):
        with pytest.raises(ValidationError):
            MCPSse(transport="stdio", url="http://localhost:8080/sse")


class TestMCPServerDiscriminator:
    """Discriminated union via MCPServer type alias."""

    ta = TypeAdapter(MCPServer)

    def test_routes_to_stdio(self):
        obj = self.ta.validate_python(
            {"transport": "stdio", "command": "node", "args": ["index.js"]}
        )
        assert isinstance(obj, MCPStdio)

    def test_routes_to_sse(self):
        obj = self.ta.validate_python(
            {"transport": "sse", "url": "http://localhost:8080/sse"}
        )
        assert isinstance(obj, MCPSse)

    def test_rejects_unknown_transport(self):
        with pytest.raises(ValidationError):
            self.ta.validate_python({"transport": "grpc", "command": "foo"})

    def test_list_of_mixed_servers(self):
        ta_list = TypeAdapter(list[MCPServer])
        servers = ta_list.validate_python([
            {"transport": "stdio", "command": "node"},
            {"transport": "sse", "url": "http://localhost:8080/sse"},
        ])
        assert isinstance(servers[0], MCPStdio)
        assert isinstance(servers[1], MCPSse)
