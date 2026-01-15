import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSandboxManager:
    def test_init_stores_config(self):
        """Should store workspace and tool configuration"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=False,
        )

        assert manager.workspace == "/tmp/test"
        assert manager.enable_filesystem is True
        assert manager.enable_shell is False
        assert manager.mcp_servers == []

    def test_init_with_mcp_servers(self):
        """Should store user MCP server configs"""
        from app.sandbox.sandbox_manager import SandboxManager

        servers = [{"url": "https://example.com/mcp"}]
        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=servers,
        )

        assert manager.mcp_servers == servers
