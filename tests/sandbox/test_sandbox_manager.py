"""Tests for the SandboxManager."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.config import MCPSse, MCPStdio, SandboxConfig

# Mock google.adk to avoid import issues in tests
mock_adk_tools = MagicMock()
sys.modules["google.adk.tools"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool.mcp_toolset"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool.mcp_session_manager"] = mock_adk_tools


class TestSandboxManager:
    def test_init_with_user_id(self):
        """Should store user_id and create userspace path"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="alice", enable_filesystem=True, enable_shell=False),
        )

        assert manager.user_id == "alice"
        assert "userspace/alice" in str(manager.user_workspace)
        assert manager.enable_filesystem is True
        assert manager.enable_shell is False
        assert manager.mcp_servers == []

    def test_init_auto_generates_user_id(self):
        """Should auto-generate user_id if not provided"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(SandboxConfig(enable_filesystem=True))

        assert manager.user_id.startswith("auto-")
        assert len(manager.user_id) == 13  # "auto-" + 8 hex chars

    def test_init_with_mcp_servers(self):
        """Should store user MCP server configs"""
        from app.sandbox.sandbox_manager import SandboxManager

        servers = [MCPSse(transport="sse", url="https://example.com/mcp")]
        manager = SandboxManager(
            SandboxConfig(user_id="test"),
            mcp_servers=servers,
        )

        assert manager.mcp_servers == servers


class TestSandboxManagerDockerCheck:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_raises_when_docker_unavailable(self, mock_docker):
        """Should raise RuntimeError when Docker is not available (shell enabled)"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_docker.from_env.side_effect = Exception("Docker not running")

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_shell=True),
        )

        with pytest.raises(RuntimeError, match="Docker"):
            asyncio.run(manager.start())

    @patch("app.sandbox.sandbox_manager.docker")
    def test_raises_when_docker_ping_fails(self, mock_docker):
        """Should raise RuntimeError when Docker ping fails"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.side_effect = Exception("Connection refused")

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_shell=True),
        )

        with pytest.raises(RuntimeError, match="Docker"):
            asyncio.run(manager.start())

    def test_filesystem_only_does_not_require_docker(self):
        """Should not require Docker when only filesystem is enabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_filesystem=True, enable_shell=False),
        )

        # Should not raise - filesystem uses local @anthropic/mcp-filesystem
        asyncio.run(manager.start())
        assert manager._container is None  # No Docker container created


class TestSandboxManagerContainer:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_start_creates_container_when_shell_enabled(self, mock_docker):
        """Should create Docker container only when shell is enabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"  # Must be a string for StdioServerParameters
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_shell=True),
        )

        asyncio.run(manager.start())

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["detach"] is True
        assert "/workspace" in str(call_kwargs["volumes"])

    @patch("app.sandbox.sandbox_manager.docker")
    def test_container_mounts_user_workspace(self, mock_docker):
        """Should mount user_workspace to /workspace in container"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            SandboxConfig(user_id="alice", enable_shell=True),
        )

        asyncio.run(manager.start())

        call_kwargs = mock_client.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        # Should mount userspace/alice to /workspace
        assert any("userspace/alice" in k for k in volumes.keys())

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_removes_container(self, mock_docker):
        """Should stop and remove container on stop"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(SandboxConfig(user_id="test", enable_shell=True))

        asyncio.run(manager.start())
        asyncio.run(manager.stop())

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    def test_stop_without_start_is_safe(self):
        """Should handle stop without start gracefully"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(SandboxConfig(user_id="test"))

        asyncio.run(manager.stop())  # Should not raise

    @patch("app.sandbox.sandbox_manager.docker")
    def test_context_manager(self, mock_docker):
        """Should support async context manager"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(SandboxConfig(user_id="test", enable_shell=True))

        async def test():
            async with manager:
                assert manager._container is not None
            mock_container.stop.assert_called_once()

        asyncio.run(test())

    @patch("app.sandbox.sandbox_manager.docker")
    def test_ensure_image_builds_if_not_found(self, mock_docker):
        """Should build image if not found"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.ImageNotFound = Exception
        mock_client.images.get.side_effect = Exception("Image not found")
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(SandboxConfig(user_id="test", enable_shell=True))

        asyncio.run(manager.start())

        mock_client.images.build.assert_called_once()


class TestSandboxManagerTools:
    def test_get_planner_tools_empty_when_disabled(self):
        """Should return empty list when filesystem disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_filesystem=False),
        )

        tools = manager.get_planner_tools()
        assert tools == []

    def test_get_executor_tools_empty_when_disabled(self):
        """Should return empty list when all tools disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_filesystem=False, enable_shell=False),
        )

        tools = manager.get_executor_tools()
        assert tools == []

    def test_get_planner_tools_returns_toolset(self):
        """Should return filesystem toolset for planner when enabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_filesystem=True),
        )

        asyncio.run(manager.start())

        tools = manager.get_planner_tools()
        assert len(tools) == 1  # read-only filesystem toolset

    @patch("app.sandbox.sandbox_manager.docker")
    def test_get_executor_tools_returns_all_toolsets(self, mock_docker):
        """Should return all toolsets for executor"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            SandboxConfig(user_id="test", enable_filesystem=True, enable_shell=True),
        )

        asyncio.run(manager.start())

        tools = manager.get_executor_tools()
        assert len(tools) == 2  # filesystem + shell


class TestSandboxManagerUserMcp:
    def test_user_mcp_servers_stored(self):
        """Should store user MCP server configs"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test"),
            mcp_servers=[
                MCPSse(transport="sse", url="https://example.com/mcp"),
                MCPStdio(transport="stdio", command="npx", args=["-y", "@mcp/server-github"]),
            ],
        )

        assert len(manager.mcp_servers) == 2
        assert manager._user_toolsets == []  # Not initialized until start()

    def test_user_mcp_toolsets_created_on_start(self):
        """Should create user MCP toolsets on start"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test"),
            mcp_servers=[
                MCPSse(transport="sse", url="https://example.com/mcp"),
                MCPStdio(transport="stdio", command="npx", args=["-y", "@mcp/server-github"]),
            ],
        )

        asyncio.run(manager.start())

        # User toolsets should be created
        assert len(manager._user_toolsets) == 2

    def test_user_mcp_included_in_executor_tools(self):
        """Should include user MCP toolsets in executor tools"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            SandboxConfig(user_id="test"),
            mcp_servers=[MCPSse(transport="sse", url="https://example.com/mcp")],
        )

        asyncio.run(manager.start())

        tools = manager.get_executor_tools()
        # Should include user toolset even without filesystem/shell enabled
        assert len(tools) == 1


class TestSandboxManagerUserspace:
    def test_userspace_directory_created_on_start(self):
        """Should create userspace directory on start"""
        import tempfile
        from pathlib import Path

        from app.sandbox.sandbox_manager import SandboxManager

        # Use temp dir to avoid polluting real userspace
        with patch.object(SandboxManager, "USERSPACE_DIR", Path(tempfile.mkdtemp())):
            manager = SandboxManager(SandboxConfig(user_id="testuser"))
            asyncio.run(manager.start())

            assert manager.user_workspace.exists()
            assert manager.user_workspace.name == "testuser"

    def test_same_user_id_uses_same_directory(self):
        """Same user_id should use the same userspace directory"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager1 = SandboxManager(SandboxConfig(user_id="alice"))
        manager2 = SandboxManager(SandboxConfig(user_id="alice"))

        assert manager1.user_workspace == manager2.user_workspace
