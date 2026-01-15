import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import sys


# Mock google.adk to avoid import issues in tests
mock_adk_tools = MagicMock()
sys.modules["google.adk.tools"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool.mcp_toolset"] = mock_adk_tools
sys.modules["google.adk.tools.mcp_tool.mcp_session_manager"] = mock_adk_tools


class TestSandboxManager:
    def test_init_stores_config(self):
        """Should store workspace and tool configuration"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=False,
        )

        assert "/tmp/test" in manager.workspace  # resolved path
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


class TestSandboxManagerDockerCheck:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_raises_when_docker_unavailable(self, mock_docker):
        """Should raise RuntimeError when Docker is not available"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_docker.from_env.side_effect = Exception("Docker not running")

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
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
            workspace="/tmp/test",
            enable_filesystem=True,
        )

        with pytest.raises(RuntimeError, match="Docker"):
            asyncio.run(manager.start())


class TestSandboxManagerContainer:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_start_creates_container(self, mock_docker):
        """Should create Docker container on start"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=False,  # Disable to skip MCP init
        )

        asyncio.run(manager.start())

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["detach"] is True
        assert "/workspace" in str(call_kwargs["volumes"])

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_removes_container(self, mock_docker):
        """Should stop and remove container on stop"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(workspace="/tmp/test")

        asyncio.run(manager.start())
        asyncio.run(manager.stop())

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_without_start_is_safe(self, mock_docker):
        """Should handle stop without start gracefully"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(workspace="/tmp/test")

        asyncio.run(manager.stop())  # Should not raise

    @patch("app.sandbox.sandbox_manager.docker")
    def test_context_manager(self, mock_docker):
        """Should support async context manager"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(workspace="/tmp/test")

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
        mock_client.containers.run.return_value = MagicMock()

        manager = SandboxManager(workspace="/tmp/test")

        asyncio.run(manager.start())

        mock_client.images.build.assert_called_once()


class TestSandboxManagerTools:
    def test_get_planner_tools_empty_when_disabled(self):
        """Should return empty list when filesystem disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=False,
        )

        tools = manager.get_planner_tools()
        assert tools == []

    def test_get_executor_tools_empty_when_disabled(self):
        """Should return empty list when all tools disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=False,
            enable_shell=False,
        )

        tools = manager.get_executor_tools()
        assert tools == []

    @patch("app.sandbox.sandbox_manager.docker")
    def test_get_planner_tools_returns_toolset(self, mock_docker):
        """Should return filesystem toolset for planner when enabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
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
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=True,
        )

        asyncio.run(manager.start())

        tools = manager.get_executor_tools()
        assert len(tools) == 2  # filesystem + shell


class TestSandboxManagerUserMcp:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_user_mcp_servers_stored(self, mock_docker):
        """Should store user MCP server configs"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=[
                {"url": "https://example.com/mcp"},
                {"command": "npx", "args": ["-y", "@mcp/server-github"]},
            ],
        )

        assert len(manager.mcp_servers) == 2
        assert manager._user_toolsets == []  # Not initialized until start()

    @patch("app.sandbox.sandbox_manager.docker")
    def test_user_mcp_toolsets_created_on_start(self, mock_docker):
        """Should create user MCP toolsets on start"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.run.return_value = MagicMock()

        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=[
                {"url": "https://example.com/mcp"},
                {"command": "npx", "args": ["-y", "@mcp/server-github"]},
            ],
        )

        asyncio.run(manager.start())

        # User toolsets should be created
        assert len(manager._user_toolsets) == 2

    @patch("app.sandbox.sandbox_manager.docker")
    def test_user_mcp_included_in_executor_tools(self, mock_docker):
        """Should include user MCP toolsets in executor tools"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.run.return_value = MagicMock()

        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=[{"url": "https://example.com/mcp"}],
        )

        asyncio.run(manager.start())

        tools = manager.get_executor_tools()
        # Should include user toolset even without filesystem/shell enabled
        assert len(tools) == 1
