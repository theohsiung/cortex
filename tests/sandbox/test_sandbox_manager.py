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
            enable_filesystem=True,
        )

        import asyncio
        asyncio.run(manager.start())

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["detach"] is True
        assert "/tmp/test" in str(call_kwargs["volumes"])

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_removes_container(self, mock_docker):
        """Should stop and remove container on stop"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(workspace="/tmp/test")

        import asyncio
        asyncio.run(manager.start())
        asyncio.run(manager.stop())

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_without_start_is_safe(self, mock_docker):
        """Should handle stop without start gracefully"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(workspace="/tmp/test")

        import asyncio
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

        import asyncio

        async def test():
            async with manager:
                assert manager._container is not None
            mock_container.stop.assert_called_once()

        asyncio.run(test())
