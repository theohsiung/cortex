from typing import Any

import docker


class SandboxManager:
    """Manages Docker container and MCP toolsets for sandboxed execution."""

    # Default Docker image for sandbox
    DEFAULT_IMAGE = "python:3.12-slim"

    def __init__(
        self,
        workspace: str,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] | None = None,
        docker_image: str | None = None,
    ):
        """
        Initialize SandboxManager.

        Args:
            workspace: Directory to mount into container
            enable_filesystem: Enable built-in filesystem MCP tool
            enable_shell: Enable built-in shell MCP tool
            mcp_servers: List of user-provided MCP server configs
            docker_image: Custom Docker image (default: python:3.12-slim)
        """
        self.workspace = workspace
        self.enable_filesystem = enable_filesystem
        self.enable_shell = enable_shell
        self.mcp_servers = mcp_servers or []
        self.docker_image = docker_image or self.DEFAULT_IMAGE
        self._container = None
        self._client = None
        self._toolsets: list[Any] = []

    async def start(self):
        """Start Docker container and initialize MCP toolsets."""
        if self._container is not None:
            return

        self._client = docker.from_env()

        # Create container with workspace mounted
        self._container = self._client.containers.run(
            self.docker_image,
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            volumes={
                self.workspace: {"bind": "/workspace", "mode": "rw"}
            },
            working_dir="/workspace",
            auto_remove=False,
        )

    async def stop(self):
        """Stop Docker container and cleanup."""
        if self._container is None:
            return

        try:
            self._container.stop(timeout=5)
            self._container.remove()
        except Exception:
            pass  # Container may already be stopped/removed
        finally:
            self._container = None
            self._toolsets = []

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False
