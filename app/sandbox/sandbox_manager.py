from typing import Any, TYPE_CHECKING
from pathlib import Path

import docker

if TYPE_CHECKING:
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


class SandboxManager:
    """Manages Docker container and MCP toolsets for sandboxed execution."""

    # Default Docker image for sandbox
    DEFAULT_IMAGE = "cortex-sandbox:latest"

    # Path to MCP servers in container
    FILESYSTEM_SERVER = "/opt/mcp_servers/filesystem_server.py"
    SHELL_SERVER = "/opt/mcp_servers/shell_server.py"

    # Read-only filesystem operations for PlannerAgent
    READONLY_FILESYSTEM_TOOLS = ["read_file", "list_directory", "get_file_info"]

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
            docker_image: Custom Docker image (default: cortex-sandbox:latest)
        """
        self.workspace = str(Path(workspace).resolve())
        self.enable_filesystem = enable_filesystem
        self.enable_shell = enable_shell
        self.mcp_servers = mcp_servers or []
        self.docker_image = docker_image or self.DEFAULT_IMAGE
        self._container = None
        self._client = None
        self._filesystem_toolset: "McpToolset | None" = None
        self._filesystem_toolset_readonly: "McpToolset | None" = None
        self._shell_toolset: "McpToolset | None" = None
        self._user_toolsets: list[Any] = []

    async def start(self):
        """Start Docker container and initialize MCP toolsets."""
        if self._container is not None:
            return

        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Docker is not available. Please ensure Docker is installed and running. Error: {e}"
            )

        # Build sandbox image if not exists
        await self._ensure_image()

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

        # Initialize MCP toolsets
        await self._init_toolsets()

    async def _ensure_image(self):
        """Ensure the sandbox Docker image exists, build if necessary."""
        try:
            self._client.images.get(self.docker_image)
        except docker.errors.ImageNotFound:
            # Build the image from Dockerfile
            dockerfile_dir = Path(__file__).parent
            self._client.images.build(
                path=str(dockerfile_dir),
                tag=self.docker_image,
                rm=True,
            )

    async def _init_toolsets(self):
        """Initialize MCP toolsets based on configuration."""
        if self._container is None:
            return

        # Lazy import to avoid namespace issues
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp import StdioServerParameters

        container_id = self._container.id

        # Create filesystem toolset (full access for executor)
        if self.enable_filesystem:
            self._filesystem_toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", "-i", container_id, "python", self.FILESYSTEM_SERVER],
                    )
                )
            )
            # Read-only version for planner
            self._filesystem_toolset_readonly = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", "-i", container_id, "python", self.FILESYSTEM_SERVER],
                    )
                ),
                tool_filter=self.READONLY_FILESYSTEM_TOOLS,
            )

        # Create shell toolset
        if self.enable_shell:
            self._shell_toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", "-i", container_id, "python", self.SHELL_SERVER],
                    )
                )
            )

        # Create user MCP toolsets
        await self._init_user_toolsets()

    async def _init_user_toolsets(self):
        """Initialize user-provided MCP toolsets."""
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import (
            StdioConnectionParams,
            SseConnectionParams,
        )
        from mcp import StdioServerParameters

        for server_config in self.mcp_servers:
            toolset = None

            if "url" in server_config:
                # SSE server (remote)
                toolset = McpToolset(
                    connection_params=SseConnectionParams(
                        url=server_config["url"],
                        headers=server_config.get("headers"),
                    )
                )
            elif "command" in server_config:
                # Stdio server (local, runs outside container)
                toolset = McpToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command=server_config["command"],
                            args=server_config.get("args", []),
                            env=server_config.get("env"),
                        )
                    )
                )

            if toolset is not None:
                self._user_toolsets.append(toolset)

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
            self._filesystem_toolset = None
            self._filesystem_toolset_readonly = None
            self._shell_toolset = None
            self._user_toolsets = []

    def get_planner_tools(self) -> list:
        """Get tools for PlannerAgent (read-only filesystem)."""
        tools = []
        if self._filesystem_toolset_readonly is not None:
            tools.append(self._filesystem_toolset_readonly)
        return tools

    def get_executor_tools(self) -> list:
        """Get tools for ExecutorAgent (all tools)."""
        tools = []
        if self._filesystem_toolset is not None:
            tools.append(self._filesystem_toolset)
        if self._shell_toolset is not None:
            tools.append(self._shell_toolset)
        tools.extend(self._user_toolsets)
        return tools

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False
