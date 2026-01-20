import uuid
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
    SHELL_SERVER = "/opt/mcp_servers/shell_server.py"

    # Userspace directory for persistent file storage
    USERSPACE_DIR = Path(__file__).parent / "userspace"

    # Read-only filesystem operations for PlannerAgent
    READONLY_FILESYSTEM_TOOLS = ["read_file", "list_directory", "get_file_info"]

    def __init__(
        self,
        user_id: str | None = None,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] | None = None,
        docker_image: str | None = None,
    ):
        """
        Initialize SandboxManager.

        Args:
            user_id: User identifier for userspace directory. Auto-generated if not provided.
            enable_filesystem: Enable built-in filesystem MCP tool (local, uses @anthropic/mcp-filesystem)
            enable_shell: Enable built-in shell MCP tool (runs in Docker container)
            mcp_servers: List of user-provided MCP server configs
            docker_image: Custom Docker image (default: cortex-sandbox:latest)
        """
        # Generate user_id if not provided
        self.user_id = user_id or f"auto-{uuid.uuid4().hex[:8]}"
        self.user_workspace = self.USERSPACE_DIR / self.user_id

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
        """Start Docker container (if shell enabled) and initialize MCP toolsets."""
        # Ensure userspace directory exists
        self.user_workspace.mkdir(parents=True, exist_ok=True)

        # Start Docker container only if shell is enabled
        if self.enable_shell:
            await self._start_container()

        # Initialize MCP toolsets
        await self._init_toolsets()

    async def _start_container(self):
        """Start Docker container for shell execution."""
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

        # Create container with user_workspace mounted
        self._container = self._client.containers.run(
            self.docker_image,
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            volumes={
                str(self.user_workspace): {"bind": "/workspace", "mode": "rw"}
            },
            working_dir="/workspace",
            auto_remove=False,
        )

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
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp import StdioServerParameters

        # Create filesystem toolset (local, using @anthropic/mcp-filesystem)
        if self.enable_filesystem:
            user_workspace_str = str(self.user_workspace.resolve())

            # Full access for executor
            self._filesystem_toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=["-y", "@anthropic/mcp-filesystem", user_workspace_str],
                    )
                )
            )

            # Read-only version for planner
            self._filesystem_toolset_readonly = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=["-y", "@anthropic/mcp-filesystem", user_workspace_str],
                    )
                ),
                tool_filter=self.READONLY_FILESYSTEM_TOOLS,
            )

        # Create shell toolset (Docker)
        if self.enable_shell and self._container is not None:
            container_id = self._container.id
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
        if self._container is not None:
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
