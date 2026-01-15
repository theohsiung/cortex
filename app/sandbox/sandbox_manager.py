from typing import Any


class SandboxManager:
    """Manages Docker container and MCP toolsets for sandboxed execution."""

    def __init__(
        self,
        workspace: str,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] | None = None,
    ):
        """
        Initialize SandboxManager.

        Args:
            workspace: Directory to mount into container
            enable_filesystem: Enable built-in filesystem MCP tool
            enable_shell: Enable built-in shell MCP tool
            mcp_servers: List of user-provided MCP server configs
        """
        self.workspace = workspace
        self.enable_filesystem = enable_filesystem
        self.enable_shell = enable_shell
        self.mcp_servers = mcp_servers or []
        self._container = None
        self._toolsets: list[Any] = []
