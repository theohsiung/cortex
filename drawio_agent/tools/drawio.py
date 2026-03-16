"""Draw.io MCP Toolset — connects to next-ai-drawio MCP server via stdio transport."""

from __future__ import annotations

import os

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

drawio_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@next-ai-drawio/mcp-server@latest"],
            env={
                **os.environ,
            },
        ),
        timeout=60.0,
    ),
)
