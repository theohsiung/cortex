"""Excalidraw MCP Toolset — connects to mcp_excalidraw via stdio transport."""

from __future__ import annotations

import os

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

_server_url = os.getenv("EXCALIDRAW_SERVER_URL", "http://localhost:3377")
_mcp_path = os.getenv(
    "EXCALIDRAW_MCP_PATH",
    os.path.expanduser("~/projects/mcp_excalidraw/dist/index.js"),
)

excalidraw_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=[_mcp_path],
            env={
                **os.environ,
                "EXPRESS_SERVER_URL": _server_url,
                "ENABLE_CANVAS_SYNC": "true",
            },
        ),
        timeout=30.0,
    ),
)
