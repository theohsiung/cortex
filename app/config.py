"""Pydantic configuration models for cortex."""
from __future__ import annotations

from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# MCP Server Models (discriminated union)
# ---------------------------------------------------------------------------

class MCPStdio(BaseModel):
    """MCP server accessed over stdio transport."""

    transport: Literal["stdio"]
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class MCPSse(BaseModel):
    """MCP server accessed over SSE transport."""

    transport: Literal["sse"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


MCPServer = Annotated[MCPStdio | MCPSse, Field(discriminator="transport")]
