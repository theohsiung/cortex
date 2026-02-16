"""Pydantic configuration models for cortex."""
from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Model and Sandbox configuration
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """LLM model configuration."""

    name: str
    api_base: str
    api_key_env_var: str = ""
    temperature: float = 0.0

    def resolve_api_key(self) -> str | None:
        """Return the API key from the environment, or None."""
        if not self.api_key_env_var:
            return None
        return os.environ.get(self.api_key_env_var)


class SandboxConfig(BaseModel):
    """Sandbox execution environment configuration."""

    enable_filesystem: bool = False
    enable_shell: bool = False
    docker_image: str = "cortex-sandbox:latest"
    user_id: str | None = None


# ---------------------------------------------------------------------------
# Executor and Tuning configuration
# ---------------------------------------------------------------------------

class ExecutorEntry(BaseModel):
    """Maps an intent to a concrete executor factory."""

    intent: str
    description: str
    factory_module: str
    factory_function: str = "create_agent"
    is_default: bool = False


class TuningConfig(BaseModel):
    """Runtime tuning knobs."""

    max_concurrent_steps: int = Field(default=3, ge=0)
    max_retries: int = Field(default=3, ge=0)
    max_replan_attempts: int = Field(default=2, ge=0)
