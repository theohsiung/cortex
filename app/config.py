"""Pydantic configuration models for cortex."""

from __future__ import annotations

import importlib
import os
import tomllib
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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

    name: str = Field(description="Model identifier, e.g. 'openai/gpt-oss-20b'")
    api_base: str = Field(description="API base URL")
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

    intent: str = Field(description="Intent name for routing, e.g. 'generate'")
    description: str = Field(description="Human-readable description")
    factory_module: str = Field(description="Dotted module path to the factory")
    factory_function: str = "create_agent"
    is_default: bool = False
    tool_names: list[str] = Field(
        default_factory=list,
        description="Tool names available to this executor, used by the replanner",
    )

    def get_factory(self) -> Callable[[], Any]:
        """Import the factory module and return the factory callable (without calling it)."""
        module = importlib.import_module(self.factory_module)
        factory: Callable[[], Any] = getattr(module, self.factory_function)
        return factory

    def create_executor(self) -> Any:
        """Import the factory module and call the factory function."""
        return self.get_factory()()


class TuningConfig(BaseModel):
    """Runtime tuning knobs."""

    max_concurrent_steps: int = Field(default=3, ge=0)
    max_retries: int = Field(default=3, ge=0)
    max_replan_attempts: int = Field(default=2, ge=0)


# ---------------------------------------------------------------------------
# Config file resolution and TOML settings source
# ---------------------------------------------------------------------------


def get_config_file_path() -> Path | None:
    """Resolve config file path with priority:

    1. {cwd}/.cortex/config.toml
    2. {project_root}/config.toml (same level as app/ directory)
    """
    project_config = Path.cwd() / ".cortex" / "config.toml"
    if project_config.is_file():
        return project_config

    root_config = Path(__file__).parent.parent / "config.toml"
    if root_config.is_file():
        return root_config

    return None


class TomlFileSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads from a TOML file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self.toml_data = self._load_toml()

    def _load_toml(self) -> dict[str, Any]:
        file = get_config_file_path()
        if file is None:
            return {}
        try:
            with file.open("rb") as f:
                return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(f"Invalid TOML in {file}: {e}") from e

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self.toml_data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self.toml_data


# ---------------------------------------------------------------------------
# Top-level CortexConfig
# ---------------------------------------------------------------------------


class CortexConfig(BaseSettings):
    """Main Cortex configuration.

    Priority (highest to lowest):
    1. Constructor arguments
    2. Environment variables (CORTEX_* prefix)
    3. TOML config file
    """

    model: ModelConfig
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    executors: list[ExecutorEntry] = Field(default_factory=list)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    model_config = SettingsConfigDict(
        env_prefix="CORTEX_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _validate_unique_intents(self) -> CortexConfig:
        seen: set[str] = set()
        for entry in self.executors:
            if entry.intent in seen:
                raise ValueError(
                    f"Duplicate executor intent: '{entry.intent}'. Intents must be unique."
                )
            seen.add(entry.intent)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlFileSettingsSource(settings_cls),
            file_secret_settings,
        )
