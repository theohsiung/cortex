"""Tests for app.config Pydantic models."""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import TypeAdapter, ValidationError

from app.config import (
    MCPStdio, MCPSse, MCPServer,
    ModelConfig, SandboxConfig,
    ExecutorEntry, TuningConfig,
)


# ---------------------------------------------------------------------------
# Task 2 – MCP Discriminated Union Models
# ---------------------------------------------------------------------------

class TestMCPStdio:
    """MCPStdio model tests."""

    def test_valid_config(self):
        cfg = MCPStdio(transport="stdio", command="/usr/bin/node", args=["server.js"])
        assert cfg.transport == "stdio"
        assert cfg.command == "/usr/bin/node"
        assert cfg.args == ["server.js"]

    def test_defaults(self):
        cfg = MCPStdio(transport="stdio", command="python")
        assert cfg.args == []
        assert cfg.env == {}

    def test_with_env(self):
        cfg = MCPStdio(
            transport="stdio",
            command="node",
            env={"NODE_ENV": "production"},
        )
        assert cfg.env == {"NODE_ENV": "production"}

    def test_rejects_wrong_transport(self):
        with pytest.raises(ValidationError):
            MCPStdio(transport="sse", command="python")


class TestMCPSse:
    """MCPSse model tests."""

    def test_valid_config(self):
        cfg = MCPSse(transport="sse", url="http://localhost:8080/sse")
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:8080/sse"

    def test_with_headers(self):
        cfg = MCPSse(
            transport="sse",
            url="http://localhost:8080/sse",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_rejects_wrong_transport(self):
        with pytest.raises(ValidationError):
            MCPSse(transport="stdio", url="http://localhost:8080/sse")


class TestMCPServerDiscriminator:
    """Discriminated union via MCPServer type alias."""

    ta = TypeAdapter(MCPServer)

    def test_routes_to_stdio(self):
        obj = self.ta.validate_python(
            {"transport": "stdio", "command": "node", "args": ["index.js"]}
        )
        assert isinstance(obj, MCPStdio)

    def test_routes_to_sse(self):
        obj = self.ta.validate_python(
            {"transport": "sse", "url": "http://localhost:8080/sse"}
        )
        assert isinstance(obj, MCPSse)

    def test_rejects_unknown_transport(self):
        with pytest.raises(ValidationError):
            self.ta.validate_python({"transport": "grpc", "command": "foo"})

    def test_list_of_mixed_servers(self):
        ta_list = TypeAdapter(list[MCPServer])
        servers = ta_list.validate_python([
            {"transport": "stdio", "command": "node"},
            {"transport": "sse", "url": "http://localhost:8080/sse"},
        ])
        assert isinstance(servers[0], MCPStdio)
        assert isinstance(servers[1], MCPSse)


# ---------------------------------------------------------------------------
# Task 3 – ModelConfig and SandboxConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    """ModelConfig model tests."""

    def test_valid_config(self):
        cfg = ModelConfig(name="gpt-4", api_base="https://api.openai.com/v1")
        assert cfg.name == "gpt-4"
        assert cfg.api_base == "https://api.openai.com/v1"

    def test_defaults(self):
        cfg = ModelConfig(name="gpt-4", api_base="https://api.openai.com/v1")
        assert cfg.api_key_env_var == ""
        assert cfg.temperature == 0.0

    def test_requires_name(self):
        with pytest.raises(ValidationError):
            ModelConfig(api_base="https://api.openai.com/v1")

    def test_requires_api_base(self):
        with pytest.raises(ValidationError):
            ModelConfig(name="gpt-4")

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret-123")
        cfg = ModelConfig(
            name="gpt-4",
            api_base="https://api.openai.com/v1",
            api_key_env_var="MY_API_KEY",
        )
        assert cfg.resolve_api_key() == "secret-123"

    def test_resolve_api_key_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = ModelConfig(
            name="gpt-4",
            api_base="https://api.openai.com/v1",
            api_key_env_var="MISSING_KEY",
        )
        assert cfg.resolve_api_key() is None

    def test_resolve_api_key_returns_none_when_no_env_var(self):
        cfg = ModelConfig(name="gpt-4", api_base="https://api.openai.com/v1")
        assert cfg.resolve_api_key() is None


class TestSandboxConfig:
    """SandboxConfig model tests."""

    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.enable_filesystem is False
        assert cfg.enable_shell is False
        assert cfg.docker_image == "cortex-sandbox:latest"
        assert cfg.user_id is None

    def test_all_fields_set(self):
        cfg = SandboxConfig(
            enable_filesystem=True,
            enable_shell=True,
            docker_image="custom:v2",
            user_id="user-42",
        )
        assert cfg.enable_filesystem is True
        assert cfg.enable_shell is True
        assert cfg.docker_image == "custom:v2"
        assert cfg.user_id == "user-42"


# ---------------------------------------------------------------------------
# Task 4 – ExecutorEntry and TuningConfig
# ---------------------------------------------------------------------------

class TestExecutorEntry:
    """ExecutorEntry model tests."""

    def test_valid_config(self):
        cfg = ExecutorEntry(
            intent="code",
            description="Coding executor",
            factory_module="app.agents.coding_agent",
        )
        assert cfg.intent == "code"
        assert cfg.description == "Coding executor"
        assert cfg.factory_module == "app.agents.coding_agent"

    def test_defaults(self):
        cfg = ExecutorEntry(
            intent="code",
            description="Coding executor",
            factory_module="app.agents.coding_agent",
        )
        assert cfg.factory_function == "create_agent"
        assert cfg.is_default is False

    def test_requires_intent(self):
        with pytest.raises(ValidationError):
            ExecutorEntry(
                description="Coding executor",
                factory_module="app.agents.coding_agent",
            )

    def test_requires_factory_module(self):
        with pytest.raises(ValidationError):
            ExecutorEntry(
                intent="code",
                description="Coding executor",
            )

    def test_is_default_flag(self):
        cfg = ExecutorEntry(
            intent="code",
            description="Coding executor",
            factory_module="app.agents.coding_agent",
            is_default=True,
        )
        assert cfg.is_default is True


class TestTuningConfig:
    """TuningConfig model tests."""

    def test_defaults(self):
        cfg = TuningConfig()
        assert cfg.max_concurrent_steps == 3
        assert cfg.max_retries == 3
        assert cfg.max_replan_attempts == 2

    def test_custom_values(self):
        cfg = TuningConfig(
            max_concurrent_steps=10,
            max_retries=5,
            max_replan_attempts=4,
        )
        assert cfg.max_concurrent_steps == 10
        assert cfg.max_retries == 5
        assert cfg.max_replan_attempts == 4

    def test_rejects_negative_max_concurrent_steps(self):
        with pytest.raises(ValidationError):
            TuningConfig(max_concurrent_steps=-1)

    def test_rejects_negative_max_retries(self):
        with pytest.raises(ValidationError):
            TuningConfig(max_retries=-1)

    def test_rejects_negative_max_replan_attempts(self):
        with pytest.raises(ValidationError):
            TuningConfig(max_replan_attempts=-1)


# ---------------------------------------------------------------------------
# Task 5 – ExecutorEntry.create_executor() and get_factory()
# ---------------------------------------------------------------------------

class TestExecutorEntryCreateExecutor:
    """Tests for dynamic import via create_executor / get_factory."""

    FAKE_MODULE = "_fake_executor_module_for_test"

    @pytest.fixture(autouse=True)
    def _install_fake_module(self):
        """Insert a throwaway module into sys.modules for the duration of each test."""
        mod = types.ModuleType(self.FAKE_MODULE)
        mod.create_agent = lambda: "fake-agent-instance"
        mod.custom_factory = lambda: "custom-instance"
        sys.modules[self.FAKE_MODULE] = mod
        yield
        sys.modules.pop(self.FAKE_MODULE, None)

    def test_create_executor_imports_and_calls_factory(self):
        entry = ExecutorEntry(
            intent="code",
            description="test",
            factory_module=self.FAKE_MODULE,
        )
        result = entry.create_executor()
        assert result == "fake-agent-instance"

    def test_create_executor_raises_import_error_on_missing_module(self):
        entry = ExecutorEntry(
            intent="code",
            description="test",
            factory_module="no_such_module_xyz_999",
        )
        with pytest.raises(ImportError):
            entry.create_executor()

    def test_create_executor_raises_attribute_error_on_missing_function(self):
        entry = ExecutorEntry(
            intent="code",
            description="test",
            factory_module=self.FAKE_MODULE,
            factory_function="nonexistent_fn",
        )
        with pytest.raises(AttributeError):
            entry.create_executor()

    def test_get_factory_returns_callable_without_calling_it(self):
        entry = ExecutorEntry(
            intent="code",
            description="test",
            factory_module=self.FAKE_MODULE,
            factory_function="custom_factory",
        )
        factory = entry.get_factory()
        assert callable(factory)
        # Has not been called yet; calling it now should produce the value.
        assert factory() == "custom-instance"
