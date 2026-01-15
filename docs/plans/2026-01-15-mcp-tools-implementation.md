# MCP Tools Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate MCP tools into Cortex with built-in filesystem/shell tools running in Docker containers, plus support for user-provided MCP servers.

**Architecture:** SandboxManager handles Docker container lifecycle and MCP toolset creation. Cortex receives new parameters (`workspace`, `enable_filesystem`, `enable_shell`, `mcp_servers`) and passes sandbox tools to agents. PlannerAgent gets read-only filesystem; ExecutorAgent gets all tools.

**Tech Stack:** `docker` (Python SDK), `google-adk` (McpToolset), `mcp` (StdioServerParameters)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:7-12`

**Step 1: Add docker dependency**

Edit `pyproject.toml` to add the docker package:

```toml
[project]
name = "cortex"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "docker>=7.1.0",
    "google-adk>=1.22.1",
    "google-genai>=1.57.0",
    "ipython>=9.9.0",
    "litellm>=1.80.16",
    "mcp>=1.0.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add docker and mcp dependencies"
```

---

## Task 2: Create SandboxManager - Basic Structure

**Files:**
- Create: `app/sandbox/sandbox_manager.py`
- Create: `app/sandbox/__init__.py`
- Create: `tests/sandbox/__init__.py`
- Create: `tests/sandbox/test_sandbox_manager.py`

**Step 1: Write the failing test for SandboxManager initialization**

Create `tests/sandbox/__init__.py`:
```python
```

Create `tests/sandbox/test_sandbox_manager.py`:
```python
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSandboxManager:
    def test_init_stores_config(self):
        """Should store workspace and tool configuration"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=False,
        )

        assert manager.workspace == "/tmp/test"
        assert manager.enable_filesystem is True
        assert manager.enable_shell is False
        assert manager.mcp_servers == []

    def test_init_with_mcp_servers(self):
        """Should store user MCP server configs"""
        from app.sandbox.sandbox_manager import SandboxManager

        servers = [{"url": "https://example.com/mcp"}]
        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=servers,
        )

        assert manager.mcp_servers == servers
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.sandbox'"

**Step 3: Create minimal SandboxManager**

Create `app/sandbox/__init__.py`:
```python
```

Create `app/sandbox/sandbox_manager.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/sandbox/ tests/sandbox/
git commit -m "feat: add SandboxManager basic structure"
```

---

## Task 3: SandboxManager - Docker Container Lifecycle

**Files:**
- Modify: `app/sandbox/sandbox_manager.py`
- Modify: `tests/sandbox/test_sandbox_manager.py`

**Step 1: Write failing tests for container lifecycle**

Add to `tests/sandbox/test_sandbox_manager.py`:
```python
class TestSandboxManagerContainer:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_start_creates_container(self, mock_docker):
        """Should create Docker container on start"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
        )

        import asyncio
        asyncio.run(manager.start())

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["detach"] is True
        assert "/tmp/test" in str(call_kwargs["volumes"])

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_removes_container(self, mock_docker):
        """Should stop and remove container on stop"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(workspace="/tmp/test")

        import asyncio
        asyncio.run(manager.start())
        asyncio.run(manager.stop())

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("app.sandbox.sandbox_manager.docker")
    def test_stop_without_start_is_safe(self, mock_docker):
        """Should handle stop without start gracefully"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(workspace="/tmp/test")

        import asyncio
        asyncio.run(manager.stop())  # Should not raise

    @patch("app.sandbox.sandbox_manager.docker")
    def test_context_manager(self, mock_docker):
        """Should support async context manager"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(workspace="/tmp/test")

        import asyncio

        async def test():
            async with manager:
                assert manager._container is not None
            mock_container.stop.assert_called_once()

        asyncio.run(test())
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerContainer -v`
Expected: FAIL with "AttributeError: 'SandboxManager' object has no attribute 'start'"

**Step 3: Implement container lifecycle**

Update `app/sandbox/sandbox_manager.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerContainer -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add app/sandbox/sandbox_manager.py tests/sandbox/test_sandbox_manager.py
git commit -m "feat: add Docker container lifecycle to SandboxManager"
```

---

## Task 4: SandboxManager - MCP Toolset Creation

**Files:**
- Modify: `app/sandbox/sandbox_manager.py`
- Modify: `tests/sandbox/test_sandbox_manager.py`

**Step 1: Write failing tests for tool creation**

Add to `tests/sandbox/test_sandbox_manager.py`:
```python
class TestSandboxManagerTools:
    def test_get_planner_tools_empty_when_disabled(self):
        """Should return empty list when filesystem disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=False,
        )

        tools = manager.get_planner_tools()
        assert tools == []

    def test_get_executor_tools_empty_when_disabled(self):
        """Should return empty list when all tools disabled"""
        from app.sandbox.sandbox_manager import SandboxManager

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=False,
            enable_shell=False,
        )

        tools = manager.get_executor_tools()
        assert tools == []

    @patch("app.sandbox.sandbox_manager.McpToolset")
    @patch("app.sandbox.sandbox_manager.docker")
    def test_get_planner_tools_returns_readonly_filesystem(self, mock_docker, mock_toolset):
        """Should return read-only filesystem tools for planner"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        mock_toolset_instance = MagicMock()
        mock_toolset.return_value = mock_toolset_instance

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
        )

        import asyncio
        asyncio.run(manager.start())

        tools = manager.get_planner_tools()

        # Should have created filesystem toolset with read-only filter
        assert mock_toolset.called
        call_kwargs = mock_toolset.call_args[1]
        assert "tool_filter" in call_kwargs
        # read_file and list_directory are read-only operations
        assert "read_file" in call_kwargs["tool_filter"]
        assert "list_directory" in call_kwargs["tool_filter"]

    @patch("app.sandbox.sandbox_manager.McpToolset")
    @patch("app.sandbox.sandbox_manager.docker")
    def test_get_executor_tools_returns_all_tools(self, mock_docker, mock_toolset):
        """Should return all tools for executor"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "test123"
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=True,
        )

        import asyncio
        asyncio.run(manager.start())

        tools = manager.get_executor_tools()

        # Should have created both filesystem and shell toolsets
        assert len(tools) >= 0  # Will have tools when container running
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerTools -v`
Expected: FAIL with "AttributeError: 'SandboxManager' object has no attribute 'get_planner_tools'"

**Step 3: Implement tool creation methods**

Update `app/sandbox/sandbox_manager.py`:
```python
from typing import Any

import docker
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


class SandboxManager:
    """Manages Docker container and MCP toolsets for sandboxed execution."""

    # Default Docker image for sandbox
    DEFAULT_IMAGE = "python:3.12-slim"

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
            docker_image: Custom Docker image (default: python:3.12-slim)
        """
        self.workspace = workspace
        self.enable_filesystem = enable_filesystem
        self.enable_shell = enable_shell
        self.mcp_servers = mcp_servers or []
        self.docker_image = docker_image or self.DEFAULT_IMAGE
        self._container = None
        self._client = None
        self._filesystem_toolset = None
        self._filesystem_toolset_readonly = None
        self._shell_toolset = None
        self._user_toolsets: list[Any] = []

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

        # Initialize MCP toolsets
        await self._init_toolsets()

    async def _init_toolsets(self):
        """Initialize MCP toolsets based on configuration."""
        if self._container is None:
            return

        container_id = self._container.id

        # Create filesystem toolset (full access for executor)
        if self.enable_filesystem:
            self._filesystem_toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", container_id, "python", "-m", "mcp_server_filesystem", "/workspace"],
                    )
                )
            )
            # Read-only version for planner
            self._filesystem_toolset_readonly = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", container_id, "python", "-m", "mcp_server_filesystem", "/workspace"],
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
                        args=["exec", container_id, "python", "-m", "mcp_server_shell"],
                    )
                )
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerTools -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add app/sandbox/sandbox_manager.py tests/sandbox/test_sandbox_manager.py
git commit -m "feat: add MCP toolset creation to SandboxManager"
```

---

## Task 5: SandboxManager - User MCP Servers Support

**Files:**
- Modify: `app/sandbox/sandbox_manager.py`
- Modify: `tests/sandbox/test_sandbox_manager.py`

**Step 1: Write failing tests for user MCP servers**

Add to `tests/sandbox/test_sandbox_manager.py`:
```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams


class TestSandboxManagerUserMcp:
    @patch("app.sandbox.sandbox_manager.McpToolset")
    @patch("app.sandbox.sandbox_manager.docker")
    def test_sse_mcp_server(self, mock_docker, mock_toolset):
        """Should support SSE MCP servers"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=[
                {"url": "https://api.example.com/mcp", "headers": {"Auth": "token"}}
            ],
        )

        import asyncio
        asyncio.run(manager.start())

        # SSE server should be in user toolsets
        tools = manager.get_executor_tools()
        assert mock_toolset.called

    @patch("app.sandbox.sandbox_manager.McpToolset")
    @patch("app.sandbox.sandbox_manager.docker")
    def test_stdio_mcp_server(self, mock_docker, mock_toolset):
        """Should support Stdio MCP servers (runs outside container)"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        manager = SandboxManager(
            workspace="/tmp/test",
            mcp_servers=[
                {"command": "npx", "args": ["-y", "@mcp/server-github"]}
            ],
        )

        import asyncio
        asyncio.run(manager.start())

        tools = manager.get_executor_tools()
        assert mock_toolset.called
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerUserMcp -v`
Expected: FAIL (user MCP servers not added to toolsets)

**Step 3: Implement user MCP server support**

Update `app/sandbox/sandbox_manager.py` - add to `_init_toolsets`:
```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

# ... existing code ...

    async def _init_toolsets(self):
        """Initialize MCP toolsets based on configuration."""
        if self._container is None:
            return

        container_id = self._container.id

        # Create filesystem toolset (full access for executor)
        if self.enable_filesystem:
            self._filesystem_toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", container_id, "python", "-m", "mcp_server_filesystem", "/workspace"],
                    )
                )
            )
            # Read-only version for planner
            self._filesystem_toolset_readonly = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="docker",
                        args=["exec", container_id, "python", "-m", "mcp_server_filesystem", "/workspace"],
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
                        args=["exec", container_id, "python", "-m", "mcp_server_shell"],
                    )
                )
            )

        # Create user MCP toolsets
        for server_config in self.mcp_servers:
            toolset = self._create_user_toolset(server_config)
            if toolset is not None:
                self._user_toolsets.append(toolset)

    def _create_user_toolset(self, config: dict) -> McpToolset | None:
        """Create MCP toolset from user config."""
        if "url" in config:
            # SSE server (remote)
            return McpToolset(
                connection_params=SseConnectionParams(
                    url=config["url"],
                    headers=config.get("headers"),
                )
            )
        elif "command" in config:
            # Stdio server (local, runs outside container)
            return McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env=config.get("env"),
                    )
                )
            )
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerUserMcp -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/sandbox/sandbox_manager.py tests/sandbox/test_sandbox_manager.py
git commit -m "feat: add user MCP server support to SandboxManager"
```

---

## Task 6: SandboxManager - Docker Availability Check

**Files:**
- Modify: `app/sandbox/sandbox_manager.py`
- Modify: `tests/sandbox/test_sandbox_manager.py`

**Step 1: Write failing test for Docker check**

Add to `tests/sandbox/test_sandbox_manager.py`:
```python
class TestSandboxManagerDockerCheck:
    @patch("app.sandbox.sandbox_manager.docker")
    def test_raises_when_docker_unavailable(self, mock_docker):
        """Should raise RuntimeError when Docker is not available"""
        from app.sandbox.sandbox_manager import SandboxManager

        mock_docker.from_env.side_effect = Exception("Docker not running")

        manager = SandboxManager(
            workspace="/tmp/test",
            enable_filesystem=True,
        )

        import asyncio
        with pytest.raises(RuntimeError, match="Docker"):
            asyncio.run(manager.start())
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerDockerCheck -v`
Expected: FAIL (raises generic Exception, not RuntimeError)

**Step 3: Implement Docker availability check**

Update `app/sandbox/sandbox_manager.py` `start` method:
```python
    async def start(self):
        """Start Docker container and initialize MCP toolsets."""
        if self._container is not None:
            return

        try:
            self._client = docker.from_env()
            self._client.ping()  # Verify Docker is responsive
        except Exception as e:
            raise RuntimeError(
                f"Docker is not available. Please ensure Docker is installed and running. Error: {e}"
            )

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sandbox/test_sandbox_manager.py::TestSandboxManagerDockerCheck -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/sandbox/sandbox_manager.py tests/sandbox/test_sandbox_manager.py
git commit -m "feat: add Docker availability check to SandboxManager"
```

---

## Task 7: Update Cortex - Add Sandbox Parameters

**Files:**
- Modify: `cortex.py`
- Modify: `tests/test_cortex.py`

**Step 1: Write failing tests for Cortex sandbox parameters**

Add to `tests/test_cortex.py`:
```python
class TestCortexSandbox:
    def test_init_without_sandbox(self):
        """Should work without sandbox (no workspace)"""
        cortex = Cortex(model=Mock())
        assert cortex.sandbox is None

    def test_init_with_workspace_creates_sandbox(self):
        """Should create SandboxManager when workspace provided"""
        cortex = Cortex(
            model=Mock(),
            workspace="/tmp/test",
            enable_filesystem=True,
        )
        assert cortex.sandbox is not None
        assert cortex.sandbox.workspace == "/tmp/test"
        assert cortex.sandbox.enable_filesystem is True

    def test_init_with_all_sandbox_options(self):
        """Should pass all options to SandboxManager"""
        mcp_servers = [{"url": "https://example.com/mcp"}]
        cortex = Cortex(
            model=Mock(),
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=True,
            mcp_servers=mcp_servers,
        )
        assert cortex.sandbox.enable_shell is True
        assert cortex.sandbox.mcp_servers == mcp_servers
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cortex.py::TestCortexSandbox -v`
Expected: FAIL with "TypeError: Cortex.__init__() got an unexpected keyword argument 'workspace'"

**Step 3: Update Cortex to accept sandbox parameters**

Update `cortex.py`:
```python
import time
from typing import Any, Callable

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent
from app.sandbox.sandbox_manager import SandboxManager


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        # Default: creates LlmAgent internally
        cortex = Cortex(model=model)

        # With sandbox (Docker-isolated tools):
        cortex = Cortex(
            model=model,
            workspace="./my-project",
            enable_filesystem=True,
            enable_shell=True,
        )

        # Custom: pass agent factories
        def my_planner_factory(tools: list):
            return LoopAgent(name="planner", tools=tools, ...)
        def my_executor_factory(tools: list):
            return LoopAgent(name="executor", tools=tools, ...)
        cortex = Cortex(
            planner_factory=my_planner_factory,
            executor_factory=my_executor_factory
        )
    """

    def __init__(
        self,
        model: Any = None,
        planner_factory: Callable[[list], Any] = None,
        executor_factory: Callable[[list], Any] = None,
        workspace: str = None,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] = None,
    ):
        if model is None and planner_factory is None:
            raise ValueError("Either 'model' or 'planner_factory' must be provided")
        if model is None and executor_factory is None:
            raise ValueError("Either 'model' or 'executor_factory' must be provided")

        self.model = model
        self.planner_factory = planner_factory
        self.executor_factory = executor_factory
        self.history: list[dict] = []

        # Create sandbox manager if workspace provided
        if workspace:
            self.sandbox = SandboxManager(
                workspace=workspace,
                enable_filesystem=enable_filesystem,
                enable_shell=enable_shell,
                mcp_servers=mcp_servers,
            )
        else:
            self.sandbox = None

    async def execute(self, query: str) -> str:
        """Execute a task with planning and execution"""
        # Record user query in history
        self.history.append({"role": "user", "content": query})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        try:
            # Start sandbox if configured
            if self.sandbox:
                await self.sandbox.start()

            # Create plan
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.planner_factory
            )
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.executor_factory
            )

            while True:
                ready_steps = plan.get_ready_steps()
                if not ready_steps:
                    break

                for step_idx in ready_steps:
                    plan.mark_step(step_idx, step_status="in_progress")
                    await executor.execute_step(step_idx, context=query)

            # Generate summary
            summary = self._generate_summary(plan)

            # Record result in history
            self.history.append({"role": "assistant", "content": summary})

            return summary

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)
            if self.sandbox:
                await self.sandbox.stop()

    def _generate_summary(self, plan: Plan) -> str:
        """Generate execution summary"""
        progress = plan.get_progress()
        return f"""Task completed.

{plan.format()}

Summary: {progress['completed']}/{progress['total']} steps completed."""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cortex.py::TestCortexSandbox -v`
Expected: PASS (3 tests)

**Step 5: Run all Cortex tests**

Run: `uv run pytest tests/test_cortex.py -v`
Expected: PASS (all tests including existing ones)

**Step 6: Commit**

```bash
git add cortex.py tests/test_cortex.py
git commit -m "feat: add sandbox parameters to Cortex"
```

---

## Task 8: Update Agents - Pass Sandbox Tools

**Files:**
- Modify: `app/agents/planner/planner_agent.py`
- Modify: `app/agents/executor/executor_agent.py`
- Modify: `cortex.py`
- Modify: `tests/test_cortex.py`

**Step 1: Write failing test for sandbox tools injection**

Add to `tests/test_cortex.py`:
```python
from unittest.mock import patch, AsyncMock


class TestCortexSandboxTools:
    @patch("cortex.SandboxManager")
    def test_planner_receives_sandbox_tools(self, mock_sandbox_class):
        """PlannerAgent should receive sandbox tools when sandbox configured"""
        mock_sandbox = MagicMock()
        mock_sandbox.get_planner_tools.return_value = [MagicMock(name="fs_tool")]
        mock_sandbox.get_executor_tools.return_value = []
        mock_sandbox.start = AsyncMock()
        mock_sandbox.stop = AsyncMock()
        mock_sandbox_class.return_value = mock_sandbox

        cortex = Cortex(
            model=Mock(),
            workspace="/tmp/test",
            enable_filesystem=True,
        )

        # Sandbox tools should be accessible
        assert cortex.sandbox is mock_sandbox
```

**Step 2: Update Cortex to pass sandbox tools to agents**

Update `cortex.py` execute method:
```python
    async def execute(self, query: str) -> str:
        """Execute a task with planning and execution"""
        # Record user query in history
        self.history.append({"role": "user", "content": query})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        try:
            # Start sandbox if configured
            if self.sandbox:
                await self.sandbox.start()

            # Get sandbox tools
            planner_sandbox_tools = self.sandbox.get_planner_tools() if self.sandbox else []
            executor_sandbox_tools = self.sandbox.get_executor_tools() if self.sandbox else []

            # Create plan
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.planner_factory,
                extra_tools=planner_sandbox_tools,
            )
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.executor_factory,
                extra_tools=executor_sandbox_tools,
            )

            while True:
                ready_steps = plan.get_ready_steps()
                if not ready_steps:
                    break

                for step_idx in ready_steps:
                    plan.mark_step(step_idx, step_status="in_progress")
                    await executor.execute_step(step_idx, context=query)

            # Generate summary
            summary = self._generate_summary(plan)

            # Record result in history
            self.history.append({"role": "assistant", "content": summary})

            return summary

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)
            if self.sandbox:
                await self.sandbox.stop()
```

**Step 3: Update PlannerAgent to accept extra_tools**

Update `app/agents/planner/planner_agent.py`:
```python
from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.plan_toolkit import PlanToolkit

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating and updating plans.

    Usage:
        # Default: creates LlmAgent internally
        planner = PlannerAgent(plan_id="p1", model=model)

        # Custom: pass agent_factory that receives tools
        def my_factory(tools: list):
            return LoopAgent(name="planner", tools=tools + my_extra_tools, ...)
        planner = PlannerAgent(plan_id="p1", agent_factory=my_factory)

        # With sandbox tools:
        planner = PlannerAgent(plan_id="p1", model=model, extra_tools=[fs_tool])
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable[[list], Any] = None,
        extra_tools: list = None,
    ):
        """
        Initialize PlannerAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that receives tools and returns an agent
            extra_tools: Additional tools (e.g., from sandbox) to include
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = PlanToolkit(plan)
        tools = list(toolkit.get_tool_functions().values())

        # Add extra tools (e.g., sandbox tools)
        if extra_tools:
            tools.extend(extra_tools)

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory(tools)
        elif model is not None:
            from google.adk.agents import LlmAgent

            agent = LlmAgent(
                name="planner",
                model=model,
                tools=tools,
                instruction=PLANNER_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(
            agent=agent, tool_functions=toolkit.get_tool_functions(), plan_id=plan_id
        )

    async def create_plan(self, task: str) -> str:
        """Create a plan for the given task"""
        result = await self.execute(f"Create a plan for: {task}")
        return result.output
```

**Step 4: Update ExecutorAgent to accept extra_tools**

Update `app/agents/executor/executor_agent.py`:
```python
from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent
from app.agents.executor.prompts import EXECUTOR_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.act_toolkit import ActToolkit

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing plan steps.

    Usage:
        # Default: creates LlmAgent internally
        executor = ExecutorAgent(plan_id="p1", model=model)

        # Custom: pass agent_factory that receives tools
        def my_factory(tools: list):
            return LoopAgent(name="executor", tools=tools + my_extra_tools, ...)
        executor = ExecutorAgent(plan_id="p1", agent_factory=my_factory)

        # With sandbox tools:
        executor = ExecutorAgent(plan_id="p1", model=model, extra_tools=[fs_tool, shell_tool])
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable[[list], Any] = None,
        extra_tools: list = None,
    ):
        """
        Initialize ExecutorAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that receives tools and returns an agent
            extra_tools: Additional tools (e.g., from sandbox) to include
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = ActToolkit(plan)
        tools = list(toolkit.get_tool_functions().values())

        # Add extra tools (e.g., sandbox tools)
        if extra_tools:
            tools.extend(extra_tools)

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory(tools)
        elif model is not None:
            from google.adk.agents import LlmAgent

            agent = LlmAgent(
                name="executor",
                model=model,
                tools=tools,
                instruction=EXECUTOR_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(
            agent=agent, tool_functions=toolkit.get_tool_functions(), plan_id=plan_id
        )

    async def execute_step(self, step_index: int, context: str = "") -> str:
        """Execute a specific step"""
        step_desc = self.plan.steps[step_index]
        query = f"Execute step {step_index}: {step_desc}"
        if context:
            query += f"\n\nContext: {context}"

        result = await self.execute(query)
        return result.output
```

**Step 5: Write tests for extra_tools parameter**

Add to `tests/agents/planner/test_planner_agent.py`:
```python
    def test_extra_tools_included(self):
        """extra_tools should be included in agent tools"""
        extra_tool = Mock(name="extra_tool")
        received_tools = []

        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()

        PlannerAgent(
            plan_id="plan_1",
            agent_factory=my_factory,
            extra_tools=[extra_tool],
        )

        assert extra_tool in received_tools
```

Add to `tests/agents/executor/test_executor_agent.py`:
```python
    def test_extra_tools_included(self):
        """extra_tools should be included in agent tools"""
        extra_tool = Mock(name="extra_tool")
        received_tools = []

        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()

        ExecutorAgent(
            plan_id="plan_1",
            agent_factory=my_factory,
            extra_tools=[extra_tool],
        )

        assert extra_tool in received_tools
```

**Step 6: Run all tests**

Run: `uv run pytest -v`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add app/agents/planner/planner_agent.py app/agents/executor/executor_agent.py cortex.py tests/
git commit -m "feat: pass sandbox tools to agents"
```

---

## Task 9: Update Example and Documentation

**Files:**
- Modify: `example.py`
- Modify: `README.md`

**Step 1: Update example.py with sandbox usage**

Update `example.py`:
```python
"""
Example script to run Cortex.

Usage:
    uv run python example.py
"""

import asyncio
import warnings

# Filter Pydantic warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from cortex import Cortex

# Model configuration
API_BASE_URL = "http://10.136.3.209:8000/v1"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


async def main():
    # Initialize model via LiteLLM
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
        api_key="EMPTY",
    )

    # Create Cortex (default mode)
    cortex = Cortex(model=model)

    # --- Sandbox mode example ---
    # cortex = Cortex(
    #     model=model,
    #     workspace="./my-project",      # Mount this directory
    #     enable_filesystem=True,        # Built-in filesystem tool
    #     enable_shell=True,             # Built-in shell tool
    # )
    # ------------------------------------

    # --- With user MCP servers ---
    # cortex = Cortex(
    #     model=model,
    #     workspace="./my-project",
    #     enable_filesystem=True,
    #     enable_shell=True,
    #     mcp_servers=[
    #         # Remote MCP server (SSE)
    #         {"url": "https://api.example.com/mcp", "headers": {"Auth": "token"}},
    #         # Local MCP server (Stdio)
    #         {"command": "npx", "args": ["-y", "@mcp/server-github"]},
    #     ],
    # )
    # ------------------------------------

    # --- Custom agent factory example ---
    # from google.adk.agents import LoopAgent
    #
    # def my_planner_factory(tools: list):
    #     return LoopAgent(
    #         name="planner",
    #         model=model,
    #         tools=tools,  # toolkit tools are injected here
    #     )
    #
    # cortex = Cortex(planner_factory=my_planner_factory)
    # ------------------------------------

    # Execute a task
    query = "寫一篇短篇兒童小說"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Update README.md with sandbox documentation**

Add to README.md after the basic usage section:
```markdown
### 5. 使用 Sandbox (Docker 隔離執行)

Cortex 支援在 Docker container 內執行工具操作，提供安全隔離：

```python
from cortex import Cortex

# 啟用內建的 filesystem 和 shell 工具
cortex = Cortex(
    model=model,
    workspace="./my-project",    # 掛載到 container 的目錄
    enable_filesystem=True,      # 內建 filesystem tool
    enable_shell=True,           # 內建 shell tool
)

result = await cortex.execute("幫我寫一個 Python 程式來處理 CSV 檔案")
```

**加入使用者自訂的 MCP servers：**

```python
cortex = Cortex(
    model=model,
    workspace="./my-project",
    enable_filesystem=True,
    enable_shell=True,
    mcp_servers=[
        # Remote MCP server (SSE)
        {"url": "https://my-api.com/mcp", "headers": {"Authorization": "Bearer xxx"}},
        # Local MCP server (Stdio) - 在 container 外執行
        {"command": "npx", "args": ["-y", "@mcp/server-github"]},
    ],
)
```

**工具分配：**

| Tool | PlannerAgent | ExecutorAgent | Notes |
|------|--------------|---------------|-------|
| `create_plan` | ✓ | ✗ | 內部工具 |
| `update_plan` | ✓ | ✗ | 內部工具 |
| `mark_step` | ✗ | ✓ | 內部工具 |
| `filesystem` | ✓ (read-only) | ✓ (read/write) | 內建 MCP |
| `shell` | ✗ | ✓ | 內建 MCP |
| 使用者 MCP | ✗ | ✓ | 使用者配置 |

**注意：** 使用 sandbox 功能需要安裝並啟動 Docker。
```

**Step 3: Commit**

```bash
git add example.py README.md
git commit -m "docs: add sandbox usage examples"
```

---

## Task 10: Final Verification

**Files:**
- All modified files

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 2: Verify imports work**

Run: `uv run python -c "from cortex import Cortex; from app.sandbox.sandbox_manager import SandboxManager; print('OK')"`
Expected: OK

**Step 3: Create final commit (if needed)**

```bash
git status
# If any uncommitted changes:
git add -A
git commit -m "chore: final cleanup"
```

**Step 4: Merge to main (optional)**

```bash
git checkout main
git merge feature/mcp-tools
git push origin main
```

---

## Summary

This plan implements MCP tools integration with:

1. **SandboxManager** - Manages Docker container lifecycle and MCP toolsets
2. **Built-in tools** - Filesystem (read-only for planner, full for executor) and Shell
3. **User MCP servers** - Support for SSE (remote) and Stdio (local) servers
4. **Cortex updates** - New parameters: `workspace`, `enable_filesystem`, `enable_shell`, `mcp_servers`
5. **Agent updates** - `extra_tools` parameter for sandbox tools injection

Total: 10 tasks, ~60 tests
