# MCP Tools Integration Design

## Overview

整合 MCP (Model Context Protocol) 工具到 Cortex，提供內建的 filesystem 和 shell 工具，並支援使用者自訂的 MCP servers。所有工具操作都在 Docker container 內執行，確保安全隔離。

## Goals

1. 提供內建的 filesystem 和 shell MCP tools
2. 支援使用者額外配置的 MCP servers
3. 使用 Docker container 隔離所有工具操作
4. 對使用者隱藏複雜度，提供簡單的 API

## User Experience

### Basic Usage

```python
from cortex import Cortex

# 啟用內建工具
cortex = Cortex(
    model=model,
    workspace="./my-project",    # 掛載到 container 的目錄
    enable_filesystem=True,      # 內建 filesystem tool
    enable_shell=True,           # 內建 shell tool
)

result = await cortex.execute("幫我寫一個 Python 程式來處理 CSV 檔案")
```

### Advanced Usage

```python
# 加上使用者自訂的 MCP servers
cortex = Cortex(
    model=model,
    workspace="./my-project",
    enable_filesystem=True,
    enable_shell=True,
    mcp_servers=[
        # Remote MCP server (SSE)
        {"url": "https://my-api.com/mcp", "headers": {"Authorization": "Bearer xxx"}},
        # Local MCP server (Stdio) - 會在 container 外執行
        {"command": "npx", "args": ["-y", "@mcp/server-github"]},
    ],
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Cortex                                 │
│                                                                  │
│  ┌───────────────┐              ┌───────────────┐               │
│  │ PlannerAgent  │              │ ExecutorAgent │               │
│  │               │              │               │               │
│  │ Tools:        │              │ Tools:        │               │
│  │ - create_plan │              │ - mark_step   │               │
│  │ - update_plan │              │ - filesystem  │◄── 內建 MCP   │
│  │ - filesystem  │              │ - shell       │               │
│  │   (read-only) │              │ - 使用者 MCP  │               │
│  └───────────────┘              └───────┬───────┘               │
│                                         │                        │
│  ┌──────────────────────────────────────┴───────────────────┐   │
│  │                    SandboxManager                         │   │
│  │  - 管理 Docker container 生命週期                         │   │
│  │  - 建立內建 MCP toolsets (filesystem, shell)              │   │
│  │  - 管理 workspace 掛載                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Docker Container         │
              │                                │
              │  /workspace/ ◄── mount from    │
              │     └── (使用者專案目錄)       │
              │                                │
              │  MCP Servers:                  │
              │  - filesystem server           │
              │  - shell server                │
              └────────────────────────────────┘
```

## Tool Distribution

| Tool | PlannerAgent | ExecutorAgent | Notes |
|------|--------------|---------------|-------|
| `create_plan` | ✓ | ✗ | 內部工具 |
| `update_plan` | ✓ | ✗ | 內部工具 |
| `mark_step` | ✗ | ✓ | 內部工具 |
| `filesystem` | ✓ (read-only) | ✓ (read/write) | 內建 MCP |
| `shell` | ✗ | ✓ | 內建 MCP |
| 使用者 MCP | ✗ | ✓ | 使用者配置 |

### Rationale

- **PlannerAgent** 只需要 read-only filesystem 來了解專案結構，幫助制定更好的計畫
- **ExecutorAgent** 需要完整的 filesystem 和 shell 來執行實際工作
- 使用者自訂的 MCP tools 預設只給 ExecutorAgent，因為通常是執行階段才需要

## Components

### 1. SandboxManager

負責管理 Docker container 和 MCP toolsets。

```python
class SandboxManager:
    """Manages Docker container and MCP toolsets for sandboxed execution."""

    def __init__(
        self,
        workspace: str,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] = None,
    ):
        self.workspace = workspace
        self.enable_filesystem = enable_filesystem
        self.enable_shell = enable_shell
        self.mcp_servers = mcp_servers or []
        self._container = None
        self._toolsets = []

    async def start(self):
        """Start Docker container and initialize MCP toolsets."""
        pass

    async def stop(self):
        """Stop Docker container and cleanup."""
        pass

    def get_planner_tools(self) -> list:
        """Get tools for PlannerAgent (read-only filesystem)."""
        pass

    def get_executor_tools(self) -> list:
        """Get tools for ExecutorAgent (all tools)."""
        pass
```

### 2. Cortex Changes

新增參數支援 MCP tools。

```python
class Cortex:
    def __init__(
        self,
        model: Any = None,
        planner_factory: Callable[[list], Any] = None,
        executor_factory: Callable[[list], Any] = None,
        # New parameters
        workspace: str = None,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] = None,
    ):
        # ...
        self.sandbox = SandboxManager(
            workspace=workspace,
            enable_filesystem=enable_filesystem,
            enable_shell=enable_shell,
            mcp_servers=mcp_servers,
        ) if workspace else None
```

### 3. Docker Container

使用輕量級 Python image，預裝常用工具。

```dockerfile
FROM python:3.12-slim

# Install common tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MCP servers
RUN pip install mcp-server-filesystem mcp-server-shell

WORKDIR /workspace
```

## MCP Integration with ADK

使用 Google ADK 的 `McpToolset` 來整合 MCP tools。

```python
from google.adk.tools import McpToolset
from google.adk.tools.mcp import StdioConnectionParams, StdioServerParameters

# Filesystem MCP (in container)
filesystem_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="docker",
            args=["exec", container_id, "python", "-m", "mcp_server_filesystem", "/workspace"]
        )
    ),
    tool_filter=["read_file", "list_directory"]  # For planner (read-only)
)

# Shell MCP (in container)
shell_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="docker",
            args=["exec", container_id, "python", "-m", "mcp_server_shell"]
        )
    )
)
```

## Container Lifecycle

```
Cortex.__init__()
    │
    ▼
SandboxManager.start()
    ├── Start Docker container
    ├── Mount workspace directory
    └── Initialize MCP toolsets
    │
    ▼
cortex.execute(query)
    ├── PlannerAgent uses read-only filesystem
    └── ExecutorAgent uses all tools (in container)
    │
    ▼
SandboxManager.stop() (on cleanup or __del__)
    ├── Stop MCP connections
    └── Remove Docker container
```

## Security Considerations

1. **Filesystem isolation**: Agent 只能存取掛載的 workspace 目錄
2. **Network isolation**: Container 預設無網路存取（除非使用者 MCP 需要）
3. **Resource limits**: Container 設定 CPU 和 memory 限制
4. **No privilege escalation**: Container 以非 root 使用者執行

## Error Handling

1. **Docker not available**: 如果 Docker 未安裝，`enable_filesystem` 和 `enable_shell` 會拋出 `RuntimeError`
2. **Container crash**: 自動重啟 container，保留 workspace 狀態
3. **MCP connection failure**: Retry with exponential backoff

## Future Enhancements

1. **Container image customization**: 讓使用者指定自己的 Docker image
2. **Persistent containers**: 重複使用 container 而不是每次重建
3. **Network policies**: 更細緻的網路存取控制
4. **Resource monitoring**: 監控 container 資源使用

## Dependencies

- `docker` Python package (for container management)
- `google-adk` (for McpToolset)
- Docker Engine (system requirement)
