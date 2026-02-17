# Cortex

基於 Google ADK 的多步驟 AI Agent 框架。自動將複雜任務拆解為 DAG，並行執行、驗證、重新規劃。

## Quick Start

**環境需求:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# 安裝
uv sync

# 執行測試
uv run pytest tests/ -v

# 執行範例
uv run python example.py
```

### 設定

複製 `config.toml.example` 為 `config.toml`，填入 LLM 設定：

```toml
[model]
name = "openai/your-model-name"
api_base = "http://localhost:8000/v1"
api_key_env_var = "YOUR_API_KEY"
```

設定檔搜尋順序：`{cwd}/.cortex/config.toml` → `{project_root}/config.toml`

也支援環境變數 (`CORTEX_` prefix) 和建構式參數，優先級：建構式 > 環境變數 > TOML。

### 基本使用

```python
import asyncio
from app.config import CortexConfig
from cortex import Cortex

async def main():
    config = CortexConfig()  # 從 config.toml 讀取
    cortex = Cortex(config)
    result = await cortex.execute("寫一個 Hello World 程式")
    print(result)

asyncio.run(main())
```

### Sandbox 模式

啟用 Docker 沙盒讓 Agent 在隔離環境中操作檔案和執行指令（需安裝 Docker）：

```toml
[sandbox]
enable_filesystem = true
enable_shell = true
```

### 外部 Executor

透過 intent routing 將步驟分派到不同 Executor：

```toml
[[executors]]
intent = "generate"
description = "Generate new code"
factory_module = "app.agents.coding_agent.agent.mistral_vibe._agent"
factory_function = "create_coding_agent"
```

Planner 會自動根據 `available_intents` 為每個步驟選擇最適合的 executor。未匹配的步驟由內部 ExecutorAgent 執行 (`intent = "default"`)。

---

## Contributing

### 開發環境

```bash
uv sync
uv run pre-commit install
```

### 程式碼規範

本專案遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：

- `from __future__ import annotations` 在每個檔案開頭
- Type hints 使用 `X | None` 而非 `Optional[X]`
- 函式簽名標註回傳型別（包含 `-> None`）
- Import 順序：標準庫 → 第三方 → 本地（由 ruff isort 自動排序）

### 品質檢查

每次 commit 會自動執行以下檢查（pre-commit hook）：

| 工具 | 用途 |
|------|------|
| `ruff check --fix` | Linting + 自動修復 |
| `ruff format` | 程式碼格式化 |
| `mypy` | 靜態型別檢查 |

手動執行：

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy .
uv run pytest tests/ -v
```

**所有 PR 必須通過 ruff、mypy 零錯誤、全部測試通過。**

### 測試

使用 pytest + pytest-asyncio，遵循 TDD（Red-Green-Refactor）流程。目前 255 個測試。

```bash
uv run pytest tests/ -v          # 全部
uv run pytest tests/task/ -v     # 特定模組
```

---

## System Architecture

```
使用者輸入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                       Cortex                          │
│                                                       │
│  1. PlannerAgent 分析任務 → 產生 Plan (DAG)            │
│  2. 並行執行引擎 (Semaphore) 按依賴關係執行步驟         │
│  3. Intent Routing → 內部/外部 Executor                │
│  4. Verifier 驗證每步輸出                              │
│  5. 失敗 → ReplannerAgent 重新規劃                     │
│  6. Aggregator 彙整最終結果                            │
└──────────────────────────────────────────────────────┘
```

### 核心元件

| 元件 | 位置 | 職責 |
|------|------|------|
| **Cortex** | `cortex.py` | 主控制器：協調規劃、執行、驗證、重試 |
| **PlannerAgent** | `app/agents/planner/` | 將任務拆解為步驟 + 依賴關係 + intent 分配 |
| **ExecutorAgent** | `app/agents/executor/` | 預設執行者（純執行，不管理 plan 狀態） |
| **Verifier** | `app/agents/verifier/` | 雙階段驗證：機械檢查 + LLM 評估 |
| **ReplannerAgent** | `app/agents/replanner/` | 失敗步驟及下游依賴的重新規劃 |
| **Plan** | `app/task/plan.py` | DAG 資料結構（步驟、狀態、依賴、工具歷史） |
| **TaskManager** | `app/task/task_manager.py` | 全域 Plan 管理器（thread-safe） |
| **SandboxManager** | `app/sandbox/` | Docker 容器 + MCP 工具生命週期管理 |
| **CortexConfig** | `app/config.py` | Pydantic Settings（TOML + 環境變數 + 建構式） |

### 執行流程

```
PlannerAgent
    │  產生 Plan (steps + dependencies + intents)
    ▼
Plan (DAG)
    │  get_ready_steps() → 找出可並行的步驟
    ▼
Cortex 並行引擎 (Semaphore=3)
    │
    ├── intent == "default" → ExecutorAgent
    ├── intent == "generate" → 外部 Executor
    └── intent == "review" → 外部 Executor
    │
    ▼
Verifier
    │  verify_step() + evaluate_output()
    │
    ├── 通過 → 標記 completed，繼續下一步
    └── 失敗 → ReplannerAgent → redesign / give_up
    │
    ▼
Aggregator → 彙整最終結果
```

### 目錄結構

```
cortex/
├── cortex.py                   # 主控制器
├── api.py                      # FastAPI 微服務
├── config.toml.example         # 設定檔範例
├── example.py                  # 使用範例
├── app/
│   ├── config.py               # Pydantic Settings 設定模型
│   ├── agents/
│   │   ├── base/base_agent.py  # BaseAgent（包裝 ADK Agent）
│   │   ├── planner/            # PlannerAgent + prompts
│   │   ├── executor/           # ExecutorAgent + prompts
│   │   ├── verifier/           # Verifier（機械 + LLM 驗證）
│   │   └── replanner/          # ReplannerAgent + prompts
│   ├── sandbox/                # Docker 沙盒 + MCP 工具
│   ├── task/                   # Plan + TaskManager
│   └── tools/                  # PlanToolkit (create_plan, update_plan)
├── tests/                      # 255 tests
└── frontend/                   # React + Vite 前端（選用）
```

---

## API

### FastAPI 微服務

```bash
uv run uvicorn api:app --reload --port 8999
```

| Method | Endpoint | 說明 |
|--------|----------|------|
| `POST` | `/api/tasks` | 建立任務 `{"query": "..."}` → `{"task_id": "...", "status": "accepted"}` |
| `GET` | `/api/tasks/{task_id}/events` | SSE 事件串流（即時規劃進度 + 執行結果） |

### SSE 事件類型

| 事件 | 說明 |
|------|------|
| `connected` | 連線成功 |
| `plan_created` | 規劃完成，包含步驟清單 |
| `step_started` / `step_completed` | 步驟開始 / 完成 |
| `execution_complete` | 全部完成，包含最終結果 |
| `error` | 執行錯誤 |

### Frontend

```bash
cd frontend && npm install && npm run dev
# 開啟 http://localhost:5173
```

即時視覺化規劃進度、Agent 思考過程、工具呼叫細節，支援 Markdown 渲染與斷線重連。

---

## License

MIT
