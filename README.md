# Cortex - AI Agent Framework

Cortex 是一個基於 Google ADK (Agent Development Kit) 的 AI Agent 框架。它能夠將複雜的任務自動分解成多個步驟，然後逐一執行完成。

## 這個專案在做什麼？

想像你有一個很聰明的助理，當你給他一個複雜的任務時，他會：

1. **規劃 (Planning)**: 先把任務拆解成多個小步驟，並建立依賴關係
2. **並行執行 (Parallel Execution)**: 根據 DAG 依賴關係，同時執行多個獨立步驟
3. **彙整 (Aggregation)**: 將各步驟的輸出整合成最終結果

這就是 Cortex 在做的事情！

```
使用者: "幫我整理房間"
    ↓
Cortex 規劃:
    步驟 0: 準備清潔工具
    步驟 1: 收拾桌面      (依賴步驟 0)
    步驟 2: 整理書櫃      (依賴步驟 0)
    步驟 3: 吸地板        (依賴步驟 1, 2)
    ↓
Cortex 並行執行:
    ✓ 步驟 0 完成
    ✓ 步驟 1, 2 同時執行 (並行)
    ✓ 步驟 3 完成
    ↓
Cortex 彙整: "房間整理完成！以下是執行摘要..."
```

---

## 專案架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                         Cortex                               │
│                    (主要控制中心)                              │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │  PlannerAgent   │         │  ExecutorAgent  │            │
│  │   (規劃者)       │         │    (執行者)      │            │
│  │                 │         │                 │            │
│  │ 負責把任務拆解   │         │ 負責執行每個步驟  │            │
│  │ 成多個步驟      │         │ 並回報狀態       │            │
│  └────────┬────────┘         └────────┬────────┘            │
│           │                           │                      │
│           │         ┌─────────────────┤                      │
│           │         │                 │                      │
│           ▼         ▼                 ▼                      │
│  ┌─────────────────────────────────────────────┐            │
│  │                   Plan                       │            │
│  │                 (執行計畫)                    │            │
│  │                                              │            │
│  │  - 步驟清單 (steps)                          │            │
│  │  - 步驟狀態 (not_started/in_progress/done)   │            │
│  │  - 步驟依賴關係 (dependencies)               │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 目錄結構

```
cortex/
├── cortex.py              # 主程式入口 - Cortex 類別
├── example.py             # 執行範例
├── app/
│   ├── agents/            # Agent 相關程式碼
│   │   ├── base/          # 基礎 Agent
│   │   │   └── base_agent.py    # BaseAgent - 所有 Agent 的父類別
│   │   ├── planner/       # 規劃 Agent
│   │   │   ├── planner_agent.py # PlannerAgent - 負責規劃
│   │   │   └── prompts.py       # 規劃用的提示詞
│   │   └── executor/      # 執行 Agent
│   │       ├── executor_agent.py # ExecutorAgent - 負責執行
│   │       └── prompts.py       # 執行用的提示詞
│   ├── sandbox/           # Docker 沙盒環境
│   │   ├── sandbox_manager.py   # SandboxManager - 管理 Docker 容器
│   │   ├── Dockerfile           # 沙盒 Docker 映像檔
│   │   └── mcp_servers/         # MCP 伺服器
│   │       ├── filesystem_server.py # 檔案系統工具
│   │       └── shell_server.py      # Shell 執行工具
│   ├── task/              # 任務管理
│   │   ├── plan.py        # Plan 類別 - 儲存執行計畫
│   │   └── task_manager.py # TaskManager - 全域計畫管理器
│   └── tools/             # LLM 可呼叫的工具
│       ├── plan_toolkit.py # 規劃工具 (create_plan, update_plan)
│       └── act_toolkit.py  # 執行工具 (mark_step)
├── templates/             # 提示詞模板
├── tests/                 # 測試程式碼
│   └── conftest.py        # 測試設定與 Mock
└── docs/                  # 文件
```

---

## 核心元件詳解

### 1. Cortex (`cortex.py`)

**這是什麼？** 整個系統的大腦，負責協調所有工作。

**它做什麼？**

- 接收使用者的任務
- 創建一個新的 Plan (計畫)
- 呼叫 PlannerAgent 來規劃步驟
- **並行執行**步驟 (根據 DAG 依賴關係，使用 `Semaphore(3)` 控制並行數)
- **自動重試**失敗的步驟 (最多 3 次嘗試)
- 使用 **Aggregator** 將各步驟輸出彙整成最終結果

```python
# 使用範例 - 預設 LlmAgent
from cortex import Cortex

cortex = Cortex(model=your_llm_model)
result = await cortex.execute("幫我寫一個計算機程式")
print(result)

# 使用 Sandbox 模式 - Docker 隔離執行
cortex = Cortex(
    model=your_llm_model,
    workspace="./my-project",    # 掛載到 Docker 的目錄
    enable_filesystem=True,      # 啟用檔案系統工具
    enable_shell=True,           # 啟用 Shell 執行工具
)

# 進階使用 - 自訂 Agent Factory
# 使用 factory 函數可以讓你的自訂 Agent 自動獲得 toolkit 工具
from google.adk.agents import LoopAgent

def my_planner_factory(tools: list):
    return LoopAgent(
        name="planner",
        model=your_llm_model,
        tools=tools,  # toolkit 工具會自動注入
    )

def my_executor_factory(tools: list):
    return LoopAgent(
        name="executor",
        model=your_llm_model,
        tools=tools,  # toolkit 工具會自動注入
    )

cortex = Cortex(
    planner_factory=my_planner_factory,
    executor_factory=my_executor_factory
)
```

---

### 2. BaseAgent (`app/agents/base/base_agent.py`)

**這是什麼？** 所有 Agent 的「父類別」，定義了 Agent 的基本行為。

**它做什麼？**

- 包裝 Google ADK 的任意 Agent 類型 (LlmAgent, LoopAgent, SequentialAgent, ParallelAgent)
- 管理與 LLM (大型語言模型) 的對話
- 追蹤工具呼叫事件
- 連接到 Plan (執行計畫)

**支援的 ADK Agent 類型：**
| Agent | 說明 |
|-------|------|
| LlmAgent | 基本的 LLM Agent (預設) |
| LoopAgent | 可重複執行的 Agent |
| SequentialAgent | 依序執行多個子 Agent |
| ParallelAgent | 同時執行多個子 Agent |

**重要概念 - 繼承：**

```
BaseAgent (父類別)
    ├── PlannerAgent (子類別) - 專門負責規劃
    └── ExecutorAgent (子類別) - 專門負責執行
```

---

### 3. PlannerAgent (`app/agents/planner/planner_agent.py`)

**這是什麼？** 負責「想」的 Agent。

**它做什麼？**

- 分析使用者的任務
- 把任務拆解成多個具體步驟
- 使用 `create_plan` 工具來建立計畫

**例子：**

```
輸入: "做一個網站"

PlannerAgent 輸出:
  步驟 1: 設計網站架構
  步驟 2: 撰寫 HTML 結構
  步驟 3: 加入 CSS 樣式
  步驟 4: 加入 JavaScript 互動
  步驟 5: 測試網站功能
```

---

### 4. ExecutorAgent (`app/agents/executor/executor_agent.py`)

**這是什麼？** 負責「做」的 Agent。

**它做什麼？**

- 接收一個步驟
- 執行該步驟
- 使用 `mark_step` 工具回報狀態 (進行中/完成/卡住)

**狀態說明：**

```
not_started  → 還沒開始
in_progress  → 正在進行中
completed    → 已完成 ✓
blocked      → 卡住了，無法繼續 ✗
```

---

### 5. Plan (`app/task/plan.py`)

**這是什麼？** 儲存執行計畫的資料結構。

**它包含什麼？**

- `title`: 計畫標題
- `steps`: 步驟清單 (例如: ["步驟 1", "步驟 2", "步驟 3"])
- `step_statuses`: 每個步驟的狀態
- `step_notes`: 每個步驟的備註
- `dependencies`: 步驟之間的依賴關係
- `step_tool_history`: 每個步驟呼叫的工具歷史記錄
- `step_files`: 每個步驟產生的檔案路徑

**依賴關係是什麼？**

```
假設有 3 個步驟:
  步驟 0: 買食材
  步驟 1: 切菜      (依賴步驟 0，要先買食材才能切)
  步驟 2: 炒菜      (依賴步驟 1，要先切菜才能炒)

dependencies = {1: [0], 2: [1]}
意思是: 步驟 1 依賴步驟 0，步驟 2 依賴步驟 1
```

**工具歷史記錄：**

Plan 會自動記錄每個步驟執行時呼叫的工具和產生的檔案：

```python
plan.format()
# 輸出:
# Plan: Build Data Pipeline
# ========================================
# Progress: 2/3 (66.7%)
#
# Steps:
#   0: [✓] Load CSV data
#       Notes: Loaded 1000 rows
#       Tools: read_file (1)
#   1: [✓] Process and save results
#       Notes: Saved processed data
#       Tools: run_python (1)
#       Files: /workspace/output.csv
#   2: [ ] Generate report (depends on: [1])
```

---

### 6. TaskManager (`app/task/task_manager.py`)

**這是什麼？** 全域的計畫管理器 (像是一個倉庫)。

**它做什麼？**

- 儲存多個 Plan
- 讓不同的 Agent 可以存取同一個 Plan
- 確保多執行緒安全 (Thread-safe)

**為什麼需要它？**

```
PlannerAgent 創建計畫 → 存到 TaskManager
                              ↓
ExecutorAgent 執行計畫 ← 從 TaskManager 取出
```

---

### 7. Toolkits (工具包)

#### PlanToolkit (`app/tools/plan_toolkit.py`)

提供給 **PlannerAgent** 使用的工具：

- `create_plan`: 創建新計畫
- `update_plan`: 更新現有計畫

#### ActToolkit (`app/tools/act_toolkit.py`)

提供給 **ExecutorAgent** 使用的工具：

- `mark_step`: 標記步驟狀態

---

### 8. SandboxManager (`app/sandbox/sandbox_manager.py`)

**這是什麼？** 管理 Docker 容器和 MCP 工具的模組，讓 Agent 能夠在隔離環境中操作檔案和執行指令。

**它做什麼？**

- 啟動/停止 Docker 容器
- 建立 MCP (Model Context Protocol) 工具連線
- 將工作目錄掛載到容器中
- 提供檔案系統和 Shell 工具給 Agent 使用

**MCP 工具說明：**

| 工具類型 | 提供的功能 | 使用者 |
|---------|-----------|--------|
| Filesystem (唯讀) | `read_file`, `list_directory`, `get_file_info` | PlannerAgent |
| Filesystem (完整) | 讀寫檔案、建立/刪除目錄 | ExecutorAgent |
| Shell | `run_command`, `run_python` | ExecutorAgent |
| 使用者自訂 MCP | 依設定而定 | ExecutorAgent |

**使用範例：**

```python
from cortex import Cortex

# 基本 Sandbox 模式
cortex = Cortex(
    model=model,
    workspace="./my-project",    # 掛載的目錄
    enable_filesystem=True,      # 啟用檔案系統工具
    enable_shell=True,           # 啟用 Shell 工具
)

# 加入自訂 MCP 伺服器
cortex = Cortex(
    model=model,
    workspace="./my-project",
    enable_filesystem=True,
    enable_shell=True,
    mcp_servers=[
        # 遠端 MCP 伺服器 (SSE)
        {"url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer xxx"}},
        # 本地 MCP 伺服器 (Stdio)
        {"command": "npx", "args": ["-y", "@mcp/server-github"]},
    ],
)
```

**注意：** 使用 Sandbox 功能需要安裝並啟動 Docker。

---

### 9. 並行執行引擎

**這是什麼？** Cortex 的核心執行引擎，根據 DAG 依賴關係並行執行步驟。

**特色：**

| 功能 | 說明 |
|------|------|
| **DAG 執行** | 根據依賴關係自動決定執行順序 |
| **並行控制** | `Semaphore(3)` 最多同時執行 3 個步驟 |
| **自動重試** | 失敗的步驟自動重試，最多 3 次嘗試 |
| **結果累積** | 前置步驟的輸出會傳遞給依賴步驟 |
| **LLM 彙整** | Aggregator 將所有步驟輸出整合成最終結果 |

**執行邏輯：**

```python
# 簡化的並行執行邏輯
while pending_steps:
    # 1. 找出所有可以執行的步驟 (依賴已完成)
    ready_steps = plan.get_ready_steps()

    # 2. 並行執行 (最多 3 個同時)
    async with Semaphore(3):
        results = await asyncio.gather(*[
            execute_step_with_retry(step)
            for step in ready_steps
        ])

    # 3. 累積輸出供後續步驟使用
    step_outputs.update(results)

# 4. 彙整最終結果
final_result = await aggregator.synthesize(step_outputs)
```

**重試機制：**

```
Step 執行失敗
    ↓
嘗試 1/3 → 失敗 → 重試
    ↓
嘗試 2/3 → 失敗 → 重試
    ↓
嘗試 3/3 → 失敗 → 標記為 blocked
```

---

## 執行流程圖

```
使用者輸入任務
       │
       ▼
┌──────────────────┐
│     Cortex       │
│  (建立新 Plan)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PlannerAgent    │
│                  │
│ 1. 分析任務       │
│ 2. 拆解成步驟     │
│ 3. 建立依賴關係   │
│ 4. 呼叫 create_plan │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│              Plan (DAG)               │
│                                       │
│  steps: [A, B, C, D]                  │
│  dependencies: {2: [0,1], 3: [2]}     │
│                                       │
│       A ──────┐                       │
│               ├──► C ──► D            │
│       B ──────┘                       │
│    (A,B 可並行)                        │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│      Cortex 並行執行引擎               │
│         (Semaphore = 3)               │
│                                       │
│  while 有待執行步驟:                   │
│    1. 找出所有 ready 步驟              │
│    2. 並行執行 (最多 3 個同時)         │
│    3. 失敗自動重試 (最多 3 次)         │
│    4. 累積 step_outputs               │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│    Aggregator    │
│                  │
│ 彙整所有步驟輸出   │
│ 生成最終結果      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  回傳結果給使用者  │
└──────────────────┘
```

---

## 快速開始

### 1. 安裝依賴

需要 Python 3.12 或更高版本。

```bash
# 使用 uv (推薦)
uv sync

# 或使用 pip
pip install -e .
```

### 2. 執行測試

```bash
# 執行所有測試
uv run pytest tests/ -v

# 執行特定測試
uv run pytest tests/task/test_plan.py -v
```

### 3. 執行範例

```bash
uv run python example.py
```

### 4. 基本使用

```python
import asyncio
from google.adk.models import LiteLlm
from cortex import Cortex

# Model 設定
API_BASE_URL = "http://10.136.3.209:8000/v1"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

async def main():
    # 使用 LiteLLM 連接本地或遠端 LLM
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
    )

    # 創建 Cortex 實例
    cortex = Cortex(model=model)

    # 執行任務
    result = await cortex.execute("寫一個 Hello World 程式")
    print(result)

asyncio.run(main())
```

**其他 Model 設定範例：**

```python
# OpenAI
model = LiteLlm(
    model="openai/gpt-4",
    api_key="your-api-key"
)

# Google Gemini (透過 google-genai)
from google import genai
client = genai.Client()
model = "gemini-2.0-flash"

# Anthropic Claude
model = LiteLlm(
    model="anthropic/claude-3-sonnet",
    api_key="your-api-key"
)
```

---

## 術語表

| 術語        | 英文                 | 說明                                   |
| ----------- | -------------------- | -------------------------------------- |
| Agent       | Agent                | 一個可以自主行動的 AI 程式             |
| LLM         | Large Language Model | 大型語言模型，如 GPT-4                 |
| Plan        | Plan                 | 執行計畫，包含多個步驟                 |
| Step        | Step                 | 計畫中的單一步驟                       |
| Toolkit     | Toolkit              | 工具包，提供 Agent 可呼叫的功能        |
| Dependency  | Dependency           | 依賴關係，某步驟需要等其他步驟完成     |
| Thread-safe | Thread-safe          | 多執行緒安全，多個程式同時存取不會出錯 |
| MCP         | Model Context Protocol | Anthropic 定義的 LLM 工具通訊協定    |
| Sandbox     | Sandbox              | 沙盒環境，隔離執行不影響主系統         |
| Docker      | Docker               | 容器化技術，用於建立隔離的執行環境     |

---

## 測試覆蓋

目前共有 **101 個測試**，涵蓋所有核心功能：

| 模組           | 測試數量 | 說明                                     |
| -------------- | -------- | ---------------------------------------- |
| TaskManager    | 5        | 計畫存取、刪除、執行緒安全               |
| Plan           | 20       | 建立、更新、狀態追蹤、依賴關係、工具歷史 |
| PlanToolkit    | 8        | create_plan、update_plan、aliased tools  |
| ActToolkit     | 8        | mark_step、aliased tools                 |
| BaseAgent      | 8        | 初始化、工具事件追蹤、Agent 儲存         |
| PlannerAgent   | 6        | 初始化、工具整合、agent_factory、extra_tools |
| ExecutorAgent  | 6        | 初始化、工具整合、agent_factory、extra_tools |
| Cortex         | 17       | 初始化、歷史記錄、factory、sandbox、**並行執行** |
| SandboxManager | 16       | Docker 生命週期、MCP 工具、使用者 MCP    |

---

## 常見問題

### Q: 為什麼要分成 PlannerAgent 和 ExecutorAgent？

**A:** 這是「分工合作」的概念。就像公司裡有人負責規劃專案，有人負責執行工作。這樣每個 Agent 可以專注在自己的任務上，更容易維護和擴展。

### Q: Plan 和 TaskManager 有什麼不同？

**A:**

- **Plan** 是「一份計畫」，像是一張待辦清單
- **TaskManager** 是「放計畫的倉庫」，可以存放多份計畫

### Q: 什麼是 Lazy Import？

**A:** 在 `BaseAgent` 中，我們使用「延遲載入」。意思是 Google ADK 的模組不會在程式啟動時就載入，而是在真正需要用到時才載入。這樣可以加快程式啟動速度。

---

## 授權

MIT License
