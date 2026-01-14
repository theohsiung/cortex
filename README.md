# Cortex - AI Agent Framework

Cortex 是一個基於 Google ADK (Agent Development Kit) 的 AI Agent 框架。它能夠將複雜的任務自動分解成多個步驟，然後逐一執行完成。

## 這個專案在做什麼？

想像你有一個很聰明的助理，當你給他一個複雜的任務時，他會：

1. **規劃 (Planning)**: 先把任務拆解成多個小步驟
2. **執行 (Execution)**: 一步一步完成每個小步驟
3. **回報 (Reporting)**: 告訴你完成了什麼

這就是 Cortex 在做的事情！

```
使用者: "幫我整理房間"
    ↓
Cortex 規劃:
    步驟 1: 收拾桌面
    步驟 2: 整理書櫃
    步驟 3: 吸地板
    ↓
Cortex 執行:
    ✓ 步驟 1 完成
    ✓ 步驟 2 完成
    ✓ 步驟 3 完成
    ↓
Cortex 回報: "房間整理完成！"
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
│   ├── task/              # 任務管理
│   │   ├── plan.py        # Plan 類別 - 儲存執行計畫
│   │   └── task_manager.py # TaskManager - 全域計畫管理器
│   └── tools/             # LLM 可呼叫的工具
│       ├── plan_toolkit.py # 規劃工具 (create_plan, update_plan)
│       └── act_toolkit.py  # 執行工具 (mark_step)
├── tests/                 # 測試程式碼
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
- 呼叫 ExecutorAgent 來執行步驟
- 回報最終結果

```python
# 使用範例
from cortex import Cortex

cortex = Cortex(model=your_llm_model)
result = await cortex.execute("幫我寫一個計算機程式")
print(result)
```

---

### 2. BaseAgent (`app/agents/base/base_agent.py`)

**這是什麼？** 所有 Agent 的「爸爸」，定義了 Agent 的基本行為。

**它做什麼？**
- 包裝 Google ADK 的 LlmAgent
- 管理與 LLM (大型語言模型) 的對話
- 追蹤工具呼叫事件
- 連接到 Plan (執行計畫)

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
- `steps`: 步驟清單 (例如: ["步驟1", "步驟2", "步驟3"])
- `step_statuses`: 每個步驟的狀態
- `step_notes`: 每個步驟的備註
- `dependencies`: 步驟之間的依賴關係

**依賴關係是什麼？**
```
假設有 3 個步驟:
  步驟 0: 買食材
  步驟 1: 切菜      (依賴步驟 0，要先買食材才能切)
  步驟 2: 炒菜      (依賴步驟 1，要先切菜才能炒)

dependencies = {1: [0], 2: [1]}
意思是: 步驟 1 依賴步驟 0，步驟 2 依賴步驟 1
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
│ 3. 呼叫 create_plan │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      Plan        │
│                  │
│ steps: [A, B, C] │
│ status: 全部待執行│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ExecutorAgent   │◄─────────┐
│                  │          │
│ 1. 取得待執行步驟 │          │
│ 2. 執行步驟      │          │ 重複直到
│ 3. 呼叫 mark_step │          │ 全部完成
└────────┬─────────┘          │
         │                    │
         ▼                    │
    還有步驟？ ──是──────────────┘
         │
         否
         │
         ▼
┌──────────────────┐
│  回傳結果給使用者  │
└──────────────────┘
```

---

## 快速開始

### 1. 安裝依賴

```bash
# 使用 uv (推薦)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 執行測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定測試
pytest tests/task/test_plan.py -v
```

### 3. 基本使用

```python
import asyncio
from cortex import Cortex
from google.adk.models.lite_llm import LiteLlm

# 設定 LLM 模型
model = LiteLlm(
    model="openai/gpt-4",
    api_key="your-api-key"
)

# 創建 Cortex 實例
cortex = Cortex(model=model)

# 執行任務
async def main():
    result = await cortex.execute("寫一個 Hello World 程式")
    print(result)

asyncio.run(main())
```

---

## 術語表

| 術語 | 英文 | 說明 |
|------|------|------|
| Agent | Agent | 一個可以自主行動的 AI 程式 |
| LLM | Large Language Model | 大型語言模型，如 GPT-4 |
| Plan | Plan | 執行計畫，包含多個步驟 |
| Step | Step | 計畫中的單一步驟 |
| Toolkit | Toolkit | 工具包，提供 Agent 可呼叫的功能 |
| Dependency | Dependency | 依賴關係，某步驟需要等其他步驟完成 |
| Thread-safe | Thread-safe | 多執行緒安全，多個程式同時存取不會出錯 |

---

## 測試覆蓋

目前共有 **47 個測試**，涵蓋所有核心功能：

| 模組 | 測試數量 | 說明 |
|------|----------|------|
| TaskManager | 5 | 計畫存取、刪除、執行緒安全 |
| Plan | 12 | 建立、更新、狀態追蹤、依賴關係 |
| PlanToolkit | 7 | create_plan、update_plan 工具 |
| ActToolkit | 7 | mark_step 工具 |
| BaseAgent | 6 | 初始化、工具事件追蹤 |
| PlannerAgent | 3 | 初始化、工具整合 |
| ExecutorAgent | 3 | 初始化、工具整合 |
| Cortex | 4 | 初始化、歷史記錄、清理 |

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
