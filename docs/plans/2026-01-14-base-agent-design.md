# BaseAgent 架構設計

## 概述

Cortex 是基於 Google ADK 開發的 AI Agent 框架。本文件定義 BaseAgent 的架構設計，作為所有 Agent（PlannerAgent、ExecutorAgent）的共同父類別。

## 設計決策

| 項目 | 決策 | 理由 |
|------|------|------|
| BaseAgent 角色 | 混合模式，包裝 ADK LlmAgent | 保留 ADK 功能，同時擴展自訂邏輯 |
| Plan 共享機制 | TaskManager（精簡版） | 多 Agent 共享同一份 Plan |
| 重試策略 | 內部迴圈自動重試 | 簡化呼叫方程式碼 |
| 工具事件追蹤 | 放進 BaseAgent 內部 | 緊密整合，Agent 自己管理事件 |
| Session vs History | Session 單任務、History 跨任務 | 任務隔離，但支援追問 |
| 工具 Schema | 手動定義，放在 Toolkit 類別內 | Schema 和實作放一起好維護 |

## 架構圖

```
┌──────────────────────────────────────────────────────────────┐
│                          Cortex                              │
│  history = []                                                │
│  execute(query) → 建立 Plan → Planner → Executor → 清理     │
└──────────────────────────────────────────────────────────────┘
        │
        ├──────────────────┬───────────────────┐
        ▼                  ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ TaskManager  │   │PlannerAgent  │   │ExecutorAgent │
│              │   │              │   │              │
│ set_plan()   │   │ PlanToolkit  │   │ ActToolkit   │
│ get_plan()   │   │ create_plan  │   │ mark_step    │
│ remove_plan()│   │ update_plan  │   │ + 其他工具   │
└──────────────┘   └──────────────┘   └──────────────┘
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                          ┌──────────────────┐
                          │    BaseAgent     │
                          │                  │
                          │ LlmAgent (ADK)   │
                          │ plan (參考)      │
                          │ tool_functions   │
                          │ _tool_events     │
                          │                  │
                          │ execute()        │
                          │ _track_tool_event│
                          │ get_tool_summary │
                          └──────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │      Plan        │
                          │ (todo_list.py)   │
                          └──────────────────┘
```

## 檔案結構

```
cortex/
├── app/
│   ├── agents/
│   │   ├── base/
│   │   │   └── base_agent.py      # BaseAgent 類別
│   │   ├── planner/
│   │   │   ├── planner_agent.py   # PlannerAgent
│   │   │   └── prompts.py         # Planner 專用 prompt
│   │   └── executor/
│   │       ├── executor_agent.py  # ExecutorAgent
│   │       └── prompts.py         # Executor 專用 prompt
│   ├── task/
│   │   ├── task_manager.py        # TaskManager 單例
│   │   └── plan.py                # Plan 類別
│   └── tools/
│       ├── plan_toolkit.py        # create_plan, update_plan 工具
│       └── act_toolkit.py         # mark_step 工具
├── cortex.py                      # Cortex 主類別（入口）
└── pyproject.toml
```

## 核心類別設計

### TaskManager

全域 Plan 管理器，精簡版只保留三個方法：

```python
class TaskManager:
    _lock = Lock()
    _plans: dict[str, Plan] = {}

    @classmethod
    def set_plan(cls, plan_id: str, plan: Plan) -> None:
        with cls._lock:
            cls._plans[plan_id] = plan

    @classmethod
    def get_plan(cls, plan_id: str) -> Optional[Plan]:
        with cls._lock:
            return cls._plans.get(plan_id)

    @classmethod
    def remove_plan(cls, plan_id: str) -> None:
        with cls._lock:
            cls._plans.pop(plan_id, None)
```

### BaseAgent

所有 Agent 的基礎類別，包裝 Google ADK 的 LlmAgent：

```python
class BaseAgent:
    def __init__(
        self,
        name: str,
        model,
        tool_declarations: list,      # Schema 列表
        tool_functions: dict,         # 函數映射
        instruction: str,
        plan_id: str = None
    ):
        # 包裝 ADK LlmAgent
        self.agent = LlmAgent(
            name=name,
            model=model,
            tools=tool_declarations,
            instruction=instruction
        )

        self.tool_functions = tool_functions
        self.plan_id = plan_id
        self.plan = TaskManager.get_plan(plan_id) if plan_id else None
        self._tool_events = []
        self._session_service = InMemorySessionService()

    async def execute(self, query: str, max_iteration: int = 10) -> AgentResult:
        """執行查詢，內部自動重試"""
        session = await self._session_service.create_session(
            app_name=self.agent.name,
            user_id="default"
        )

        for i in range(max_iteration):
            result = await self._run_once(query, session)
            if result.is_complete:
                return result

        return self._handle_max_iteration()

    def _track_tool_event(self, event):
        """追蹤工具呼叫事件"""
        ...

    def get_tool_summary(self) -> dict:
        """取得工具呼叫統計"""
        ...
```

### PlanToolkit

Planner 專用工具，Schema 和實作放在同一類別：

```python
class PlanToolkit:
    def __init__(self, plan: Plan):
        self.plan = plan

    CREATE_PLAN_SCHEMA = FunctionDeclaration(
        name="create_plan",
        description="建立新的執行計畫",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "計畫標題"},
                "steps": {"type": "array", "items": {"type": "string"}},
                "dependencies": {"type": "object"}
            },
            "required": ["title", "steps"]
        }
    )

    def create_plan(self, title: str, steps: list, dependencies: dict = None) -> str:
        ...

    def get_tool_declarations(self) -> list:
        return [self.CREATE_PLAN_SCHEMA, self.UPDATE_PLAN_SCHEMA]

    def get_tool_functions(self) -> dict:
        return {"create_plan": self.create_plan, "update_plan": self.update_plan}
```

### ActToolkit

Executor 專用工具：

```python
class ActToolkit:
    def __init__(self, plan: Plan):
        self.plan = plan

    MARK_STEP_SCHEMA = FunctionDeclaration(
        name="mark_step",
        description="標記步驟的執行狀態",
        parameters={
            "type": "object",
            "properties": {
                "step_index": {"type": "integer"},
                "status": {"type": "string", "enum": ["in_progress", "completed", "blocked"]},
                "notes": {"type": "string"}
            },
            "required": ["step_index", "status"]
        }
    )

    def mark_step(self, step_index: int, status: str, notes: str = None) -> str:
        ...
```

## Session 與 History 的分工

| 層級 | 管理者 | 生命週期 | 用途 |
|------|--------|----------|------|
| History | Cortex | 整個對話 | 追問時提供上下文 |
| Session | ADK | 單一任務 | Agent 內部執行狀態 |
| Plan | TaskManager | 單一任務 | 步驟追蹤 |

```python
class Cortex:
    def __init__(self):
        self.history = []  # 長期記憶

    def execute(self, user_query: str):
        self.history.append({"role": "user", "content": user_query})

        # 建立短期 Plan
        plan_id = f"plan_{int(time.time())}"
        TaskManager.set_plan(plan_id, Plan())

        # 執行...

        # 任務完成，摘要存入 history
        self.history.append({"role": "assistant", "content": summary})
        TaskManager.remove_plan(plan_id)
```

## 執行流程

1. 用戶發起任務
2. Cortex 建立空 Plan，註冊到 TaskManager
3. PlannerAgent 呼叫 `create_plan` 工具填入步驟
4. ExecutorAgent 依序執行步驟，呼叫 `mark_step` 更新狀態
5. 任務完成，摘要存入 Cortex.history
6. 清理 TaskManager 中的 Plan

## 參考

- Co-Sight: https://github.com/ZTE/Co-Sight
- Google ADK: https://github.com/google/adk-python
