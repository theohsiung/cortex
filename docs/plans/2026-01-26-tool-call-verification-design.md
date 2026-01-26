# Tool Call 驗證機制設計

日期: 2026-01-26
分支: feature/tool-call-verification

## 目標

驗證每個 step 的 tool call 是否真實執行，偵測 LLM 幻覺。

## 範圍（第一版）

- **確定性驗證**：比對 call 與 response 是否配對
- 不做 LLM 語意審查（未來再考慮）

## 核心發現

| 項目 | 結論 |
|------|------|
| Tool result 來源 | 來自真實執行，可信任 |
| 目前記錄機制 | 只記錄有 response 的 call，無法偵測失敗 |
| 多次 tool call | FIFO 配對，順序執行下沒問題 |

---

## 整體架構

```
┌─────────────────────────────────────────────────────┐
│                     Cortex                          │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Planner  │───▶│ Executor │───▶│  Verifier    │  │
│  └──────────┘    └──────────┘    └──────┬───────┘  │
│       ▲                                 │          │
│       │         ┌──────────┐            │          │
│       └─────────│ Replanner│◀───────────┘          │
│                 └──────────┘      (if failed)      │
└─────────────────────────────────────────────────────┘
```

**元件職責：**

| 元件 | 職責 |
|------|------|
| **Planner** | 建立初始計畫 (現有) |
| **Executor** | 執行 step (現有) |
| **Verifier** | 驗證 tool call 狀態，判定 pass/fail (新增) |
| **Replanner** | 根據失敗資訊重寫/拆解 step (新增) |

**執行流程：**

1. Planner 建立計畫
2. Executor 並行執行 ready steps
3. 每個 step 完成後 → Verifier 驗證
4. 驗證失敗 → Replanner 重寫/拆解
5. 更新 Plan → 回到步驟 2
6. 全部完成或超過重試上限 → 結束

---

## 資料結構改動

### Plan 類別擴展

```python
class Plan:
    # 現有欄位
    title: str
    steps: list[str]
    dependencies: dict[int, list[int]]
    step_statuses: dict[str, str]
    step_notes: dict[str, str]
    step_tool_history: dict[int, list[dict]]
    step_files: dict[int, list[str]]

    # 新增欄位
    step_outputs: dict[int, str]      # 儲存成功 step 的輸出
    replan_attempts: dict[int, int]   # 每個 step 的重寫次數
```

### step_tool_history 新結構

```python
step_tool_history[step_idx] = [
    {
        "tool": "write_file",
        "args": {"path": "main.py", "content": "..."},
        "status": "success",       # "pending" | "success"
        "result": "File written",  # 只有 success 才有
        "call_time": "...",
        "response_time": "..."     # 只有 success 才有
    }
]
```

### Plan 新增方法

| 方法 | 功能 |
|------|------|
| `add_tool_call(step_idx, tool, args)` | 記錄 pending call |
| `update_tool_result(step_idx, tool, result)` | 更新為 success |
| `finalize_step(step_idx)` | step 結束，檢查有無 pending |
| `get_downstream_steps(step_idx)` | 取得某 step 的所有下游依賴 |
| `remove_steps(step_indices)` | 移除指定的 steps 並更新 DAG |
| `add_steps(steps, dependencies, insert_after)` | 新增 steps 到 DAG |
| `format_tool_history(step_indices)` | 格式化指定 steps 的 tool history |

### get_downstream_steps 實作

```python
def get_downstream_steps(self, step_idx: int) -> list[int]:
    """取得某個 step 的所有下游 step（直接 + 間接依賴它的）"""
    downstream = set()
    to_check = [step_idx]

    while to_check:
        current = to_check.pop()
        for idx, deps in self.dependencies.items():
            if current in deps and idx not in downstream:
                downstream.add(idx)
                to_check.append(idx)

    return sorted(downstream)
```

---

## Verifier 驗證邏輯

Verifier 是純 Python 邏輯，不需要 LLM：

```python
class Verifier:
    def verify_step(self, plan: Plan, step_idx: int) -> bool:
        """
        驗證 step 的 tool call 狀態
        Returns: True = pass, False = fail
        """
        tool_history = plan.step_tool_history.get(step_idx, [])

        # 情況 1: 沒有 tool call → pass (不需要 tool)
        if not tool_history:
            return True

        # 情況 2: 檢查是否有 pending
        for call in tool_history:
            if call["status"] == "pending":
                return False  # 有幻覺

        # 情況 3: 全部 success → pass
        return True

    def get_failed_calls(self, plan: Plan, step_idx: int) -> list[dict]:
        """取得失敗的 tool call 資訊，供 Replanner 使用"""
        tool_history = plan.step_tool_history.get(step_idx, [])
        return [call for call in tool_history if call["status"] == "pending"]
```

---

## Replanner 機制

Replanner 是 LLM Agent，負責重新設計失敗的 step 及其所有下游依賴。

### 重要概念：Subgraph Replan

當 step 失敗時，不只重寫該 step，而是重新設計整個受影響的子圖。

**範例：** `{1:[0], 2:[0], 3:[1], 4:[2], 5:[3,4], 6:[5], 7:[5]}`

```
0 → 1 → 3 ─┐
  ↘ 2 → 4 ─┴→ 5 → 6
              ↘ 7
```

如果 step 5 失敗：
- **保留**：Steps 0, 1, 2, 3, 4（已完成）
- **重新設計**：Steps 5, 6, 7（失敗的 + 下游）

### 重試上限

- 最多重寫 2 次
- 原執行 + 2 次重寫 = 共 3 次機會

### Replanner 輸入資訊

Replanner 收到**完整的 tool_history**（非 LLM 總結），以便做出準確判斷：

```
## 已完成的 Steps 詳細記錄

### Step 0: 分析需求
Tool calls:
- read_file(path="requirements.md") → "需要建立 REST API..."

### Step 1: 建立專案結構
Tool calls:
- create_directory(path="src/") → "Directory created"
- create_directory(path="tests/") → "Directory created"

### Step 3: 建立資料模型
Tool calls:
- write_file(path="src/models.py", content="...") → "File written"

### Step 4: 設定資料庫連線
Tool calls:
- write_file(path="src/database.py", content="...") → "File written"
- run_command(command="python -c 'import sqlalchemy'") → "OK"

---

## 失敗的 Step 及下游（需重新設計）

### Step 5: 撰寫 API endpoints [FAILED]
Tool calls:
- write_file(path="src/api.py", ...) → status: "pending" ❌

### Step 6: 撰寫測試 (depends on: 5)
### Step 7: 執行測試 (depends on: 5)

---

## 可用 Tools
{available_tools}

## 請重新設計 steps 5, 6, 7
- 可以合併、拆分、或完全重寫
- 輸出新的 steps 和 dependencies
```

### ReplanResult 結構

```python
@dataclass
class ReplanResult:
    action: str  # "redesign" | "give_up"
    new_steps: list[str]              # 新的 step 描述列表
    new_dependencies: dict[int, list[int]]  # 新的依賴關係（相對索引）
```

**範例輸出：**

```python
# 原本 steps 5, 6, 7 被重新設計為 3 個新 steps
ReplanResult(
    action="redesign",
    new_steps=[
        "建立 API 基本框架和路由設定",
        "實作各 endpoint 的商業邏輯",
        "撰寫單元測試並執行"
    ],
    new_dependencies={
        0: [],      # 新 step 0 (原 5) 依賴已完成的 steps (由系統處理)
        1: [0],     # 新 step 1 依賴新 step 0
        2: [1]      # 新 step 2 依賴新 step 1
    }
)

---

## Cortex 整合流程

```python
async def execute(self, query: str) -> str:
    # 1. 建立計畫
    planner = PlannerAgent(...)
    await planner.create_plan(query)

    # 2. 初始化元件
    executor = ExecutorAgent(...)
    verifier = Verifier()
    replanner = ReplannerAgent(...)

    # 3. 執行迴圈
    while True:
        ready_steps = plan.get_ready_steps()
        if not ready_steps:
            break

        for step_idx in ready_steps:
            # 執行 step
            output = await executor.execute_step(step_idx, ...)
            plan.finalize_step(step_idx)

            # 驗證
            if verifier.verify_step(plan, step_idx):
                plan.mark_step(step_idx, step_status="completed")
            else:
                # 檢查重試次數
                attempts = plan.replan_attempts.get(step_idx, 0)
                if attempts >= ReplannerAgent.MAX_REPLAN_ATTEMPTS:
                    plan.mark_step(step_idx, step_status="blocked")
                    continue

                # 找出需要重新設計的 steps (失敗的 + 下游)
                downstream = plan.get_downstream_steps(step_idx)
                steps_to_replan = [step_idx] + downstream

                # 取得已完成 steps 的 tool history
                completed_steps = [
                    i for i in range(len(plan.steps))
                    if plan.step_statuses[plan.steps[i]] == "completed"
                ]
                completed_tool_history = plan.format_tool_history(completed_steps)

                # Replan 整個子圖
                result = await replanner.replan_subgraph(
                    plan=plan,
                    steps_to_replan=steps_to_replan,
                    completed_tool_history=completed_tool_history,
                    available_tools=available_tools
                )
                plan.replan_attempts[step_idx] = attempts + 1

                if result.action == "redesign":
                    # 移除舊的 steps，加入新的
                    plan.remove_steps(steps_to_replan)
                    plan.add_steps(
                        result.new_steps,
                        result.new_dependencies,
                        insert_after=max(completed_steps)
                    )
                elif result.action == "give_up":
                    plan.mark_step(step_idx, step_status="blocked")

    # 4. 彙整結果
    return await self._aggregate_results(query, plan)
```

### 狀態轉換圖

```
not_started
    ↓ (開始執行)
in_progress
    ↓ (執行完成)
finalize_step()
    ↓
verify_step()
    ├─ pass → completed
    └─ fail → 找出下游 steps
              ↓
              replan_subgraph(failed + downstream)
              ├─ 次數 < 2 → redesign → 移除舊 steps，加入新 steps
              └─ 次數 >= 2 → blocked (放棄)
```

---

## 檔案結構

### 新增/修改的檔案

| 檔案 | 類型 | 說明 |
|------|------|------|
| `app/task/plan.py` | 修改 | 新增欄位與方法 |
| `app/agents/base/base_agent.py` | 修改 | function_call 時記錄 pending |
| `app/agents/verifier/verifier.py` | 新增 | 驗證邏輯 |
| `app/agents/replanner/replanner_agent.py` | 新增 | Replan Agent |
| `app/agents/replanner/prompts.py` | 新增 | Replanner system prompt |
| `cortex.py` | 修改 | 整合驗證與 replan 流程 |

---

## 測試計畫

### 測試檔案

| 測試檔案 | 測試項目 |
|----------|----------|
| `tests/task/test_plan.py` | `add_tool_call`, `update_tool_result`, `finalize_step`, `get_downstream_steps`, `remove_steps`, `add_steps` |
| `tests/agents/verifier/test_verifier.py` | pass/fail 判定邏輯 |
| `tests/agents/replanner/test_replanner_agent.py` | subgraph redesign 輸出格式 |
| `tests/test_cortex.py` | 整合測試：驗證失敗觸發 replan |

### 測試案例

```python
# Verifier 測試
def test_verify_pass_when_no_tool_calls(): ...
def test_verify_pass_when_all_success(): ...
def test_verify_fail_when_has_pending(): ...

# Plan 測試
def test_get_downstream_steps_direct_dependency(): ...
def test_get_downstream_steps_indirect_dependency(): ...
def test_get_downstream_steps_complex_dag(): ...
def test_remove_steps_updates_dependencies(): ...
def test_add_steps_preserves_completed(): ...
def test_format_tool_history(): ...

# Replanner 測試
def test_replan_subgraph_redesign(): ...
def test_replan_subgraph_give_up_after_max_attempts(): ...

# 整合測試
def test_failed_step_triggers_subgraph_replan(): ...
def test_completed_steps_preserved_after_replan(): ...
```
