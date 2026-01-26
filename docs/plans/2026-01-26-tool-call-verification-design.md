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
| `rewrite_step(idx, new_desc)` | 重寫描述 |
| `split_step(idx, new_steps, new_deps)` | 拆解 step |
| `get_step_output(idx)` | 取得成功 step 的輸出 |

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

Replanner 是 LLM Agent，負責重寫/拆解失敗的 step。

### 重試上限

- 最多重寫 2 次
- 原執行 + 2 次重寫 = 共 3 次機會

### Replanner System Prompt

```
你是一個計畫修正專家。一個執行步驟失敗了，請根據資訊決定如何修正。

## 整體計畫
{plan.format_with_outputs(step_outputs)}

## 失敗的 Step
- Index: {failed_step_idx}
- 描述: {plan.steps[failed_step_idx]}
- 已重試次數: {plan.replan_attempts[failed_step_idx]}

## 失敗的 Tool Calls
{failed_calls}

## 可用 Tools
{available_tools}

## 你可以選擇：
1. rewrite - 重寫這個 step 的描述
2. split - 把這個 step 拆成多個更小的步驟

請用 replan tool 輸出你的決定。
```

### ReplanResult 結構

```python
@dataclass
class ReplanResult:
    action: str  # "rewrite" | "split" | "give_up"
    new_description: str = None        # for rewrite
    new_steps: list[str] = None        # for split
    new_dependencies: dict = None      # for split
```

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

    step_outputs: dict[int, str] = {}

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
                step_outputs[step_idx] = output
            else:
                # 檢查重試次數
                attempts = plan.replan_attempts.get(step_idx, 0)
                if attempts >= ReplannerAgent.MAX_REPLAN_ATTEMPTS:
                    plan.mark_step(step_idx, step_status="blocked")
                    continue

                # Replan
                result = await replanner.replan(plan, step_idx, ...)
                plan.replan_attempts[step_idx] = attempts + 1

                if result.action == "rewrite":
                    plan.rewrite_step(step_idx, result.new_description)
                elif result.action == "split":
                    plan.split_step(step_idx, result.new_steps, result.new_dependencies)

                # 重設狀態為 not_started，下輪會重新執行
                plan.mark_step(step_idx, step_status="not_started")

    # 4. 彙整結果
    return await self._aggregate_results(query, plan, step_outputs)
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
    └─ fail → replan
              ├─ 次數 < 2 → rewrite/split → not_started (重新執行)
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
| `tests/task/test_plan.py` | `add_tool_call`, `update_tool_result`, `finalize_step`, `rewrite_step`, `split_step` |
| `tests/agents/verifier/test_verifier.py` | pass/fail 判定邏輯 |
| `tests/agents/replanner/test_replanner_agent.py` | rewrite/split 輸出格式 |
| `tests/test_cortex.py` | 整合測試：驗證失敗觸發 replan |

### 測試案例

```python
# Verifier 測試
def test_verify_pass_when_no_tool_calls(): ...
def test_verify_pass_when_all_success(): ...
def test_verify_fail_when_has_pending(): ...

# Plan 測試
def test_rewrite_step_updates_description(): ...
def test_split_step_updates_dag(): ...
def test_split_step_preserves_completed_steps(): ...
```
