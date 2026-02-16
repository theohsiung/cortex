# Executor Decoupling — Known Issues & Follow-ups

Branch `feature/executor-decoupling` merged 2026-02-17. Below are known issues to address in subsequent work.

---

## High Priority

### 1. `Plan.add_steps()` 缺少輸入驗證

**位置:** `app/task/plan.py` — `add_steps()`

沒有驗證：
- `new_steps` 是否為空
- `insert_after` 是否在合法範圍內
- `new_dependencies` 的 index 是否指向合法的 step
- `new_intents` 的 key 是否在 `[0, len(new_steps))` 範圍內

可能導致靜默產生無效 plan。

### 2. Replanner dependency index 未驗證

**位置:** `app/agents/replanner/replanner_agent.py` — `_parse_replan_response()`

Parse 出來的 `new_dependencies` 沒有檢查：
- Index 是否在 `[0, len(new_steps))` 範圍內
- 是否構成合法 DAG（無環）

同樣地，`new_intents` 的值沒有檢查是否為 `available_intents` 中的合法 key。

### 3. 外部 executor factory 錯誤處理不足

**位置:** `cortex.py` — `_execute_step()` 中的 external executor 路徑

`factory()` 或 `BaseAgent()` 如果 raise exception，會直接 bubble up 到 retry loop。但這不是 transient error，不應該浪費 retry 次數。應區分 factory 初始化錯誤（non-retryable）與執行錯誤（retryable）。

---

## Medium Priority

### 4. Verifier 在 model=None 時靜默通過

**位置:** `app/agents/verifier/verifier.py`

當 `self.model is None` 時回傳 `VerifyResult(passed=True, ...)`，語義不清楚——無法區分「驗證通過」與「未驗證」。建議使用類似 `"UNVERIFIED (no model available)"` 的 notes 來標示。

### 5. Replanner / Verifier 靜默 fallback 缺少 logging

**位置:**
- `app/agents/replanner/replanner_agent.py` — `_parse_replan_response()` 在 JSON parse 失敗時回傳 `give_up`，沒有 log
- `app/agents/verifier/verifier.py` — LLM response parsing 失敗時靜默回傳空字串

Debug 時會很痛苦。

### 6. `cortex.py` 中的 inline import

**位置:** `cortex.py` — external executor 路徑

```python
from app.agents.base.base_agent import BaseAgent as _BaseAgent, ExecutionContext
```

此 import 在 execution loop 中，應移至 module level。

---

## Low Priority

### 7. String-based status constants

整個 codebase 使用 `"completed"`, `"blocked"`, `"in_progress"`, `"not_started"` 等字串常數，容易 typo 且不 type-safe。建議改用 `Enum`。

### 8. 重複的 response parsing pattern

`verifier.py` 和 `cortex.py` 中有幾乎相同的 LLM response parsing 邏輯（`hasattr(event, "content")` 一系列 check），可抽成共用 utility。

### 9. 硬編碼的 magic numbers

- `asyncio.Semaphore(3)` — concurrency limit
- `RESULT_MAX_LENGTH = 200` — truncation limit

應改為可配置或至少加註說明。（pydantic-config branch 會處理 semaphore 部分）
