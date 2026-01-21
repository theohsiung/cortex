# Parallel Executor Design for DAG Plans

## Overview

Enable parallel execution of independent steps in a DAG-based plan. When multiple steps have their dependencies satisfied, execute them concurrently using `asyncio.gather` with concurrency control.

## Problem

Current execution in `cortex.py` is sequential:

```python
while True:
    ready_steps = plan.get_ready_steps()
    if not ready_steps:
        break
    for step_idx in ready_steps:  # Sequential
        plan.mark_step(step_idx, step_status="in_progress")
        await executor.execute_step(step_idx, context=query)
```

Even though `get_ready_steps()` returns multiple ready steps, they execute one by one.

## Design Decisions

| Item | Decision |
|------|----------|
| Parallelism | `asyncio.gather` to execute multiple ready steps concurrently |
| Agent sharing | Share single ExecutorAgent instance across parallel steps |
| State isolation | New `ExecutionContext` dataclass passed as parameter (not instance variables) |
| Error handling | Continue execution; mark failed step as "blocked" |
| Concurrency limit | `asyncio.Semaphore(3)` - max 3 concurrent steps, dynamic backfill |
| Dependency output | `step_outputs` local dict; inject dependent outputs into context |

## Changes

### 1. base_agent.py

Add `ExecutionContext` dataclass and modify methods to receive it as parameter.

**New dataclass:**

```python
@dataclass
class ExecutionContext:
    step_index: int
    pending_calls: dict[str, dict] = field(default_factory=dict)
```

**Modified methods:**

- `execute(query, context: ExecutionContext = None)` - accept optional context
- `_run_once(query, session, context)` - pass context through
- `_process_event(event, context)` - use context.pending_calls instead of self._pending_calls
- `_record_tool_to_plan(tool_name, result, context)` - use context.step_index and context.pending_calls

**Remove instance variables:**

- `self._current_step_index`
- `self._pending_calls`

### 2. executor_agent.py

Modify `execute_step` to create and pass `ExecutionContext`:

```python
async def execute_step(self, step_index: int, context: str = "") -> str:
    exec_ctx = ExecutionContext(step_index=step_index)

    step_desc = self.plan.steps[step_index]
    query = f"Execute step {step_index}: {step_desc}"
    if context:
        query += f"\n\nContext: {context}"

    result = await self.execute(query, context=exec_ctx)
    return result.output
```

### 3. cortex.py

Replace sequential loop with parallel execution:

```python
step_outputs: dict[int, str] = {}
semaphore = asyncio.Semaphore(3)

async def execute_with_limit(step_idx: int) -> tuple[int, str]:
    # Build context from dependency outputs
    deps = plan.dependencies.get(step_idx, [])
    dep_context = "\n".join(
        f"Step {d} result: {step_outputs[d]}"
        for d in deps if d in step_outputs
    )
    full_context = f"{query}\n\n{dep_context}" if dep_context else query

    async with semaphore:
        plan.mark_step(step_idx, step_status="in_progress")
        try:
            output = await executor.execute_step(step_idx, context=full_context)
            return step_idx, output
        except Exception as e:
            return step_idx, e

while True:
    ready_steps = plan.get_ready_steps()
    if not ready_steps:
        break

    results = await asyncio.gather(
        *[execute_with_limit(idx) for idx in ready_steps]
    )

    for step_idx, result in results:
        if isinstance(result, Exception):
            plan.mark_step(step_idx, step_status="blocked", step_notes=str(result))
        else:
            step_outputs[step_idx] = result
            plan.mark_step(step_idx, step_status="completed")
```

## Execution Flow Example

Given a plan with dependencies:
```
Step 0: No deps (ready)
Step 1: No deps (ready)
Step 2: Depends on [0]
Step 3: Depends on [0, 1]
Step 4: Depends on [2, 3]
```

Execution waves:
1. **Wave 1:** Steps 0, 1 execute in parallel
2. **Wave 2:** Step 2 (gets output from 0), Step 3 (gets outputs from 0, 1) execute in parallel
3. **Wave 3:** Step 4 executes (gets outputs from 2, 3)

## Files Changed

| File | Change |
|------|--------|
| `app/agents/base/base_agent.py` | Add `ExecutionContext`, modify methods to use it |
| `app/agents/executor/executor_agent.py` | Create `ExecutionContext` in `execute_step` |
| `cortex.py` | Parallel execution loop with semaphore and step_outputs |
