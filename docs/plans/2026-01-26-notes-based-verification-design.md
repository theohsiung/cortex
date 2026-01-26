# Notes-Based Step Verification Design

## Problem

LLM honestly reports failures in Notes (e.g., "Unable to access Facebook API"), but Verifier only checks for pending tool calls. Steps are incorrectly marked as "completed" when:
- LLM doesn't make any tool calls (just writes text)
- LLM admits failure in Notes but verification passes

## Solution Overview

1. Require LLM to use structured status format in Notes
2. Extend Verifier to check Notes content
3. Defer replan until batch execution completes

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection scope | LLM honest failures only | LLM is already reporting failures accurately |
| Status format | `[SUCCESS]/[FAIL]` prefix | Simple, unambiguous, easy to parse |
| Format strictness | `startswith("[FAIL]")` | Allows `[FAIL]:` or `[FAIL] ` variants |
| Replan timing | Batch-end (Option B) | Maximizes parallel execution efficiency |
| PARTIAL status | Not needed | Steps should be atomic; split in planning phase |

## Implementation Details

### 1. Notes Format Specification

```
[SUCCESS]: <brief description of what was accomplished>
[FAIL]: <reason why the step could not be completed>
```

Examples:
- `[SUCCESS]: Posted joke to Facebook successfully`
- `[FAIL]: Unable to access Facebook API - no credentials available`

### 2. Verifier Changes

**File:** `app/agents/verifier/verifier.py`

```python
def verify_step(self, plan: Plan, step_idx: int) -> bool:
    # 1. Check Notes for [FAIL] marker
    notes = plan.step_notes.get(step_idx, "").strip()
    if notes.startswith("[FAIL]"):
        return False  # LLM self-reported failure

    # 2. Original pending tool call check
    tool_history = plan.step_tool_history.get(step_idx, [])
    for call in tool_history:
        if call.get("status") == "pending":
            return False

    return True
```

### 3. Executor Prompt Addition

**File:** `app/agents/executor/prompts.py`

Add to system prompt:
```
## Step Completion Reporting

When you complete a step, you MUST call mark_step with a notes field that starts with a status tag:

- [SUCCESS]: <brief description of what was accomplished>
- [FAIL]: <reason why the step could not be completed>

Examples:
- [SUCCESS]: Posted joke to Facebook successfully
- [FAIL]: Unable to access Facebook API - no credentials available
- [SUCCESS]: Generated Chinese joke about programmers
- [FAIL]: Cannot create image - no image generation tool available

Be HONEST about failures. If you cannot complete a step due to missing tools,
permissions, or external dependencies, report [FAIL] with a clear reason.
```

### 4. Deferred Replan Flow

**File:** `cortex.py`

Change from immediate replan to batch-end replan:

```python
while True:
    ready_steps = plan.get_ready_steps()
    if not ready_steps:
        break

    results = await asyncio.gather(
        *[execute_with_limit(idx) for idx in ready_steps]
    )

    # Phase 1: Classify results (no replan yet)
    blocked_steps = []
    for step_idx, result in results:
        if isinstance(result, Exception):
            plan.mark_step(step_idx, step_status="blocked", step_notes=str(result))
            blocked_steps.append(step_idx)
        else:
            plan.finalize_step(step_idx)
            if verifier.verify_step(plan, step_idx):
                step_outputs[step_idx] = result
                plan.mark_step(step_idx, step_status="completed")
            else:
                plan.mark_step(step_idx, step_status="blocked")
                blocked_steps.append(step_idx)

    # Phase 2: Batch replan after all results processed
    if blocked_steps:
        all_to_replan = set()
        for idx in blocked_steps:
            if plan.replan_attempts.get(idx, 0) < ReplannerAgent.MAX_REPLAN_ATTEMPTS:
                all_to_replan.add(idx)
                all_to_replan.update(plan.get_downstream_steps(idx))

        if all_to_replan:
            replan_result = await replanner.replan_subgraph(
                steps_to_replan=sorted(all_to_replan),
                available_tools=available_tools
            )
            # Update replan attempts for all blocked steps
            for idx in blocked_steps:
                plan.replan_attempts[idx] = plan.replan_attempts.get(idx, 0) + 1

            if replan_result.action == "redesign":
                # ... apply replan changes
```

## Files to Modify

| File | Changes |
|------|---------|
| `app/agents/verifier/verifier.py` | Add Notes checking logic |
| `app/agents/executor/prompts.py` | Add status format instructions |
| `cortex.py` | Deferred batch replan logic |
| `tests/test_verifier.py` | Add Notes verification tests |
| `tests/test_cortex.py` | Add deferred replan tests |

## Test Cases

1. **Notes [FAIL] detection**: Step with `[FAIL]: reason` should fail verification
2. **Notes [SUCCESS] passes**: Step with `[SUCCESS]: done` should pass verification
3. **No Notes passes**: Backward compatibility - empty notes should pass (unless pending tool calls)
4. **Batch replan**: Multiple blocked steps in same batch should trigger single replan
5. **Parallel execution preserved**: Blocked step doesn't stop independent parallel steps

## Migration

- Existing steps without `[SUCCESS]/[FAIL]` prefix will pass verification (backward compatible)
- No database changes required
- Prompt changes take effect immediately for new executions
