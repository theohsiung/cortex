# Refactor: DRY Principle & Logging

## Summary

Extract duplicated code and add proper logging to improve code quality.

## Changes

### 1. Extract `should_include_aliases` to BaseAgent

**Problem:** `_should_include_aliases` method is duplicated in both `PlannerAgent` and `ExecutorAgent`.

**Solution:** Move to `BaseAgent` as a static method.

```python
# app/agents/base/base_agent.py
class BaseAgent:
    @staticmethod
    def should_include_aliases(model: Any) -> bool:
        """Check if model supports aliased tool names with special characters."""
        if model is None:
            return False
        model_str = str(model).lower()
        if "gemini" in model_str:
            return False
        if "gpt-oss" in model_str or "openai" in model_str:
            return True
        return False
```

**Affected files:**
- `app/agents/base/base_agent.py` - Add method
- `app/agents/planner/planner_agent.py` - Remove method, use inherited
- `app/agents/executor/executor_agent.py` - Remove method, use inherited

### 2. Add Logging to cortex.py

**Problem:** Using `print` statements with ANSI colors instead of proper logging.

**Solution:** Use Python's `logging` module with `__name__` pattern.

```python
# cortex.py
import logging

logger = logging.getLogger(__name__)

# Replace print statements:
logger.warning("Step %d failed (attempt %d/%d): %s. Retrying...", ...)
```

**Log levels:**
| Situation | Level |
|-----------|-------|
| Step retry | WARNING |
| Step final failure | ERROR |
| Execution progress | INFO |
| Detailed info | DEBUG |

**Configuration:** Library does not configure logging. Users configure in their entry point:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Testing

- Run existing tests to ensure no regressions
- No new tests needed (behavior unchanged)
