# BaseAgent Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the BaseAgent architecture as the foundation for Cortex agent framework, wrapping Google ADK's LlmAgent with Plan integration and tool event tracking.

**Architecture:** BaseAgent wraps ADK LlmAgent using composition. TaskManager provides global Plan storage. PlanToolkit/ActToolkit expose LLM-callable tools with manual schemas. Session is per-task, History is managed by Cortex for cross-task context.

**Tech Stack:** Python 3.12, Google ADK, pytest

---

## Task 1: TaskManager

**Files:**
- Create: `app/task/__init__.py`
- Create: `app/task/task_manager.py`
- Create: `tests/task/__init__.py`
- Create: `tests/task/test_task_manager.py`

**Step 1: Create directory structure**

```bash
mkdir -p app/task tests/task
touch app/task/__init__.py tests/task/__init__.py
```

**Step 2: Write failing tests for TaskManager**

```python
# tests/task/test_task_manager.py
import pytest
from app.task.task_manager import TaskManager


class TestTaskManager:
    def setup_method(self):
        """Clear TaskManager state before each test"""
        TaskManager._plans.clear()

    def test_set_and_get_plan(self):
        """Should store and retrieve a plan by ID"""
        plan = {"title": "Test Plan"}
        TaskManager.set_plan("plan_1", plan)

        result = TaskManager.get_plan("plan_1")

        assert result == plan

    def test_get_nonexistent_plan_returns_none(self):
        """Should return None for unknown plan ID"""
        result = TaskManager.get_plan("nonexistent")

        assert result is None

    def test_remove_plan(self):
        """Should remove a plan by ID"""
        plan = {"title": "Test Plan"}
        TaskManager.set_plan("plan_1", plan)

        TaskManager.remove_plan("plan_1")

        assert TaskManager.get_plan("plan_1") is None

    def test_remove_nonexistent_plan_no_error(self):
        """Should not raise error when removing nonexistent plan"""
        TaskManager.remove_plan("nonexistent")  # Should not raise

    def test_thread_safety(self):
        """Should handle concurrent access safely"""
        import threading

        def set_plans():
            for i in range(100):
                TaskManager.set_plan(f"plan_{i}", {"id": i})

        threads = [threading.Thread(target=set_plans) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have stored plans without errors
        assert TaskManager.get_plan("plan_0") is not None
```

**Step 3: Run tests to verify they fail**

```bash
cd /home/theo/projects/cortex/.worktrees/base-agent
.venv/bin/pytest tests/task/test_task_manager.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.task.task_manager'"

**Step 4: Implement TaskManager**

```python
# app/task/task_manager.py
from threading import Lock
from typing import Any, Optional


class TaskManager:
    """Global Plan manager (singleton pattern via class methods)"""

    _lock = Lock()
    _plans: dict[str, Any] = {}

    @classmethod
    def set_plan(cls, plan_id: str, plan: Any) -> None:
        """Register a plan with the given ID"""
        with cls._lock:
            cls._plans[plan_id] = plan

    @classmethod
    def get_plan(cls, plan_id: str) -> Optional[Any]:
        """Retrieve a plan by ID, returns None if not found"""
        with cls._lock:
            return cls._plans.get(plan_id)

    @classmethod
    def remove_plan(cls, plan_id: str) -> None:
        """Remove a plan by ID, no-op if not found"""
        with cls._lock:
            cls._plans.pop(plan_id, None)
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/task/test_task_manager.py -v
```

Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add app/task/ tests/task/
git commit -m "feat: add TaskManager for global Plan storage"
```

---

## Task 2: Plan Class

**Files:**
- Create: `app/task/plan.py`
- Create: `tests/task/test_plan.py`

**Step 1: Write failing tests for Plan**

```python
# tests/task/test_plan.py
import pytest
from app.task.plan import Plan


class TestPlan:
    def test_create_empty_plan(self):
        """Should create plan with empty defaults"""
        plan = Plan()

        assert plan.title == ""
        assert plan.steps == []
        assert plan.dependencies == {}

    def test_create_plan_with_title_and_steps(self):
        """Should create plan with provided values"""
        plan = Plan(
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"]
        )

        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        # Default sequential dependencies
        assert plan.dependencies == {1: [0], 2: [1]}

    def test_create_plan_with_custom_dependencies(self):
        """Should use custom dependencies when provided"""
        plan = Plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]}  # C depends on both A and B
        )

        assert plan.dependencies == {2: [0, 1]}

    def test_update_plan(self):
        """Should update plan properties"""
        plan = Plan()

        plan.update(
            title="Updated",
            steps=["New Step 1", "New Step 2"],
            dependencies={1: [0]}
        )

        assert plan.title == "Updated"
        assert plan.steps == ["New Step 1", "New Step 2"]

    def test_step_statuses_initialized(self):
        """Should initialize all steps as not_started"""
        plan = Plan(steps=["A", "B", "C"])

        assert plan.step_statuses["A"] == "not_started"
        assert plan.step_statuses["B"] == "not_started"
        assert plan.step_statuses["C"] == "not_started"

    def test_mark_step_status(self):
        """Should update step status"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="in_progress")

        assert plan.step_statuses["A"] == "in_progress"

    def test_mark_step_with_notes(self):
        """Should store step notes"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="completed", step_notes="Done successfully")

        assert plan.step_notes["A"] == "Done successfully"

    def test_get_ready_steps_initial(self):
        """Should return first step when no dependencies completed"""
        plan = Plan(steps=["A", "B", "C"])

        ready = plan.get_ready_steps()

        assert ready == [0]  # Only first step is ready

    def test_get_ready_steps_after_completion(self):
        """Should return next steps after dependencies complete"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")

        ready = plan.get_ready_steps()

        assert ready == [1]  # Second step is now ready

    def test_get_ready_steps_parallel(self):
        """Should return multiple ready steps for parallel execution"""
        plan = Plan(
            steps=["A", "B", "C", "D"],
            dependencies={2: [0], 3: [1]}  # C depends on A, D depends on B
        )

        ready = plan.get_ready_steps()

        assert set(ready) == {0, 1}  # A and B can run in parallel

    def test_get_progress(self):
        """Should return progress statistics"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="in_progress")

        progress = plan.get_progress()

        assert progress["total"] == 3
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["not_started"] == 1

    def test_format_plan(self):
        """Should return formatted string representation"""
        plan = Plan(title="Test", steps=["A", "B"])

        output = plan.format()

        assert "Test" in output
        assert "A" in output
        assert "B" in output
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/task/test_plan.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.task.plan'"

**Step 3: Implement Plan class**

```python
# app/task/plan.py
from typing import Optional


class Plan:
    """Represents an execution plan with steps, dependencies, and status tracking"""

    def __init__(
        self,
        title: str = "",
        steps: list[str] = None,
        dependencies: dict[int, list[int]] = None
    ):
        self.title = title
        self.steps = steps if steps else []

        # Auto-generate sequential dependencies if not provided
        if dependencies is not None:
            self.dependencies = dependencies
        elif len(self.steps) > 1:
            self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))}
        else:
            self.dependencies = {}

        # Initialize status tracking
        self.step_statuses: dict[str, str] = {step: "not_started" for step in self.steps}
        self.step_notes: dict[str, str] = {step: "" for step in self.steps}

    def update(
        self,
        title: Optional[str] = None,
        steps: Optional[list[str]] = None,
        dependencies: Optional[dict[int, list[int]]] = None
    ) -> None:
        """Update plan properties"""
        if title is not None:
            self.title = title

        if steps is not None:
            self.steps = steps
            # Reinitialize status tracking for new steps
            self.step_statuses = {step: "not_started" for step in self.steps}
            self.step_notes = {step: "" for step in self.steps}

            # Auto-generate dependencies if not provided
            if dependencies is None and len(self.steps) > 1:
                self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))}

        if dependencies is not None:
            self.dependencies = dependencies

    def mark_step(
        self,
        step_index: int,
        step_status: Optional[str] = None,
        step_notes: Optional[str] = None
    ) -> None:
        """Mark a step with status and/or notes"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        step = self.steps[step_index]

        if step_status is not None:
            self.step_statuses[step] = step_status

        if step_notes is not None:
            self.step_notes[step] = step_notes

    def get_ready_steps(self) -> list[int]:
        """Get indices of steps ready to execute (dependencies satisfied)"""
        ready = []

        for idx, step in enumerate(self.steps):
            # Skip if already started or completed
            if self.step_statuses[step] != "not_started":
                continue

            # Check if all dependencies are completed
            deps = self.dependencies.get(idx, [])
            all_deps_done = all(
                self.step_statuses[self.steps[dep]] == "completed"
                for dep in deps
            )

            if all_deps_done:
                ready.append(idx)

        return ready

    def get_progress(self) -> dict[str, int]:
        """Get progress statistics"""
        statuses = list(self.step_statuses.values())
        return {
            "total": len(self.steps),
            "completed": statuses.count("completed"),
            "in_progress": statuses.count("in_progress"),
            "blocked": statuses.count("blocked"),
            "not_started": statuses.count("not_started")
        }

    def format(self) -> str:
        """Format plan for display"""
        lines = [f"Plan: {self.title}", "=" * 40, ""]

        progress = self.get_progress()
        pct = (progress["completed"] / progress["total"] * 100) if progress["total"] > 0 else 0
        lines.append(f"Progress: {progress['completed']}/{progress['total']} ({pct:.1f}%)")
        lines.append("")
        lines.append("Steps:")

        status_symbols = {
            "not_started": "[ ]",
            "in_progress": "[→]",
            "completed": "[✓]",
            "blocked": "[!]"
        }

        for idx, step in enumerate(self.steps):
            symbol = status_symbols.get(self.step_statuses[step], "[ ]")
            deps = self.dependencies.get(idx, [])
            dep_str = f" (depends on: {deps})" if deps else ""
            lines.append(f"  {idx}: {symbol} {step}{dep_str}")

            if self.step_notes[step]:
                lines.append(f"      Notes: {self.step_notes[step]}")

        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/task/test_plan.py -v
```

Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add app/task/plan.py tests/task/test_plan.py
git commit -m "feat: add Plan class for step and dependency tracking"
```

---

## Task 3: PlanToolkit

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/plan_toolkit.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_plan_toolkit.py`

**Step 1: Create directory structure**

```bash
mkdir -p tests/tools
touch app/tools/__init__.py tests/tools/__init__.py
```

**Step 2: Write failing tests for PlanToolkit**

```python
# tests/tools/test_plan_toolkit.py
import pytest
from app.task.plan import Plan
from app.tools.plan_toolkit import PlanToolkit


class TestPlanToolkit:
    def setup_method(self):
        self.plan = Plan()
        self.toolkit = PlanToolkit(self.plan)

    def test_create_plan(self):
        """Should populate plan with title and steps"""
        result = self.toolkit.create_plan(
            title="Test Plan",
            steps=["Step 1", "Step 2"]
        )

        assert self.plan.title == "Test Plan"
        assert self.plan.steps == ["Step 1", "Step 2"]
        assert "Test Plan" in result

    def test_create_plan_with_dependencies(self):
        """Should use provided dependencies"""
        self.toolkit.create_plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]}
        )

        assert self.plan.dependencies == {2: [0, 1]}

    def test_create_plan_auto_dependencies(self):
        """Should generate sequential dependencies when not provided"""
        self.toolkit.create_plan(
            title="Test",
            steps=["A", "B", "C"]
        )

        assert self.plan.dependencies == {1: [0], 2: [1]}

    def test_update_plan_title(self):
        """Should update only title when only title provided"""
        self.toolkit.create_plan(title="Original", steps=["A", "B"])

        self.toolkit.update_plan(title="Updated")

        assert self.plan.title == "Updated"
        assert self.plan.steps == ["A", "B"]

    def test_update_plan_steps(self):
        """Should update steps"""
        self.toolkit.create_plan(title="Test", steps=["A"])

        self.toolkit.update_plan(steps=["A", "B", "C"])

        assert self.plan.steps == ["A", "B", "C"]

    def test_get_tool_declarations(self):
        """Should return list of FunctionDeclarations"""
        declarations = self.toolkit.get_tool_declarations()

        assert len(declarations) == 2
        names = [d.name for d in declarations]
        assert "create_plan" in names
        assert "update_plan" in names

    def test_get_tool_functions(self):
        """Should return dict mapping names to functions"""
        functions = self.toolkit.get_tool_functions()

        assert "create_plan" in functions
        assert "update_plan" in functions
        assert callable(functions["create_plan"])
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/tools/test_plan_toolkit.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.tools.plan_toolkit'"

**Step 4: Implement PlanToolkit**

```python
# app/tools/plan_toolkit.py
from google.genai.types import FunctionDeclaration
from app.task.plan import Plan


class PlanToolkit:
    """Planner tools: create and update plans"""

    def __init__(self, plan: Plan):
        self.plan = plan

    # Schema definitions
    CREATE_PLAN_SCHEMA = FunctionDeclaration(
        name="create_plan",
        description="Create a new execution plan. Break down the task into executable steps.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Plan title, briefly describe the goal"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of steps, each should be a concrete executable action"
                },
                "dependencies": {
                    "type": "object",
                    "description": "Step dependencies. Format: {\"1\": [0]} means step 1 depends on step 0"
                }
            },
            "required": ["title", "steps"]
        }
    )

    UPDATE_PLAN_SCHEMA = FunctionDeclaration(
        name="update_plan",
        description="Update the existing plan's title, steps, or dependencies",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "New title (optional)"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New steps list (optional)"
                },
                "dependencies": {
                    "type": "object",
                    "description": "New dependencies (optional)"
                }
            }
        }
    )

    def create_plan(
        self,
        title: str,
        steps: list[str],
        dependencies: dict[int, list[int]] = None
    ) -> str:
        """Create a new plan with title, steps, and optional dependencies"""
        if dependencies is None and len(steps) > 1:
            dependencies = {i: [i - 1] for i in range(1, len(steps))}

        self.plan.update(title=title, steps=steps, dependencies=dependencies)
        return f"Plan created:\n{self.plan.format()}"

    def update_plan(
        self,
        title: str = None,
        steps: list[str] = None,
        dependencies: dict[int, list[int]] = None
    ) -> str:
        """Update existing plan"""
        self.plan.update(title=title, steps=steps, dependencies=dependencies)
        return f"Plan updated:\n{self.plan.format()}"

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Return schema list for LlmAgent"""
        return [self.CREATE_PLAN_SCHEMA, self.UPDATE_PLAN_SCHEMA]

    def get_tool_functions(self) -> dict:
        """Return function mapping for tool execution"""
        return {
            "create_plan": self.create_plan,
            "update_plan": self.update_plan
        }
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/tools/test_plan_toolkit.py -v
```

Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add app/tools/ tests/tools/
git commit -m "feat: add PlanToolkit with create_plan and update_plan tools"
```

---

## Task 4: ActToolkit

**Files:**
- Create: `app/tools/act_toolkit.py`
- Create: `tests/tools/test_act_toolkit.py`

**Step 1: Write failing tests for ActToolkit**

```python
# tests/tools/test_act_toolkit.py
import pytest
from app.task.plan import Plan
from app.tools.act_toolkit import ActToolkit


class TestActToolkit:
    def setup_method(self):
        self.plan = Plan(title="Test", steps=["A", "B", "C"])
        self.toolkit = ActToolkit(self.plan)

    def test_mark_step_in_progress(self):
        """Should mark step as in_progress"""
        result = self.toolkit.mark_step(step_index=0, status="in_progress")

        assert self.plan.step_statuses["A"] == "in_progress"
        assert "0" in result
        assert "in_progress" in result

    def test_mark_step_completed(self):
        """Should mark step as completed"""
        self.toolkit.mark_step(step_index=0, status="completed")

        assert self.plan.step_statuses["A"] == "completed"

    def test_mark_step_with_notes(self):
        """Should store notes with step"""
        self.toolkit.mark_step(
            step_index=0,
            status="completed",
            notes="Successfully completed"
        )

        assert self.plan.step_notes["A"] == "Successfully completed"

    def test_mark_step_blocked(self):
        """Should mark step as blocked"""
        self.toolkit.mark_step(step_index=1, status="blocked", notes="Dependency failed")

        assert self.plan.step_statuses["B"] == "blocked"

    def test_mark_step_invalid_index(self):
        """Should raise error for invalid step index"""
        with pytest.raises(ValueError):
            self.toolkit.mark_step(step_index=99, status="completed")

    def test_get_tool_declarations(self):
        """Should return list with mark_step declaration"""
        declarations = self.toolkit.get_tool_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "mark_step"

    def test_get_tool_functions(self):
        """Should return dict with mark_step function"""
        functions = self.toolkit.get_tool_functions()

        assert "mark_step" in functions
        assert callable(functions["mark_step"])
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/tools/test_act_toolkit.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.tools.act_toolkit'"

**Step 3: Implement ActToolkit**

```python
# app/tools/act_toolkit.py
from google.genai.types import FunctionDeclaration
from app.task.plan import Plan


class ActToolkit:
    """Executor tools: mark step status"""

    def __init__(self, plan: Plan):
        self.plan = plan

    MARK_STEP_SCHEMA = FunctionDeclaration(
        name="mark_step",
        description="Mark a step's execution status and optional notes",
        parameters={
            "type": "object",
            "properties": {
                "step_index": {
                    "type": "integer",
                    "description": "Step index (0-based)"
                },
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "blocked"],
                    "description": "Step status"
                },
                "notes": {
                    "type": "string",
                    "description": "Execution result or notes"
                }
            },
            "required": ["step_index", "status"]
        }
    )

    def mark_step(
        self,
        step_index: int,
        status: str,
        notes: str = None
    ) -> str:
        """Mark step status and notes"""
        self.plan.mark_step(
            step_index=step_index,
            step_status=status,
            step_notes=notes
        )
        return f"Step {step_index} marked as {status}"

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Return schema list for LlmAgent"""
        return [self.MARK_STEP_SCHEMA]

    def get_tool_functions(self) -> dict:
        """Return function mapping for tool execution"""
        return {"mark_step": self.mark_step}
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/tools/test_act_toolkit.py -v
```

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add app/tools/act_toolkit.py tests/tools/test_act_toolkit.py
git commit -m "feat: add ActToolkit with mark_step tool"
```

---

## Task 5: BaseAgent

**Files:**
- Create: `app/agents/__init__.py`
- Create: `app/agents/base/__init__.py`
- Create: `app/agents/base/base_agent.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/base/__init__.py`
- Create: `tests/agents/base/test_base_agent.py`

**Step 1: Create directory structure**

```bash
mkdir -p app/agents/base tests/agents/base
touch app/agents/__init__.py app/agents/base/__init__.py
touch tests/agents/__init__.py tests/agents/base/__init__.py
```

**Step 2: Write failing tests for BaseAgent**

```python
# tests/agents/base/test_base_agent.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.agents.base.base_agent import BaseAgent, AgentResult
from app.task.task_manager import TaskManager
from app.task.plan import Plan


class TestBaseAgent:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_without_plan(self):
        """Should initialize without plan when plan_id not provided"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test instruction"
        )

        assert agent.plan is None
        assert agent.plan_id is None

    def test_init_with_plan(self):
        """Should get plan from TaskManager when plan_id provided"""
        plan = Plan(title="Test")
        TaskManager.set_plan("plan_1", plan)

        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test",
            plan_id="plan_1"
        )

        assert agent.plan is plan
        assert agent.plan_id == "plan_1"

    def test_tool_events_initialized_empty(self):
        """Should initialize with empty tool events"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        assert agent._tool_events == []

    def test_track_tool_event(self):
        """Should track tool call events"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        agent._track_tool_event({
            "type": "call",
            "name": "test_tool",
            "args": {"arg1": "value1"}
        })

        assert len(agent._tool_events) == 1
        assert agent._tool_events[0]["name"] == "test_tool"

    def test_get_tool_summary(self):
        """Should return tool usage statistics"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        agent._tool_events = [
            {"type": "call", "name": "tool_a"},
            {"type": "response", "name": "tool_a"},
            {"type": "call", "name": "tool_b"},
            {"type": "response", "name": "tool_b"},
            {"type": "call", "name": "tool_a"},
            {"type": "response", "name": "tool_a"},
        ]

        summary = agent.get_tool_summary()

        assert summary["total_calls"] == 3
        assert summary["total_responses"] == 3
        assert set(summary["tools_used"]) == {"tool_a", "tool_b"}


class TestAgentResult:
    def test_agent_result_creation(self):
        """Should create AgentResult with all fields"""
        result = AgentResult(
            events=[{"event": 1}],
            output="Test output",
            is_complete=True
        )

        assert result.events == [{"event": 1}]
        assert result.output == "Test output"
        assert result.is_complete is True
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/agents/base/test_base_agent.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.agents.base.base_agent'"

**Step 4: Implement BaseAgent**

```python
# app/agents/base/base_agent.py
from dataclasses import dataclass
from typing import Any, Optional
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.task.task_manager import TaskManager


@dataclass
class AgentResult:
    """Result from agent execution"""
    events: list[Any]
    output: str
    is_complete: bool = True


class BaseAgent:
    """Base class for all agents, wraps Google ADK LlmAgent"""

    def __init__(
        self,
        name: str,
        model: Any,
        tool_declarations: list,
        tool_functions: dict,
        instruction: str,
        plan_id: str = None
    ):
        # Wrap ADK LlmAgent
        self.agent = LlmAgent(
            name=name,
            model=model,
            tools=tool_declarations,
            instruction=instruction
        )

        # Tool function mapping for execution
        self.tool_functions = tool_functions

        # Plan integration
        self.plan_id = plan_id
        self.plan = TaskManager.get_plan(plan_id) if plan_id else None

        # Event tracking
        self._tool_events: list[dict] = []

        # Session service (new session per execute call)
        self._session_service = InMemorySessionService()

    async def execute(self, query: str, max_iteration: int = 10) -> AgentResult:
        """Execute query with automatic retry loop"""
        session = await self._session_service.create_session(
            app_name=self.agent.name,
            user_id="default"
        )

        for i in range(max_iteration):
            result = await self._run_once(query, session)
            if result.is_complete:
                return result

        return self._handle_max_iteration()

    async def _run_once(self, query: str, session) -> AgentResult:
        """Single execution run"""
        runner = Runner(
            agent=self.agent,
            session_service=self._session_service,
            app_name=self.agent.name
        )

        events = []
        final_output = ""

        async for event in runner.run_async(
            user_id="default",
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            events.append(event)
            self._process_event(event)

            if event.is_final_response():
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_output += part.text

        # Check if complete (no pending tool calls)
        is_complete = True
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Has tool call, need to check if responded
                        is_complete = self._has_tool_response(events, part.function_call.name)

        return AgentResult(events=events, output=final_output, is_complete=is_complete)

    def _process_event(self, event) -> None:
        """Process event and track tool calls"""
        if not event.content or not event.content.parts:
            return

        for part in event.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                self._track_tool_event({
                    "type": "call",
                    "name": part.function_call.name,
                    "args": part.function_call.args
                })
            elif hasattr(part, 'function_response') and part.function_response:
                self._track_tool_event({
                    "type": "response",
                    "name": part.function_response.name
                })

    def _has_tool_response(self, events: list, tool_name: str) -> bool:
        """Check if tool call has a response"""
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        if part.function_response.name == tool_name:
                            return True
        return False

    def _handle_max_iteration(self) -> AgentResult:
        """Handle max iteration reached"""
        return AgentResult(
            events=[],
            output="Max iterations reached",
            is_complete=True
        )

    def _track_tool_event(self, event: dict) -> None:
        """Track tool call/response event"""
        self._tool_events.append(event)

    def get_tool_summary(self) -> dict:
        """Get tool usage statistics"""
        calls = [e for e in self._tool_events if e["type"] == "call"]
        responses = [e for e in self._tool_events if e["type"] == "response"]

        return {
            "total_calls": len(calls),
            "total_responses": len(responses),
            "tools_used": list(set(e["name"] for e in calls))
        }
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/agents/base/test_base_agent.py -v
```

Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add app/agents/ tests/agents/
git commit -m "feat: add BaseAgent wrapping ADK LlmAgent"
```

---

## Task 6: PlannerAgent

**Files:**
- Create: `app/agents/planner/__init__.py`
- Create: `app/agents/planner/planner_agent.py`
- Create: `app/agents/planner/prompts.py`
- Create: `tests/agents/planner/__init__.py`
- Create: `tests/agents/planner/test_planner_agent.py`

**Step 1: Create directory structure**

```bash
mkdir -p app/agents/planner tests/agents/planner
touch app/agents/planner/__init__.py tests/agents/planner/__init__.py
```

**Step 2: Write failing tests for PlannerAgent**

```python
# tests/agents/planner/test_planner_agent.py
import pytest
from unittest.mock import Mock
from app.agents.planner.planner_agent import PlannerAgent
from app.task.task_manager import TaskManager
from app.task.plan import Plan


class TestPlannerAgent:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan()
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_gets_plan_from_task_manager(self):
        """Should get plan from TaskManager"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert agent.plan is self.plan

    def test_has_plan_tools(self):
        """Should have create_plan and update_plan tools"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert "create_plan" in agent.tool_functions
        assert "update_plan" in agent.tool_functions

    def test_tools_modify_same_plan(self):
        """Tools should modify the TaskManager plan"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        agent.tool_functions["create_plan"](
            title="Test Plan",
            steps=["Step 1", "Step 2"]
        )

        # Verify the TaskManager plan was modified
        plan = TaskManager.get_plan("plan_1")
        assert plan.title == "Test Plan"
        assert plan.steps == ["Step 1", "Step 2"]
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/agents/planner/test_planner_agent.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 4: Implement prompts**

```python
# app/agents/planner/prompts.py

PLANNER_SYSTEM_PROMPT = """You are a task planning agent. Your job is to break down user requests into clear, executable steps.

When creating a plan:
1. Analyze the user's request carefully
2. Break it down into concrete, actionable steps
3. Consider dependencies between steps
4. Use the create_plan tool to create the plan

Each step should be:
- Specific and actionable
- Independent where possible
- Clearly described

After creating the plan, confirm what was created."""
```

**Step 5: Implement PlannerAgent**

```python
# app/agents/planner/planner_agent.py
from typing import Any
from app.agents.base.base_agent import BaseAgent
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.plan_toolkit import PlanToolkit


class PlannerAgent(BaseAgent):
    """Agent responsible for creating and updating plans"""

    def __init__(self, plan_id: str, model: Any):
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = PlanToolkit(plan)

        super().__init__(
            name="planner",
            model=model,
            tool_declarations=toolkit.get_tool_declarations(),
            tool_functions=toolkit.get_tool_functions(),
            instruction=PLANNER_SYSTEM_PROMPT,
            plan_id=plan_id
        )

    async def create_plan(self, task: str) -> str:
        """Create a plan for the given task"""
        result = await self.execute(f"Create a plan for: {task}")
        return result.output
```

**Step 6: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/agents/planner/test_planner_agent.py -v
```

Expected: All 3 tests PASS

**Step 7: Commit**

```bash
git add app/agents/planner/ tests/agents/planner/
git commit -m "feat: add PlannerAgent for plan creation"
```

---

## Task 7: ExecutorAgent

**Files:**
- Create: `app/agents/executor/__init__.py`
- Create: `app/agents/executor/executor_agent.py`
- Create: `app/agents/executor/prompts.py`
- Create: `tests/agents/executor/__init__.py`
- Create: `tests/agents/executor/test_executor_agent.py`

**Step 1: Create directory structure**

```bash
mkdir -p app/agents/executor tests/agents/executor
touch app/agents/executor/__init__.py tests/agents/executor/__init__.py
```

**Step 2: Write failing tests for ExecutorAgent**

```python
# tests/agents/executor/test_executor_agent.py
import pytest
from unittest.mock import Mock
from app.agents.executor.executor_agent import ExecutorAgent
from app.task.task_manager import TaskManager
from app.task.plan import Plan


class TestExecutorAgent:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan(title="Test", steps=["Step A", "Step B"])
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_gets_plan_from_task_manager(self):
        """Should get plan from TaskManager"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert agent.plan is self.plan

    def test_has_mark_step_tool(self):
        """Should have mark_step tool"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert "mark_step" in agent.tool_functions

    def test_mark_step_modifies_plan(self):
        """mark_step should modify the TaskManager plan"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        agent.tool_functions["mark_step"](
            step_index=0,
            status="completed",
            notes="Done"
        )

        plan = TaskManager.get_plan("plan_1")
        assert plan.step_statuses["Step A"] == "completed"
        assert plan.step_notes["Step A"] == "Done"
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/agents/executor/test_executor_agent.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 4: Implement prompts**

```python
# app/agents/executor/prompts.py

EXECUTOR_SYSTEM_PROMPT = """You are a task execution agent. Your job is to execute assigned steps and report results.

When executing a step:
1. Understand what the step requires
2. Perform the necessary actions
3. Use mark_step to report completion status

Status options:
- in_progress: Currently working on step
- completed: Step finished successfully
- blocked: Step cannot be completed

Always provide notes describing what was done or why it was blocked."""
```

**Step 5: Implement ExecutorAgent**

```python
# app/agents/executor/executor_agent.py
from typing import Any
from app.agents.base.base_agent import BaseAgent
from app.agents.executor.prompts import EXECUTOR_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.act_toolkit import ActToolkit


class ExecutorAgent(BaseAgent):
    """Agent responsible for executing plan steps"""

    def __init__(self, plan_id: str, model: Any):
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = ActToolkit(plan)

        super().__init__(
            name="executor",
            model=model,
            tool_declarations=toolkit.get_tool_declarations(),
            tool_functions=toolkit.get_tool_functions(),
            instruction=EXECUTOR_SYSTEM_PROMPT,
            plan_id=plan_id
        )

    async def execute_step(self, step_index: int, context: str = "") -> str:
        """Execute a specific step"""
        step_desc = self.plan.steps[step_index]
        query = f"Execute step {step_index}: {step_desc}"
        if context:
            query += f"\n\nContext: {context}"

        result = await self.execute(query)
        return result.output
```

**Step 6: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/agents/executor/test_executor_agent.py -v
```

Expected: All 3 tests PASS

**Step 7: Commit**

```bash
git add app/agents/executor/ tests/agents/executor/
git commit -m "feat: add ExecutorAgent for step execution"
```

---

## Task 8: Cortex Main Class

**Files:**
- Create: `cortex.py`
- Create: `tests/test_cortex.py`

**Step 1: Write failing tests for Cortex**

```python
# tests/test_cortex.py
import pytest
from unittest.mock import Mock, AsyncMock
from cortex import Cortex
from app.task.task_manager import TaskManager


class TestCortex:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_creates_empty_history(self):
        """Should initialize with empty history"""
        cortex = Cortex(model=Mock())

        assert cortex.history == []

    def test_init_stores_model(self):
        """Should store the model"""
        model = Mock()
        cortex = Cortex(model=model)

        assert cortex.model is model

    def test_history_persists_across_tasks(self):
        """History should persist after task execution"""
        cortex = Cortex(model=Mock())

        cortex.history.append({"role": "user", "content": "Task 1"})
        cortex.history.append({"role": "assistant", "content": "Done"})

        assert len(cortex.history) == 2

    def test_plan_cleanup(self):
        """Should clean up plan after task"""
        cortex = Cortex(model=Mock())

        # Simulate plan creation and cleanup
        plan_id = "test_plan"
        from app.task.plan import Plan
        TaskManager.set_plan(plan_id, Plan())

        assert TaskManager.get_plan(plan_id) is not None

        TaskManager.remove_plan(plan_id)

        assert TaskManager.get_plan(plan_id) is None
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_cortex.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'cortex'"

**Step 3: Implement Cortex**

```python
# cortex.py
import time
from typing import Any

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent


class Cortex:
    """Main orchestrator for the agent framework"""

    def __init__(self, model: Any):
        self.model = model
        self.history: list[dict] = []

    async def execute(self, query: str) -> str:
        """Execute a task with planning and execution"""
        # Record user query in history
        self.history.append({"role": "user", "content": query})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        try:
            # Create plan
            planner = PlannerAgent(plan_id=plan_id, model=self.model)
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(plan_id=plan_id, model=self.model)

            while True:
                ready_steps = plan.get_ready_steps()
                if not ready_steps:
                    break

                for step_idx in ready_steps:
                    plan.mark_step(step_idx, step_status="in_progress")
                    await executor.execute_step(step_idx, context=query)

            # Generate summary
            summary = self._generate_summary(plan)

            # Record result in history
            self.history.append({"role": "assistant", "content": summary})

            return summary

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)

    def _generate_summary(self, plan: Plan) -> str:
        """Generate execution summary"""
        progress = plan.get_progress()
        return f"""Task completed.

{plan.format()}

Summary: {progress['completed']}/{progress['total']} steps completed."""
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_cortex.py -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add cortex.py tests/test_cortex.py
git commit -m "feat: add Cortex main orchestrator class"
```

---

## Task 9: Run All Tests

**Step 1: Run complete test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short
```

Expected: All tests PASS

**Step 2: Commit final state**

```bash
git add -A
git commit -m "test: verify all tests pass" --allow-empty
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | TaskManager | 5 |
| 2 | Plan | 13 |
| 3 | PlanToolkit | 7 |
| 4 | ActToolkit | 7 |
| 5 | BaseAgent | 6 |
| 6 | PlannerAgent | 3 |
| 7 | ExecutorAgent | 3 |
| 8 | Cortex | 4 |
| **Total** | | **48** |
