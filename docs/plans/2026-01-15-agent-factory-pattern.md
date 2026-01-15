# Agent Factory Pattern Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 將 `PlannerAgent` 和 `ExecutorAgent` 的 `agent` 參數改為 `agent_factory`，讓使用者可以傳入一個接受 `tools` 的函數，解決 ADK agent 無法事後注入工具的問題。

**Architecture:** 使用者傳入 `agent_factory: Callable[[list], Agent]` 而非 agent 實例。在 `__init__` 中，先建立 toolkit 取得 tools，再呼叫 factory 建立 agent。這樣使用者的自定義 agent 就能拿到正確的 toolkit 工具。

**Tech Stack:** Python, Google ADK, Callable type hints

---

### Task 1: Update PlannerAgent to use agent_factory

**Files:**
- Modify: `app/agents/planner/planner_agent.py`
- Test: `tests/agents/planner/test_planner_agent.py`

**Step 1: Write the failing test for agent_factory**

```python
def test_agent_factory_receives_tools(self):
    """agent_factory should receive toolkit tools"""
    received_tools = []

    def my_factory(tools: list):
        received_tools.extend(tools)
        return Mock()

    PlannerAgent(
        plan_id="plan_1",
        agent_factory=my_factory
    )

    # Factory should receive create_plan and update_plan tools
    assert len(received_tools) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/planner/test_planner_agent.py::TestPlannerAgent::test_agent_factory_receives_tools -v`
Expected: FAIL with "unexpected keyword argument 'agent_factory'"

**Step 3: Update PlannerAgent to accept agent_factory**

修改 `app/agents/planner/planner_agent.py`:

```python
from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.plan_toolkit import PlanToolkit

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating and updating plans.

    Usage:
        # Default: creates LlmAgent internally
        planner = PlannerAgent(plan_id="p1", model=model)

        # Custom: pass agent_factory that receives tools
        def my_factory(tools: list):
            return LoopAgent(name="planner", tools=tools + my_extra_tools, ...)
        planner = PlannerAgent(plan_id="p1", agent_factory=my_factory)
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable[[list], Any] = None
    ):
        """
        Initialize PlannerAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that receives tools and returns an agent
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = PlanToolkit(plan)
        tools = list(toolkit.get_tool_functions().values())

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory(tools)
        elif model is not None:
            from google.adk.agents import LlmAgent

            agent = LlmAgent(
                name="planner",
                model=model,
                tools=tools,
                instruction=PLANNER_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(
            agent=agent, tool_functions=toolkit.get_tool_functions(), plan_id=plan_id
        )

    async def create_plan(self, task: str) -> str:
        """Create a plan for the given task"""
        result = await self.execute(f"Create a plan for: {task}")
        return result.output
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/planner/test_planner_agent.py::TestPlannerAgent::test_agent_factory_receives_tools -v`
Expected: PASS

**Step 5: Run all PlannerAgent tests to ensure no regression**

Run: `pytest tests/agents/planner/test_planner_agent.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add app/agents/planner/planner_agent.py tests/agents/planner/test_planner_agent.py
git commit -m "feat(planner): replace agent param with agent_factory for tool injection"
```

---

### Task 2: Update ExecutorAgent to use agent_factory

**Files:**
- Modify: `app/agents/executor/executor_agent.py`
- Test: `tests/agents/executor/test_executor_agent.py`

**Step 1: Write the failing test for agent_factory**

```python
def test_agent_factory_receives_tools(self):
    """agent_factory should receive toolkit tools"""
    received_tools = []

    def my_factory(tools: list):
        received_tools.extend(tools)
        return Mock()

    ExecutorAgent(
        plan_id="plan_1",
        agent_factory=my_factory
    )

    # Factory should receive mark_step tool
    assert len(received_tools) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/executor/test_executor_agent.py::TestExecutorAgent::test_agent_factory_receives_tools -v`
Expected: FAIL with "unexpected keyword argument 'agent_factory'"

**Step 3: Update ExecutorAgent to accept agent_factory**

修改 `app/agents/executor/executor_agent.py`:

```python
from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent
from app.agents.executor.prompts import EXECUTOR_SYSTEM_PROMPT
from app.task.task_manager import TaskManager
from app.tools.act_toolkit import ActToolkit

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing plan steps.

    Usage:
        # Default: creates LlmAgent internally
        executor = ExecutorAgent(plan_id="p1", model=model)

        # Custom: pass agent_factory that receives tools
        def my_factory(tools: list):
            return LoopAgent(name="executor", tools=tools + my_extra_tools, ...)
        executor = ExecutorAgent(plan_id="p1", agent_factory=my_factory)
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable[[list], Any] = None
    ):
        """
        Initialize ExecutorAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that receives tools and returns an agent
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = ActToolkit(plan)
        tools = list(toolkit.get_tool_functions().values())

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory(tools)
        elif model is not None:
            from google.adk.agents import LlmAgent

            agent = LlmAgent(
                name="executor",
                model=model,
                tools=tools,
                instruction=EXECUTOR_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(
            agent=agent, tool_functions=toolkit.get_tool_functions(), plan_id=plan_id
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

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/executor/test_executor_agent.py::TestExecutorAgent::test_agent_factory_receives_tools -v`
Expected: PASS

**Step 5: Run all ExecutorAgent tests to ensure no regression**

Run: `pytest tests/agents/executor/test_executor_agent.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add app/agents/executor/executor_agent.py tests/agents/executor/test_executor_agent.py
git commit -m "feat(executor): replace agent param with agent_factory for tool injection"
```

---

### Task 3: Update Cortex to use agent_factory

**Files:**
- Modify: `cortex.py`

**Step 1: Update Cortex to accept factory parameters**

修改 `cortex.py`:

```python
import time
from typing import Any, Callable

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        # Default: creates LlmAgent internally
        cortex = Cortex(model=model)

        # Custom: pass agent factories
        def my_planner_factory(tools: list):
            return LoopAgent(name="planner", tools=tools, ...)
        def my_executor_factory(tools: list):
            return LoopAgent(name="executor", tools=tools, ...)
        cortex = Cortex(
            planner_factory=my_planner_factory,
            executor_factory=my_executor_factory
        )
    """

    def __init__(
        self,
        model: Any = None,
        planner_factory: Callable[[list], Any] = None,
        executor_factory: Callable[[list], Any] = None
    ):
        if model is None and planner_factory is None:
            raise ValueError("Either 'model' or 'planner_factory' must be provided")
        if model is None and executor_factory is None:
            raise ValueError("Either 'model' or 'executor_factory' must be provided")

        self.model = model
        self.planner_factory = planner_factory
        self.executor_factory = executor_factory
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
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.planner_factory
            )
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.executor_factory
            )

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

**Step 2: Run all tests to ensure no regression**

Run: `pytest -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add cortex.py
git commit -m "feat(cortex): replace agent params with factory params"
```

---

### Task 4: Update example.py with factory usage example

**Files:**
- Modify: `example.py`

**Step 1: Add commented example of factory usage**

在 `example.py` 加入 factory 用法範例：

```python
"""
Example script to run Cortex.

Usage:
    uv run python example.py
"""

import asyncio
import warnings

# Filter Pydantic warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from cortex import Cortex

# Model configuration
API_BASE_URL = "http://10.136.3.209:8000/v1"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


async def main():
    # Initialize model via LiteLLM
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
        api_key="EMPTY",
    )

    # Create Cortex (default mode)
    cortex = Cortex(model=model)

    # --- Custom agent factory example ---
    # from google.adk.agents import LoopAgent
    #
    # def my_planner_factory(tools: list):
    #     return LoopAgent(
    #         name="planner",
    #         model=model,
    #         tools=tools,  # toolkit tools are injected here
    #     )
    #
    # cortex = Cortex(planner_factory=my_planner_factory)
    # ------------------------------------

    # Execute a task
    query = "寫一篇短篇兒童小說"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add example.py
git commit -m "docs(example): add agent_factory usage example"
```

---

### Task 5: Final verification

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Run example to verify it works**

Run: `uv run python example.py`
Expected: Executes without errors
