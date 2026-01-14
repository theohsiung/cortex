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
