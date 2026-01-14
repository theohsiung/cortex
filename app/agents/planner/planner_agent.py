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
