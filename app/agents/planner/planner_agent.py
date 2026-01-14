from typing import Any, TYPE_CHECKING
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

        # Custom: pass your own agent
        from google.adk.agents import LoopAgent
        my_agent = LoopAgent(name="planner", ...)
        planner = PlannerAgent(plan_id="p1", agent=my_agent)
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent: Any = None
    ):
        """
        Initialize PlannerAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent is None)
            agent: Optional pre-built ADK agent (LlmAgent, LoopAgent, etc.)
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        toolkit = PlanToolkit(plan)

        # Use provided agent or create default LlmAgent
        if agent is None:
            if model is None:
                raise ValueError("Either 'model' or 'agent' must be provided")
            from google.adk.agents import LlmAgent
            agent = LlmAgent(
                name="planner",
                model=model,
                tools=toolkit.get_tool_declarations(),
                instruction=PLANNER_SYSTEM_PROMPT
            )

        super().__init__(
            agent=agent,
            tool_functions=toolkit.get_tool_functions(),
            plan_id=plan_id
        )

    async def create_plan(self, task: str) -> str:
        """Create a plan for the given task"""
        result = await self.execute(f"Create a plan for: {task}")
        return result.output
