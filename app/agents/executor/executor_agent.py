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
