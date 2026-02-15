import inspect
from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent, ExecutionContext
from app.agents.executor.prompts import EXECUTOR_SYSTEM_PROMPT
from app.task.task_manager import TaskManager

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing plan steps.

    The executor is "pure" - it executes tasks and returns output.
    Plan state management (marking steps complete/blocked) is handled
    externally by Cortex via the Verifier.

    Usage:
        # Default: creates LlmAgent internally
        executor = ExecutorAgent(plan_id="p1", model=model)

        # Custom: external self-contained agent (no tools injected)
        def my_factory():
            return MyCustomAgent(...)
        executor = ExecutorAgent(plan_id="p1", agent_factory=my_factory)

        # Legacy: factory that receives tools list
        def my_factory(tools: list):
            return LoopAgent(name="executor", tools=tools + my_extra_tools, ...)
        executor = ExecutorAgent(plan_id="p1", agent_factory=my_factory)

        # With sandbox tools:
        executor = ExecutorAgent(plan_id="p1", model=model, extra_tools=[fs_tool, shell_tool])
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable = None,
        extra_tools: list = None,
    ):
        """
        Initialize ExecutorAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that returns an agent.
                           Supports two signatures:
                           - No args: factory() - for external self-contained agents
                           - With tools: factory(tools) - legacy backward compat
            extra_tools: Additional tools (e.g., from sandbox) to include
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        tools = list(extra_tools) if extra_tools else []

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            sig = inspect.signature(agent_factory)
            if len(sig.parameters) == 0:
                agent = agent_factory()
            else:
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

        super().__init__(agent=agent, tool_functions={}, plan_id=plan_id)

    async def execute_step(self, step_index: int, context: str = "") -> str:
        """Execute a specific step"""
        exec_context = ExecutionContext(step_index=step_index)

        step_desc = self.plan.steps[step_index]
        query = f"Execute step {step_index}: {step_desc}"
        if context:
            query += f"\n\nContext: {context}"

        result = await self.execute(query, exec_context=exec_context)
        return result.output
