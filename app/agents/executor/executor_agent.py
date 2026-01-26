from typing import Any, Callable, TYPE_CHECKING
from app.agents.base.base_agent import BaseAgent, ExecutionContext
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

        # With sandbox tools:
        executor = ExecutorAgent(plan_id="p1", model=model, extra_tools=[fs_tool, shell_tool])
    """

    def __init__(
        self,
        plan_id: str,
        model: Any = None,
        agent_factory: Callable[[list], Any] = None,
        extra_tools: list = None,
    ):
        """
        Initialize ExecutorAgent.

        Args:
            plan_id: ID of the plan in TaskManager
            model: LLM model (required if agent_factory is None)
            agent_factory: Optional factory function that receives tools and returns an agent
            extra_tools: Additional tools (e.g., from sandbox) to include
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        # Detect if model supports aliased tool names (Gemini doesn't)
        include_aliases = self._should_include_aliases(model)

        toolkit = ActToolkit(plan)
        tools = toolkit.get_tool_functions(include_aliases=include_aliases)

        # Add extra tools (e.g., sandbox tools)
        if extra_tools:
            tools.extend(extra_tools)

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

        # Convert tools list to dict for BaseAgent (only core tools, not aliases)
        tool_functions = {
            "mark_step": toolkit.mark_step,
        }
        super().__init__(agent=agent, tool_functions=tool_functions, plan_id=plan_id)

    @staticmethod
    def _should_include_aliases(model: Any) -> bool:
        """Check if model supports aliased tool names with special characters.

        Gemini API doesn't support special chars like <|channel|> in function names.
        gpt-oss models may hallucinate these suffixes and need aliases.
        """
        if model is None:
            return False
        model_str = str(model).lower()
        # Gemini doesn't support special chars in function names
        if "gemini" in model_str:
            return False
        # gpt-oss models may need aliases for hallucinated tool names
        if "gpt-oss" in model_str or "openai" in model_str:
            return True
        return False

    async def execute_step(self, step_index: int, context: str = "") -> str:
        """Execute a specific step"""
        exec_context = ExecutionContext(step_index=step_index)

        step_desc = self.plan.steps[step_index]
        query = f"Execute step {step_index}: {step_desc}"
        if context:
            query += f"\n\nContext: {context}"

        result = await self.execute(query, exec_context=exec_context)
        return result.output
