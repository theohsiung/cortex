"""Planner agent for creating and updating plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from app.agents.base.base_agent import BaseAgent
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT, build_intent_prompt_section
from app.task.task_manager import TaskManager
from app.tools.plan_toolkit import PlanToolkit

if TYPE_CHECKING:
    pass


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

        # With sandbox tools:
        planner = PlannerAgent(plan_id="p1", model=model, extra_tools=[fs_tool])
    """

    def __init__(
        self,
        plan_id: str,
        model: Any | None = None,
        agent_factory: Callable[[list], Any] | None = None,
        extra_tools: list | None = None,
        available_intents: dict[str, str] | None = None,
    ) -> None:
        """Initialize PlannerAgent.

        Args:
            plan_id: ID of the plan in TaskManager.
            model: LLM model (required if agent_factory is None).
            agent_factory: Optional factory function that receives tools and returns an agent.
            extra_tools: Additional tools (e.g., from sandbox) to include.
            available_intents: Dict mapping intent names to descriptions for prompt injection.
        """
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        # Detect if model supports aliased tool names (Gemini doesn't)
        include_aliases = self.should_include_aliases(model)

        toolkit = PlanToolkit(plan)
        tools = toolkit.get_tool_functions(include_aliases=include_aliases)

        # Add extra tools (e.g., sandbox tools)
        if extra_tools:
            tools.extend(extra_tools)

        # Build instruction with optional intent section
        instruction = PLANNER_SYSTEM_PROMPT
        if available_intents and len(available_intents) > 1:
            intent_section = build_intent_prompt_section(available_intents)
            instruction = instruction + "\n\n" + intent_section

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory(tools)
        elif model is not None:
            from google.adk.agents import LlmAgent

            agent = LlmAgent(
                name="planner",
                model=model,
                tools=tools,
                instruction=instruction,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        # Convert tools list to dict for BaseAgent (only core tools, not aliases)
        tool_functions = {
            "create_plan": toolkit.create_plan,
            "update_plan": toolkit.update_plan,
        }
        super().__init__(agent=agent, tool_functions=tool_functions, plan_id=plan_id)

    async def create_plan(self, task: str) -> str:
        """Create a plan for the given task."""
        result = await self.execute(f"Create a plan for: {task}")
        return result.output
