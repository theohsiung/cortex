"""ReplannerAgent - Redesigns failed steps and their downstream dependencies."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.base.base_agent import BaseAgent
from app.agents.replanner.prompts import REPLANNER_SYSTEM_PROMPT, build_replan_prompt
from app.task.task_manager import TaskManager


@functools.lru_cache(maxsize=1)
def _get_llm_agent():
    """Lazy import LlmAgent to avoid circular imports."""
    from google.adk.agents import LlmAgent

    return LlmAgent


@dataclass
class ReplanResult:
    """Result from replanning operation."""

    action: str  # "redesign" | "give_up"
    new_steps: list[str]
    new_dependencies: dict[int, list[int]]
    new_intents: dict[int, str] = field(default_factory=dict)


class ReplannerAgent(BaseAgent):
    """
    Agent responsible for redesigning failed steps and their downstream dependencies.

    Usage:
        replanner = ReplannerAgent(plan_id="p1", model=model)
        result = await replanner.replan_subgraph(
            steps_to_replan=[5, 6, 7],
            available_tools=["write_file", "run_command"]
        )
    """

    MAX_REPLAN_ATTEMPTS = 2

    def __init__(
        self,
        plan_id: str,
        model: Any | None = None,
        agent_factory: Callable[[list], Any] | None = None,
        available_intents: dict[str, str] | None = None,
    ) -> None:
        """Initialize ReplannerAgent.

        Args:
            plan_id: ID of the plan in TaskManager.
            model: LLM model (required if agent_factory is None).
            agent_factory: Optional factory function that returns an agent.
            available_intents: Dict of intent_name -> description for routing.
        """
        self.available_intents = available_intents or {}
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        self.plan = plan

        # Use factory or create default LlmAgent
        if agent_factory is not None:
            agent = agent_factory([])
        elif model is not None:
            llm_agent_class = _get_llm_agent()
            agent = llm_agent_class(
                name="replanner",
                model=model,
                tools=[],  # Replanner doesn't need tools, just outputs JSON
                instruction=REPLANNER_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(agent=agent, tool_functions={}, plan_id=plan_id)

    def _build_replan_prompt(self, steps_to_replan: list[int], available_tools: list[str]) -> str:
        """
        Build the prompt for replanning.

        Args:
            steps_to_replan: List of step indices to redesign
            available_tools: List of available tool names

        Returns:
            Complete prompt for the replanner
        """
        # Get completed steps
        completed_indices = [
            i
            for i, step in enumerate(self.plan.steps)
            if self.plan.step_statuses[step] == "completed"
        ]

        # Format completed steps tool history
        completed_tool_history = self.plan.format_tool_history(completed_indices)

        # Get step descriptions for steps to replan
        steps_with_desc = [
            (idx, self.plan.steps[idx]) for idx in steps_to_replan if idx < len(self.plan.steps)
        ]

        return build_replan_prompt(
            completed_tool_history=completed_tool_history,
            steps_to_replan=steps_with_desc,
            available_tools=available_tools,
            available_intents=self.available_intents,
        )

    def _parse_replan_response(self, response: str) -> ReplanResult:
        """
        Parse the LLM response to extract ReplanResult.

        Args:
            response: Raw LLM response text

        Returns:
            ReplanResult extracted from response
        """
        # Try to extract JSON from code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Could not find valid JSON
                return ReplanResult(action="give_up", new_steps=[], new_dependencies={})

        try:
            data = json.loads(json_str)
            action = data.get("action", "give_up")
            new_steps = data.get("new_steps", [])
            raw_deps = data.get("new_dependencies", {})

            # Convert string keys to int (JSON doesn't support int keys)
            new_dependencies = {int(k): v for k, v in raw_deps.items()}

            raw_intents = data.get("new_intents", {})
            new_intents = {int(k): v for k, v in raw_intents.items()} if raw_intents else {}

            return ReplanResult(
                action=action,
                new_steps=new_steps,
                new_dependencies=new_dependencies,
                new_intents=new_intents,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return ReplanResult(action="give_up", new_steps=[], new_dependencies={})

    async def replan_subgraph(
        self, steps_to_replan: list[int], available_tools: list[str]
    ) -> ReplanResult:
        """
        Redesign the specified steps.

        Args:
            steps_to_replan: List of step indices to redesign
            available_tools: List of available tool names

        Returns:
            ReplanResult with new steps and dependencies
        """
        prompt = self._build_replan_prompt(steps_to_replan, available_tools)
        result = await self.execute(prompt)
        return self._parse_replan_response(result.output)
