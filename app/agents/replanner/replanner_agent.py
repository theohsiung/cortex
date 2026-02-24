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
class RetryStepInfo:
    """Info for retrying the failed step with a new approach."""

    description: str
    intent: str


@dataclass
class ReplanResult:
    """Result from replanning operation."""

    action: str  # "redesign" | "give_up"
    new_steps: dict[int, str]  # {3: "step A", 4: "step B"}
    new_dependencies: dict[int, list[int]]
    new_intents: dict[int, str] = field(default_factory=dict)
    retry_step: RetryStepInfo | None = None


@dataclass
class ReplanContext:
    """Context about the failure, passed from Cortex to the replanner."""

    original_query: str
    failed_step_notes: dict[int, str]
    failed_step_outputs: dict[int, str]
    failed_tool_history: dict[int, list[dict]]
    attempt_number: int
    max_attempts: int


class ReplannerAgent(BaseAgent):
    """
    Agent responsible for redesigning failed steps and their downstream dependencies.

    Usage:
        replanner = ReplannerAgent(plan_id="p1", model=model)
        result = await replanner.replan_subgraph(
            failed_step_idx=5,
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

    def _build_replan_prompt(
        self,
        failed_step_idx: int,
        available_tools: list[str],
        context: ReplanContext | None = None,
    ) -> str:
        """
        Build the prompt for replanning.

        Args:
            failed_step_idx: Index of the single failed step
            available_tools: List of available tool names
            context: Optional failure context for richer prompts

        Returns:
            Complete prompt for the replanner
        """
        assert self.plan is not None

        completed_indices = [
            i for i in self.plan.steps if self.plan.step_statuses[i] == "completed"
        ]
        completed_dag = self.plan.format_completed_dag()
        completed_tool_history = self.plan.format_tool_history(completed_indices)

        # Build failed step info (single step only)
        failed_lines = []
        desc = self.plan.steps.get(failed_step_idx, f"Step {failed_step_idx}")
        failed_lines.append(f"### Step {failed_step_idx}: {desc}")

        if context:
            failed_lines.append(f"Attempt: {context.attempt_number}/{context.max_attempts}")

            notes = context.failed_step_notes.get(failed_step_idx, "")
            if notes:
                failed_lines.append(f"\nFailure reason:\n{notes}")

            output = context.failed_step_outputs.get(failed_step_idx, "")
            if output:
                if len(output) > 500:
                    output = "...[truncated]\n" + output[-500:]
                failed_lines.append(f"\nExecutor output (last 500 chars):\n{output}")

            tool_history = context.failed_tool_history.get(failed_step_idx, [])
            if tool_history:
                failed_lines.append("\nTool calls:")
                for call in tool_history:
                    tool = call.get("tool", "?")
                    args = call.get("args", {})
                    result = call.get("result", "")
                    args_str = (
                        ", ".join(f'{k}="{v}"' for k, v in args.items())
                        if isinstance(args, dict)
                        else str(args)
                    )
                    failed_lines.append(f"- {tool}({args_str}) -> {result}")

        failed_step_info = "\n".join(failed_lines)
        next_id = self.plan._next_id

        return build_replan_prompt(
            completed_dag=completed_dag,
            completed_tool_history=completed_tool_history,
            failed_step_info=failed_step_info,
            failed_step_id=failed_step_idx,
            next_id=next_id,
            available_tools=available_tools,
            available_intents=self.available_intents,
            context=context,
        )

    def _parse_replan_response(self, response: str) -> ReplanResult | None:
        """
        Parse the LLM response to extract ReplanResult.

        Args:
            response: Raw LLM response text

        Returns:
            ReplanResult extracted from response, or None if parsing failed
            (caller should retry on None; give_up is an intentional LLM decision)
        """
        # Try to extract JSON from code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object (supports nested braces)
            json_match = re.search(
                r'\{(?:[^{}]|\{[^{}]*\})*"action"(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                # Could not find valid JSON - parse failure, caller should retry
                return None

        try:
            data = json.loads(json_str)
            action = data.get("action", "give_up")

            raw_steps = data.get("new_steps", {})
            # Handle both dict and list formats
            if isinstance(raw_steps, list):
                # Legacy list format: convert to dict using plan's _next_id
                next_id = self.plan._next_id if self.plan else 0
                new_steps = {next_id + i: step for i, step in enumerate(raw_steps)}
            elif isinstance(raw_steps, dict):
                new_steps = {int(k): v for k, v in raw_steps.items()}
            else:
                new_steps = {}

            raw_deps = data.get("new_dependencies", {})
            new_dependencies = {int(k): [int(v) for v in vals] for k, vals in raw_deps.items()}

            raw_intents = data.get("new_intents", {})
            new_intents = {int(k): v for k, v in raw_intents.items()} if raw_intents else {}

            # Parse retry_step
            raw_retry = data.get("retry_step")
            retry_step = None
            if raw_retry and isinstance(raw_retry, dict):
                retry_step = RetryStepInfo(
                    description=raw_retry.get("description", ""),
                    intent=raw_retry.get("intent", "default"),
                )

            return ReplanResult(
                action=action,
                new_steps=new_steps,
                new_dependencies=new_dependencies,
                new_intents=new_intents,
                retry_step=retry_step,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            # Parse failure - caller should retry
            return None

    async def replan_subgraph(
        self,
        failed_step_idx: int,
        available_tools: list[str],
        context: ReplanContext | None = None,
    ) -> ReplanResult:
        """
        Redesign the specified failed step.

        Args:
            failed_step_idx: Index of the single failed step
            available_tools: List of available tool names
            context: Optional failure context for richer prompts

        Returns:
            ReplanResult with new steps and dependencies
        """
        max_parse_retries = 3
        prompt = self._build_replan_prompt(failed_step_idx, available_tools, context=context)

        for attempt in range(max_parse_retries):
            result = await self.execute(prompt)
            parsed = self._parse_replan_response(result.output)
            if parsed is not None:
                return parsed
            if attempt < max_parse_retries - 1:
                import logging

                logging.getLogger(__name__).warning(
                    "Replanner JSON parse failed (attempt %d/%d), retrying...",
                    attempt + 1,
                    max_parse_retries,
                )

        return ReplanResult(action="give_up", new_steps={}, new_dependencies={})
