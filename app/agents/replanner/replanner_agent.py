"""ReplannerAgent - Redesigns the remaining plan when a step fails."""

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
    """Result from replanning operation.

    Replanner 輸出：
    - failed step 的新 description 和 intent
    - 可選的 continuation steps（local IDs 0, 1, 2...）
    系統負責 map local IDs → actual IDs 並連接到 DAG。
    """

    action: str  # "redesign" | "give_up"
    failed_step_description: str = ""
    failed_step_intent: str = ""
    continuation_steps: dict[int, str] = field(default_factory=dict)
    continuation_dependencies: dict[int, list[int]] = field(default_factory=dict)
    continuation_intents: dict[int, str] = field(default_factory=dict)


class ReplannerAgent(BaseAgent):
    """
    Agent responsible for redesigning the remaining plan when a step fails.

    Usage:
        replanner = ReplannerAgent(plan_id="p1", model=model)
        result = await replanner.replan(
            original_query="Find ZIP codes...",
            failed_step_id=2,
            failed_step_desc="Extract data from table",
            failed_output="I would use python...",
            failed_reason="Executor did not call any tools",
            failed_tool_history=[],
            attempt=1,
            max_attempts=3,
            available_tools=["web_search", "python_executor"],
        )
    """

    def __init__(
        self,
        plan_id: str,
        model: Any | None = None,
        agent_factory: Callable[[list], Any] | None = None,
        available_intents: dict[str, str] | None = None,
    ) -> None:
        self.available_intents = available_intents or {}
        plan = TaskManager.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        self.plan = plan

        if agent_factory is not None:
            agent = agent_factory([])
        elif model is not None:
            llm_agent_class = _get_llm_agent()
            agent = llm_agent_class(
                name="replanner",
                model=model,
                tools=[],
                instruction=REPLANNER_SYSTEM_PROMPT,
            )
        else:
            raise ValueError("Either 'model' or 'agent_factory' must be provided")

        super().__init__(agent=agent, tool_functions={}, plan_id=plan_id)

    def _build_prompt(
        self,
        original_query: str,
        failed_step_id: int,
        failed_step_desc: str,
        failed_output: str,
        failed_reason: str,
        failed_tool_history: list[dict],
        attempt: int,
        max_attempts: int,
        available_tools: list[str],
    ) -> str:
        """Build the prompt for replanning."""
        assert self.plan is not None

        # Gather completed steps info
        completed_steps: dict[int, dict[str, str]] = {}
        for sid in sorted(self.plan.steps.keys()):
            if self.plan.step_statuses.get(sid) == "completed":
                completed_steps[sid] = {
                    "description": self.plan.steps[sid],
                    "deps": str(self.plan.dependencies.get(sid, [])),
                }

        completed_indices = list(completed_steps.keys())
        completed_tool_history = self.plan.format_tool_history(completed_indices)

        # Build failed step info
        failed_lines = [f"### Step {failed_step_id}: {failed_step_desc}"]
        failed_lines.append(f"Attempt: {attempt}/{max_attempts}")
        if failed_reason:
            failed_lines.append(f"\nFailure reason:\n{failed_reason}")
        if failed_output:
            if len(failed_output) > 2000:
                failed_output = "...[truncated]\n" + failed_output[-2000:]
            failed_lines.append(f"\nExecutor output:\n{failed_output}")
        if failed_tool_history:
            failed_lines.append("\nTool calls:")
            for call in failed_tool_history:
                tool = call.get("tool", "?")
                args = call.get("args", {})
                result = call.get("result", "")
                args_str = (
                    ", ".join(f'{k}="{v}"' for k, v in args.items())
                    if isinstance(args, dict)
                    else str(args)
                )
                failed_lines.append(f"- {tool}({args_str}) -> {result}")

        return build_replan_prompt(
            original_query=original_query,
            completed_steps=completed_steps,
            completed_tool_history=completed_tool_history,
            failed_step_info="\n".join(failed_lines),
            failed_step_id=failed_step_id,
            available_tools=available_tools,
            available_intents=self.available_intents,
            attempt=attempt,
            max_attempts=max_attempts,
        )

    def _parse_response(self, response: str) -> ReplanResult | None:
        """Parse the LLM response to extract ReplanResult.

        Returns None on parse failure (caller should retry).
        """
        # Try to extract JSON from code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(
                r'\{(?:[^{}]|\{[^{}]*\})*"action"(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            data = json.loads(json_str)
            action = data.get("action", "give_up")

            if action == "give_up":
                return ReplanResult(action="give_up")

            failed_step_description = data.get("failed_step_description", "")
            failed_step_intent = data.get("failed_step_intent", "default")

            raw_cont = data.get("continuation_steps", {})
            continuation_steps = {int(k): v for k, v in raw_cont.items()} if raw_cont else {}

            raw_cont_deps = data.get("continuation_dependencies", {})
            continuation_dependencies = (
                {int(k): [int(v) for v in vals] for k, vals in raw_cont_deps.items()}
                if raw_cont_deps
                else {}
            )

            raw_cont_intents = data.get("continuation_intents", {})
            continuation_intents = (
                {int(k): v for k, v in raw_cont_intents.items()} if raw_cont_intents else {}
            )

            return ReplanResult(
                action=action,
                failed_step_description=failed_step_description,
                failed_step_intent=failed_step_intent,
                continuation_steps=continuation_steps,
                continuation_dependencies=continuation_dependencies,
                continuation_intents=continuation_intents,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    async def replan(
        self,
        original_query: str,
        failed_step_id: int,
        failed_step_desc: str,
        failed_output: str,
        failed_reason: str,
        failed_tool_history: list[dict],
        attempt: int,
        max_attempts: int,
        available_tools: list[str],
    ) -> ReplanResult:
        """Redesign the remaining plan after a step failure.

        Returns:
            ReplanResult with complete new DAG.
        """
        max_parse_retries = 3
        prompt = self._build_prompt(
            original_query=original_query,
            failed_step_id=failed_step_id,
            failed_step_desc=failed_step_desc,
            failed_output=failed_output,
            failed_reason=failed_reason,
            failed_tool_history=failed_tool_history,
            attempt=attempt,
            max_attempts=max_attempts,
            available_tools=available_tools,
        )

        for attempt_num in range(max_parse_retries):
            result = await self.execute(prompt)
            parsed = self._parse_response(result.output)
            if parsed is not None:
                return parsed
            if attempt_num < max_parse_retries - 1:
                import logging

                logging.getLogger(__name__).warning(
                    "Replanner JSON parse failed (attempt %d/%d), retrying...",
                    attempt_num + 1,
                    max_parse_retries,
                )

        return ReplanResult(action="give_up")
