"""Planner tools for creating and updating execution plans."""

from __future__ import annotations

import functools
import logging

from google.genai.types import FunctionDeclaration

from app.task.plan import Plan

logger = logging.getLogger(__name__)


class PlanToolkit:
    """Planner tools: create and update plans."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    # Schema definitions
    CREATE_PLAN_SCHEMA = FunctionDeclaration(
        name="create_plan",
        description="Create a new execution plan. Break down the task into executable steps.",
        parameters={  # type: ignore[arg-type]
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Plan title, briefly describe the goal",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of steps, each should be a concrete executable action",
                },
                "dependencies": {
                    "type": "object",
                    "description": 'Step dependencies as DAG. Format: {"step_index": [list of prerequisite step indices]}. Example: {"0": [], "1": [0], "2": [0], "3": [1, 2]} means step 0 has no deps, step 1 and 2 depend on step 0, step 3 depends on both 1 and 2. Every step index must be included as a key.',
                },
                "intents": {
                    "type": "object",
                    "description": 'Map of step index to executor intent type. Format: {"0": "generate", "1": "review", "2": "default"}. Use "default" for general tasks.',
                },
            },
            "required": ["title", "steps", "dependencies"],
        },
    )

    UPDATE_PLAN_SCHEMA = FunctionDeclaration(
        name="update_plan",
        description="Update the existing plan's title, steps, or dependencies",
        parameters={  # type: ignore[arg-type]
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "New title (optional)"},
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New steps list (optional)",
                },
                "dependencies": {
                    "type": "object",
                    "description": 'New dependencies as DAG. Format: {"step_index": [list of prerequisite step indices]}. Every step index must be included as a key.',
                },
                "intents": {
                    "type": "object",
                    "description": 'Map of step index to executor intent type. Format: {"0": "generate", "1": "review", "2": "default"}. Use "default" for general tasks.',
                },
            },
        },
    )

    @staticmethod
    def _normalize_intents(intents: dict) -> dict[int, str]:
        """Normalize intents dict to ensure all keys are integers.

        JSON parsing from LLM converts dict keys to strings, so we convert them back.
        """
        if not intents:
            return {}
        return {int(k): v for k, v in intents.items()}

    def create_plan(
        self,
        title: str,
        steps: list[str],
        dependencies: dict[int, list[int]] | None = None,
        intents: dict[int, str] | None = None,
    ) -> str:
        """Create a new plan with title, steps, optional dependencies and intents."""
        fallback_used = False
        if dependencies is None and len(steps) > 1:
            dependencies = {i: [i - 1] for i in range(1, len(steps))}
            fallback_used = True
            logger.warning(
                "No dependencies provided. Using sequential fallback: %s",
                dependencies,
            )

        normalized_intents = self._normalize_intents(intents) if intents else None
        self.plan.update(
            title=title, steps=steps, dependencies=dependencies, step_intents=normalized_intents
        )
        result = f"Plan created:\n{self.plan.format()}"
        if fallback_used:
            result += "\n\nNote: Dependencies were auto-generated as sequential. Consider providing explicit dependencies for parallel execution."
        return result

    def update_plan(
        self,
        title: str | None = None,
        steps: list[str] | None = None,
        dependencies: dict[int, list[int]] | None = None,
        intents: dict[int, str] | None = None,
    ) -> str:
        """Update existing plan."""
        normalized_intents = self._normalize_intents(intents) if intents else None
        self.plan.update(
            title=title, steps=steps, dependencies=dependencies, step_intents=normalized_intents
        )
        return f"Plan updated:\n{self.plan.format()}"

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Return schema list for LlmAgent."""
        return [self.CREATE_PLAN_SCHEMA, self.UPDATE_PLAN_SCHEMA]

    def get_tool_functions(self, include_aliases: bool = False) -> list:
        """Return list of tool functions for LlmAgent.

        Args:
            include_aliases: If True, include aliased versions for handling
                             LLM hallucinated tool names (for gpt-oss models).
                             Gemini doesn't support special chars in function names.
        """
        tools = [self.create_plan, self.update_plan]

        if include_aliases:
            # Add aliased tools for common hallucinated suffixes (gpt-oss specific)
            hallucinated_suffixes = ["<|channel|>json", "<|end|>", "<|tool|>", "json"]
            for suffix in hallucinated_suffixes:
                tools.append(self._create_aliased_tool(self.create_plan, f"create_plan{suffix}"))
                tools.append(self._create_aliased_tool(self.update_plan, f"update_plan{suffix}"))

        return tools

    def _create_aliased_tool(self, func, alias_name: str):
        """Create a wrapper function with a custom __name__ for ADK registration."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = alias_name
        return wrapper
