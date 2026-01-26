import functools
from google.genai.types import FunctionDeclaration
from app.task.plan import Plan


class ActToolkit:
    """Executor tools: mark step progress"""

    def __init__(self, plan: Plan):
        self.plan = plan

    MARK_STEP_SCHEMA = FunctionDeclaration(
        name="mark_step",
        description="Mark a plan step's status. Use this to track progress during execution.",
        parameters={
            "type": "object",
            "properties": {
                "step_index": {
                    "type": "integer",
                    "description": "Index of the step to mark (0-based)"
                },
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "blocked"],
                    "description": "New status for the step"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the step (e.g., completion details, blockers)"
                }
            },
            "required": ["step_index", "status"]
        }
    )

    def mark_step(
        self,
        step_index: int,
        status: str,
        notes: str = None
    ) -> str:
        """Mark a step's status and optionally add notes"""
        if step_index < 0 or step_index >= len(self.plan.steps):
            raise ValueError(f"Invalid step index: {step_index}")

        self.plan.mark_step(step_index, status, notes)
        return f"Step {step_index} marked as {status}"

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Return schema list for LlmAgent"""
        return [self.MARK_STEP_SCHEMA]

    def get_tool_functions(self, include_aliases: bool = False) -> list:
        """Return list of tool functions for LlmAgent.

        Args:
            include_aliases: If True, include aliased versions for handling
                             LLM hallucinated tool names (for gpt-oss models).
                             Gemini doesn't support special chars in function names.
        """
        tools = [self.mark_step]

        if include_aliases:
            # Add aliased tools for common hallucinated suffixes (gpt-oss specific)
            hallucinated_suffixes = [
                "<|channel|>json",
                "<|channel|>commentary",
                "<|end|>",
                "<|tool|>",
            ]
            for suffix in hallucinated_suffixes:
                tools.append(self._create_aliased_tool(self.mark_step, f"mark_step{suffix}"))

        return tools

    def _create_aliased_tool(self, func, alias_name: str):
        """Create a wrapper function with a custom __name__ for ADK registration."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = alias_name
        return wrapper
