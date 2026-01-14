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

    def get_tool_functions(self) -> dict:
        """Return function mapping for tool execution"""
        return {
            "mark_step": self.mark_step
        }
