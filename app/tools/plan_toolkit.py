from google.genai.types import FunctionDeclaration
from app.task.plan import Plan


class PlanToolkit:
    """Planner tools: create and update plans"""

    def __init__(self, plan: Plan):
        self.plan = plan

    # Schema definitions
    CREATE_PLAN_SCHEMA = FunctionDeclaration(
        name="create_plan",
        description="Create a new execution plan. Break down the task into executable steps.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Plan title, briefly describe the goal"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of steps, each should be a concrete executable action"
                },
                "dependencies": {
                    "type": "object",
                    "description": "Step dependencies. Format: {\"1\": [0]} means step 1 depends on step 0"
                }
            },
            "required": ["title", "steps"]
        }
    )

    UPDATE_PLAN_SCHEMA = FunctionDeclaration(
        name="update_plan",
        description="Update the existing plan's title, steps, or dependencies",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "New title (optional)"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New steps list (optional)"
                },
                "dependencies": {
                    "type": "object",
                    "description": "New dependencies (optional)"
                }
            }
        }
    )

    def create_plan(
        self,
        title: str,
        steps: list[str],
        dependencies: dict[int, list[int]] = None
    ) -> str:
        """Create a new plan with title, steps, and optional dependencies"""
        if dependencies is None and len(steps) > 1:
            dependencies = {i: [i - 1] for i in range(1, len(steps))}

        self.plan.update(title=title, steps=steps, dependencies=dependencies)
        return f"Plan created:\n{self.plan.format()}"

    def update_plan(
        self,
        title: str = None,
        steps: list[str] = None,
        dependencies: dict[int, list[int]] = None
    ) -> str:
        """Update existing plan"""
        self.plan.update(title=title, steps=steps, dependencies=dependencies)
        return f"Plan updated:\n{self.plan.format()}"

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Return schema list for LlmAgent"""
        return [self.CREATE_PLAN_SCHEMA, self.UPDATE_PLAN_SCHEMA]

    def get_tool_functions(self) -> dict:
        """Return function mapping for tool execution"""
        return {
            "create_plan": self.create_plan,
            "update_plan": self.update_plan
        }
