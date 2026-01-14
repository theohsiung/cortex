import time
from typing import Any

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent


class Cortex:
    """Main orchestrator for the agent framework"""

    def __init__(self, model: Any):
        self.model = model
        self.history: list[dict] = []

    async def execute(self, query: str) -> str:
        """Execute a task with planning and execution"""
        # Record user query in history
        self.history.append({"role": "user", "content": query})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        try:
            # Create plan
            planner = PlannerAgent(plan_id=plan_id, model=self.model)
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(plan_id=plan_id, model=self.model)

            while True:
                ready_steps = plan.get_ready_steps()
                if not ready_steps:
                    break

                for step_idx in ready_steps:
                    plan.mark_step(step_idx, step_status="in_progress")
                    await executor.execute_step(step_idx, context=query)

            # Generate summary
            summary = self._generate_summary(plan)

            # Record result in history
            self.history.append({"role": "assistant", "content": summary})

            return summary

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)

    def _generate_summary(self, plan: Plan) -> str:
        """Generate execution summary"""
        progress = plan.get_progress()
        return f"""Task completed.

{plan.format()}

Summary: {progress['completed']}/{progress['total']} steps completed."""
