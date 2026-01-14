import time
from typing import Any

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        # Default: creates LlmAgent internally
        cortex = Cortex(model=model)

        # Custom: pass your own agents
        from google.adk.agents import LoopAgent
        my_planner = LoopAgent(name="planner", ...)
        my_executor = LoopAgent(name="executor", ...)
        cortex = Cortex(planner_agent=my_planner, executor_agent=my_executor)
    """

    def __init__(
        self,
        model: Any = None,
        planner_agent: Any = None,
        executor_agent: Any = None
    ):
        if model is None and planner_agent is None:
            raise ValueError("Either 'model' or 'planner_agent' must be provided")
        if model is None and executor_agent is None:
            raise ValueError("Either 'model' or 'executor_agent' must be provided")

        self.model = model
        self.planner_agent = planner_agent
        self.executor_agent = executor_agent
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
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent=self.planner_agent
            )
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent=self.executor_agent
            )

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
