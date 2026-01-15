import time
from typing import Any, Callable

from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent
from app.sandbox.sandbox_manager import SandboxManager


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        # Default: creates LlmAgent internally
        cortex = Cortex(model=model)

        # With sandbox (Docker-isolated tools):
        cortex = Cortex(
            model=model,
            workspace="./my-project",
            enable_filesystem=True,
            enable_shell=True,
        )

        # Custom: pass agent factories
        def my_planner_factory(tools: list):
            return LoopAgent(name="planner", tools=tools, ...)
        def my_executor_factory(tools: list):
            return LoopAgent(name="executor", tools=tools, ...)
        cortex = Cortex(
            planner_factory=my_planner_factory,
            executor_factory=my_executor_factory
        )
    """

    def __init__(
        self,
        model: Any = None,
        planner_factory: Callable[[list], Any] = None,
        executor_factory: Callable[[list], Any] = None,
        workspace: str = None,
        enable_filesystem: bool = False,
        enable_shell: bool = False,
        mcp_servers: list[dict] = None,
    ):
        if model is None and planner_factory is None:
            raise ValueError("Either 'model' or 'planner_factory' must be provided")
        if model is None and executor_factory is None:
            raise ValueError("Either 'model' or 'executor_factory' must be provided")

        self.model = model
        self.planner_factory = planner_factory
        self.executor_factory = executor_factory
        self.history: list[dict] = []

        # Create sandbox manager if workspace provided
        if workspace:
            self.sandbox = SandboxManager(
                workspace=workspace,
                enable_filesystem=enable_filesystem,
                enable_shell=enable_shell,
                mcp_servers=mcp_servers,
            )
        else:
            self.sandbox = None

    async def execute(self, query: str) -> str:
        """Execute a task with planning and execution"""
        # Record user query in history
        self.history.append({"role": "user", "content": query})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        try:
            # Start sandbox if configured
            if self.sandbox:
                await self.sandbox.start()

            # Get sandbox tools
            planner_sandbox_tools = self.sandbox.get_planner_tools() if self.sandbox else []
            executor_sandbox_tools = self.sandbox.get_executor_tools() if self.sandbox else []

            # Create plan
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.planner_factory,
                extra_tools=planner_sandbox_tools,
            )
            await planner.create_plan(query)

            # Execute steps
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.executor_factory,
                extra_tools=executor_sandbox_tools,
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
            if self.sandbox:
                await self.sandbox.stop()

    def _generate_summary(self, plan: Plan) -> str:
        """Generate execution summary"""
        progress = plan.get_progress()
        return f"""Task completed.

{plan.format()}

Summary: {progress['completed']}/{progress['total']} steps completed."""
