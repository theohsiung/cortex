import asyncio
import logging
import time
from typing import Any, Callable, Union

from app.task.task_manager import TaskManager

logger = logging.getLogger(__name__)
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.executor.executor_agent import ExecutorAgent
from app.agents.verifier.verifier import Verifier
from app.agents.replanner.replanner_agent import ReplannerAgent
from app.sandbox.sandbox_manager import SandboxManager


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        # Default: creates LlmAgent internally
        cortex = Cortex(model=model)

        # With sandbox tools:
        cortex = Cortex(
            model=model,
            user_id="alice",          # Optional, auto-generated if not provided
            enable_filesystem=True,   # Local filesystem via @anthropic/mcp-filesystem
            enable_shell=True,        # Shell execution in Docker container
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
        user_id: str = None,
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

        # Create sandbox manager if any sandbox feature is enabled
        if enable_filesystem or enable_shell or mcp_servers:
            self.sandbox = SandboxManager(
                user_id=user_id,
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
            planner_sandbox_tools = (
                self.sandbox.get_planner_tools() if self.sandbox else []
            )
            executor_sandbox_tools = (
                self.sandbox.get_executor_tools() if self.sandbox else []
            )

            # Create plan
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.planner_factory,
                extra_tools=planner_sandbox_tools,
            )
            await planner.create_plan(query)

            # Execute steps with parallel execution
            executor = ExecutorAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=self.executor_factory,
                extra_tools=executor_sandbox_tools,
            )

            # Initialize verifier and replanner
            verifier = Verifier()
            replanner = ReplannerAgent(
                plan_id=plan_id,
                model=self.model,
            )

            # Get available tool names for replanner
            available_tools = [t.__name__ if callable(t) else str(t)
                              for t in executor_sandbox_tools]

            step_outputs: dict[int, str] = {}
            semaphore = asyncio.Semaphore(3)

            async def execute_with_limit(
                step_idx: int,
                max_retries: int = 3,
            ) -> tuple[int, Union[str, Exception]]:
                # Build context from dependency outputs
                deps = plan.dependencies.get(step_idx, [])
                dep_context = "\n".join(
                    f"Step {d} result: {step_outputs[d]}"
                    for d in deps
                    if d in step_outputs
                )
                full_context = f"{query}\n\n{dep_context}" if dep_context else query

                async with semaphore:
                    plan.mark_step(step_idx, step_status="in_progress")
                    last_error = None
                    for attempt in range(max_retries + 1):
                        try:
                            output = await executor.execute_step(
                                step_idx, context=full_context
                            )
                            return step_idx, output
                        except Exception as e:
                            last_error = e
                            if attempt < max_retries:
                                logger.warning(
                                    "Step %d failed (attempt %d/%d): %s. Retrying...",
                                    step_idx,
                                    attempt + 1,
                                    max_retries + 1,
                                    e,
                                )
                                continue
                    return step_idx, last_error

            while True:
                ready_steps = plan.get_ready_steps()
                if not ready_steps:
                    break

                results = await asyncio.gather(
                    *[execute_with_limit(idx) for idx in ready_steps]
                )

                for step_idx, result in results:
                    if isinstance(result, Exception):
                        plan.mark_step(
                            step_idx, step_status="blocked", step_notes=str(result)
                        )
                    else:
                        # Finalize step and verify tool calls
                        plan.finalize_step(step_idx)

                        if verifier.verify_step(plan, step_idx):
                            # Verification passed - mark completed
                            step_outputs[step_idx] = result
                            plan.mark_step(step_idx, step_status="completed")
                        else:
                            # Verification failed - check replan attempts
                            attempts = plan.replan_attempts.get(step_idx, 0)
                            if attempts >= ReplannerAgent.MAX_REPLAN_ATTEMPTS:
                                plan.mark_step(
                                    step_idx,
                                    step_status="blocked",
                                    step_notes="Max replan attempts reached"
                                )
                            else:
                                # Replan the failed step and downstream
                                downstream = plan.get_downstream_steps(step_idx)
                                steps_to_replan = [step_idx] + downstream

                                replan_result = await replanner.replan_subgraph(
                                    steps_to_replan=steps_to_replan,
                                    available_tools=available_tools
                                )
                                plan.replan_attempts[step_idx] = attempts + 1

                                if replan_result.action == "redesign":
                                    # Find last completed step index
                                    completed_indices = [
                                        i for i, step in enumerate(plan.steps)
                                        if plan.step_statuses[step] == "completed"
                                    ]
                                    insert_after = max(completed_indices) if completed_indices else -1

                                    # Update plan DAG
                                    plan.remove_steps(steps_to_replan)
                                    plan.add_steps(
                                        replan_result.new_steps,
                                        replan_result.new_dependencies,
                                        insert_after=insert_after
                                    )
                                else:
                                    # Replanner gave up
                                    plan.mark_step(
                                        step_idx,
                                        step_status="blocked",
                                        step_notes="Replanner gave up"
                                    )

            # Generate final result by aggregating step outputs
            final_result = await self._aggregate_results(query, plan, step_outputs)

            # Record result in history
            self.history.append({"role": "assistant", "content": final_result})

            return final_result

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)
            if self.sandbox:
                await self.sandbox.stop()

    async def _aggregate_results(
        self, query: str, plan: Plan, step_outputs: dict[int, str]
    ) -> str:
        """Aggregate step outputs into a final result using LLM"""
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        # Build context from all step outputs in order
        outputs_text = "\n\n".join(
            f"=== Step {i}: {plan.steps[i]} ===\n{step_outputs.get(i, '[No output]')}"
            for i in range(len(plan.steps))
        )

        aggregation_prompt = f"""Based on the following task and step outputs, synthesize a final coherent result.

Original task: {query}

Step outputs:
{outputs_text}

Reference the outputs of these detailed steps to generate a corresponding complete response to the original task.
Important: Detect the language used in the 'Original task' (e.g., English, Traditional Chinese, etc.) and write your final result in the SAME language.
Do not include meta-commentary about the steps - just provide the final deliverable in the detected language."""

        # Create aggregator agent
        aggregator = LlmAgent(
            name="aggregator",
            model=self.model,
            instruction="You are a content synthesizer. Combine multiple step outputs into a coherent final result.",
        )

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="aggregator", user_id="aggregator"
        )

        runner = Runner(
            agent=aggregator,
            session_service=session_service,
            app_name="aggregator",
        )
        content = Content(parts=[Part(text=aggregation_prompt)])

        final_output = ""
        async for event in runner.run_async(
            user_id="aggregator", session_id=session.id, new_message=content
        ):
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts"):
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_output = part.text

        # Include execution stats and step details
        progress = plan.get_progress()
        return f"""{final_output}

---
Execution: {progress['completed']}/{progress['total']} steps completed.

{plan.format()}"""
