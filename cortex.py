"""Main Cortex orchestrator for multi-step task execution."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Union

from app.agents.executor.executor_agent import ExecutorAgent
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.replanner.replanner_agent import FailureRecord, ReplannerAgent
from app.agents.verifier.verifier import Verifier
from app.config import CortexConfig
from app.sandbox.sandbox_manager import SandboxManager
from app.task.plan import Plan
from app.task.task_manager import TaskManager

logger = logging.getLogger(__name__)


class Cortex:
    """
    Main orchestrator for the agent framework.

    Usage:
        from app.config import CortexConfig, ModelConfig, SandboxConfig

        # Minimal:
        config = CortexConfig(model=ModelConfig(name="openai/gpt-4o", api_base="http://localhost/v1"))
        cortex = Cortex(config)

        # With sandbox tools:
        config = CortexConfig(
            model=ModelConfig(name="openai/gpt-4o", api_base="http://localhost/v1"),
            sandbox=SandboxConfig(enable_filesystem=True, enable_shell=True, user_id="alice"),
        )
        cortex = Cortex(config)

        result = await cortex.execute("Build a REST API")
    """

    def __init__(self, config: CortexConfig) -> None:
        self.config = config
        self._model = None  # Lazy-created LLM
        self._executor_entries = {e.intent: e for e in config.executors}

        # Create sandbox manager if any sandbox feature is enabled
        sandbox_needed = (
            config.sandbox.enable_filesystem or config.sandbox.enable_shell or config.mcp_servers
        )
        self.sandbox: SandboxManager | None = None
        if sandbox_needed:
            self.sandbox = SandboxManager(config.sandbox, mcp_servers=config.mcp_servers)

    @property
    def model(self):
        """Lazy-create LLM model from config."""
        if self._model is None:
            from google.adk.models import LiteLlm

            self._model = LiteLlm(
                model=self.config.model.name,
                api_base=self.config.model.api_base,
                api_key=self.config.model.resolve_api_key(),
            )
        return self._model

    def _get_executor_factory(self, intent: str):
        """Get executor factory for an intent via importlib, or None.

        找不到對應 intent 時 fallback 到第一個設定的 executor
        (例如 planner 給 "default" 但 config 只有 "general")。
        """
        entry = self._executor_entries.get(intent)
        if entry:
            return entry.get_factory()
        if self.config.executors:
            fallback = self.config.executors[0]
            logger.info("Intent %r 找不到，fallback 到 %r", intent, fallback.intent)
            return fallback.get_factory()
        return None

    def _get_available_intents(self) -> dict[str, str]:
        """Build available intents dict from executors + default."""
        # intents = {"default": "General purpose tasks"}
        intents = {}  # for AS testing senerio
        for entry in self.config.executors:
            if entry.tool_names:
                tools_str = ", ".join(entry.tool_names)
                intents[entry.intent] = f"{entry.description} [tools: {tools_str}]"
            else:
                intents[entry.intent] = entry.description
        return intents

    async def execute(
        self,
        query: str,
        history: list[dict] | None = None,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> str:
        """Execute a task with planning and execution.

        Args:
            query: The user's goal/task.
            on_event: Optional callback for streaming events.
                Signature: on_event(event_type: str, data: dict).
        """

        async def emit(event_type: str, data: dict | None = None) -> None:
            if on_event:
                if asyncio.iscoroutinefunction(on_event):
                    await on_event(event_type, data or {})
                else:
                    on_event(event_type, data or {})

        # Create new plan for this task
        plan_id = f"plan_{int(time.time())}"
        plan = Plan()
        TaskManager.set_plan(plan_id, plan)

        await emit("plan_created", {"plan_id": plan_id})

        try:
            # Start sandbox if configured
            if self.sandbox:
                logger.info("Starting sandbox...")
                await self.sandbox.start()

            # Get sandbox tools
            planner_sandbox_tools = self.sandbox.get_planner_tools() if self.sandbox else []
            executor_sandbox_tools = self.sandbox.get_executor_tools() if self.sandbox else []

            # Create plan
            logger.info("=== PLANNING PHASE ===")
            logger.info(
                "Creating plan for query: %s",
                query[:100] + "..." if len(query) > 100 else query,
            )
            planner = PlannerAgent(
                plan_id=plan_id,
                model=self.model,
                agent_factory=None,
                extra_tools=planner_sandbox_tools,
                available_intents=self._get_available_intents(),
            )
            # Build planner query with history context
            if history:
                recent = history[-20:]  # last 3 rounds
                cleaned = []
                for m in recent:
                    content = m["content"]
                    if m["role"] == "assistant" and "---" in content:
                        content = content.split("---")[0].strip()
                    cleaned.append(f"{m['role'].capitalize()}: {content}")
                history_text = "\n".join(cleaned)
                planner_query = (
                    f"## Past conversation (for reference)\n{history_text}\n\n"
                    f"## Current task\n{query}"
                )
            else:
                planner_query = query

            logger.info("\033[95m*** HISTORY (cleaned) ***:\n%s\033[0m", planner_query)
            await planner.create_plan(planner_query)
            logger.info("Plan created with %d steps", len(plan.steps))

            # Emit initial plan structure
            await emit(
                "plan_updated",
                {
                    "plan": plan.format_dag(),
                    "steps": plan.steps,
                    "dependencies": plan.dependencies,
                    "step_intents": plan.step_intents,
                },
            )

            for i, step in plan.steps.items():
                logger.info("  Step %d: %s", i, step)

            # Initialize verifier and replanner
            verifier = Verifier(model=self.model)
            replanner = ReplannerAgent(
                plan_id=plan_id,
                model=self.model,
                available_intents=self._get_available_intents(),
            )

            # Get available tool names for replanner (sandbox tools + all executor tools)
            available_tools = [
                t.__name__ if callable(t) else str(t) for t in executor_sandbox_tools
            ]
            for entry in self.config.executors:
                available_tools.extend(entry.tool_names)

            logger.info("=== EXECUTION PHASE ===")
            step_outputs: dict[int, str] = {}
            failed_outputs: dict[int, str] = {}
            failure_history: list[FailureRecord] = []
            semaphore = asyncio.Semaphore(self.config.tuning.max_concurrent_steps)

            async def execute_with_limit(
                step_idx: int,
            ) -> tuple[int, Union[str, Exception]]:
                max_retries = self.config.tuning.max_retries
                # Build context from dependency outputs
                deps = plan.dependencies.get(step_idx, [])
                dep_context = "\n".join(
                    f"Step {d} result: {step_outputs[d]}" for d in deps if d in step_outputs
                )
                full_context = f"{query}\n\n{dep_context}" if dep_context else query

                async with semaphore:
                    step_desc = plan.steps.get(step_idx, f"Step {step_idx}")
                    logger.info("▶ Executing step %d: %s", step_idx, step_desc)
                    plan.mark_step(step_idx, step_status="in_progress")
                    await emit(
                        "step_status",
                        {
                            "step_idx": step_idx,
                            "status": "in_progress",
                            "intent": plan.get_step_intent(step_idx),
                        },
                    )

                    last_error: Exception = Exception("step did not execute")
                    for attempt in range(max_retries + 1):
                        try:
                            # Dispatch to correct executor based on step intent
                            intent = plan.get_step_intent(step_idx)
                            factory = self._get_executor_factory(intent)
                            if factory:
                                # External executor from executors config
                                from app.agents.base.base_agent import (
                                    BaseAgent as _BaseAgent,
                                )
                                from app.agents.base.base_agent import ExecutionContext

                                agent = factory()
                                executor = _BaseAgent(agent=agent, plan_id=plan_id)
                                exec_context = ExecutionContext(step_index=step_idx)
                                # Only allow submit_final_answer if this step explicitly requires it
                                submit_note = (
                                    ""
                                    if any(
                                        kw in step_desc.lower() for kw in ["submit", "final answer"]
                                    )
                                    else "\nDo NOT call submit_final_answer — it is reserved for the final submission step only.\n"
                                )
                                query_text = (
                                    f"Execute ONLY this step (do not attempt subsequent steps or solve the full task):{submit_note}\n"
                                    f"Step {step_idx}: {step_desc}\n\n"
                                    f"Overall task context (for reference only):\n{full_context}"
                                )
                                result = await executor.execute(
                                    query_text, exec_context=exec_context
                                )
                                # Nudge retry if 0 tool calls
                                tool_history = plan.step_tool_history.get(step_idx, [])
                                if not tool_history:
                                    logger.warning(
                                        "Step %d had 0 tool calls, retrying with nudge",
                                        step_idx,
                                    )
                                    nudge = (
                                        f"Your previous response for step {step_idx} contained NO tool calls. "
                                        f"You only described what should be done instead of actually doing it.\n\n"
                                        f"You MUST call the appropriate tool(s) now to complete this step:\n"
                                        f"{step_desc}\n\n"
                                        f"Call the tool directly. Do NOT explain — just execute."
                                    )
                                    exec_context_retry = ExecutionContext(step_index=step_idx)
                                    result = await executor.execute(
                                        nudge, exec_context=exec_context_retry
                                    )
                                output = result.output
                            else:
                                # Default internal executor
                                executor = ExecutorAgent(
                                    plan_id=plan_id,
                                    model=self.model,
                                    extra_tools=executor_sandbox_tools,
                                )
                                output = await executor.execute_step(step_idx, context=full_context)
                            logger.info("✓ Step %d completed", step_idx)
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

                results = await asyncio.gather(*[execute_with_limit(idx) for idx in ready_steps])

                for step_idx, result in results:
                    if isinstance(result, Exception):
                        logger.error("✗ Step %d failed with exception: %s", step_idx, result)
                        plan.mark_step(step_idx, step_status="blocked", step_notes=str(result))
                        await emit(
                            "step_status",
                            {
                                "step_idx": step_idx,
                                "status": "blocked",
                                "error": str(result),
                            },
                        )
                    else:
                        # Finalize step and verify tool calls
                        plan.finalize_step(step_idx)

                        verify_result = verifier.verify_step(plan, step_idx)

                        # LLM evaluation for all steps that passed mechanical check
                        if verify_result.passed:
                            step_desc = plan.steps.get(step_idx, f"Step {step_idx}")
                            tool_calls = plan.step_tool_history.get(step_idx, [])
                            completed_calls = sum(
                                1 for c in tool_calls if c.get("status") == "success"
                            )
                            verify_result = await verifier.evaluate_output(
                                step_desc, result, tool_call_count=completed_calls
                            )

                        if verify_result.passed:
                            # Verification passed - mark completed
                            logger.info(
                                "✓ Step %d verified - all tool calls confirmed",
                                step_idx,
                            )
                            step_outputs[step_idx] = result
                            plan.mark_step(
                                step_idx,
                                step_status="completed",
                                step_notes=verify_result.notes,
                            )
                            failure_history.clear()
                            if step_idx in plan.replanned_steps:
                                plan.replanned_steps.discard(step_idx)
                                plan.global_replan_count = 0
                            await emit(
                                "step_status",
                                {
                                    "step_idx": step_idx,
                                    "status": "completed",
                                    "output": result,
                                },
                            )
                        else:
                            # Verification failed - check reason
                            failed_calls = verifier.get_failed_calls(plan, step_idx)
                            failure_reason = verifier.get_failure_reason(plan, step_idx)

                            if failure_reason:
                                logger.warning(
                                    "⚠ Step %d verification FAILED - LLM reported: %s",
                                    step_idx,
                                    failure_reason,
                                )
                            elif failed_calls:
                                logger.warning(
                                    "⚠ Step %d verification FAILED - %d pending tool calls detected (hallucination)",
                                    step_idx,
                                    len(failed_calls),
                                )
                                for call in failed_calls:
                                    logger.warning(
                                        "  - Pending: %s(%s)",
                                        call["tool"],
                                        call.get("args", {}),
                                    )
                            else:
                                logger.warning(
                                    "⚠ Step %d verification FAILED - evaluator: %s",
                                    step_idx,
                                    verify_result.notes,
                                )

                            # Save failure context
                            failed_outputs[step_idx] = result
                            step_desc = plan.steps.get(step_idx, f"Step {step_idx}")

                            max_replan = self.config.tuning.max_replan_attempts
                            if plan.global_replan_count >= max_replan:
                                logger.error(
                                    "✗ Step %d blocked - max global replan attempts (%d) reached",
                                    step_idx,
                                    max_replan,
                                )
                                plan.mark_step(
                                    step_idx,
                                    step_status="blocked",
                                    step_notes="Max replan attempts reached",
                                )
                                await emit(
                                    "step_status",
                                    {
                                        "step_idx": step_idx,
                                        "status": "blocked",
                                        "error": "Max replan attempts reached",
                                    },
                                )
                            else:
                                plan.global_replan_count += 1
                                logger.info(
                                    "🔄 REPLANNING after step %d failure (global attempt %d/%d)",
                                    step_idx,
                                    plan.global_replan_count,
                                    max_replan,
                                )

                                await emit(
                                    "replanning",
                                    {
                                        "step_idx": step_idx,
                                        "attempt": plan.global_replan_count,
                                    },
                                )

                                replan_result = await replanner.replan(
                                    original_query=planner_query,
                                    failed_step_id=step_idx,
                                    failed_step_desc=step_desc,
                                    failed_output=failed_outputs.get(step_idx, ""),
                                    failed_reason=verify_result.notes,
                                    failed_tool_history=plan.step_tool_history.get(step_idx, []),
                                    attempt=plan.global_replan_count,
                                    max_attempts=max_replan,
                                    available_tools=available_tools,
                                    failure_history=list(failure_history),
                                )

                                # Record current failure for next replan attempt
                                failure_history.append(
                                    FailureRecord(
                                        step_description=step_desc,
                                        failure_notes=verify_result.notes,
                                        tool_history=list(plan.step_tool_history.get(step_idx, [])),
                                    )
                                )

                                if replan_result.action == "redesign":
                                    plan.apply_replan(
                                        failed_step_id=step_idx,
                                        new_description=replan_result.failed_step_description,
                                        new_intent=replan_result.failed_step_intent,
                                        continuation_steps=replan_result.continuation_steps,
                                        continuation_dependencies=replan_result.continuation_dependencies,
                                        continuation_intents=replan_result.continuation_intents,
                                    )
                                    new_count = len(replan_result.continuation_steps) + 1
                                    logger.info(
                                        "✓ Replanner redesigned plan with %d new steps",
                                        new_count,
                                    )
                                    await emit(
                                        "plan_updated",
                                        {
                                            "plan": plan.format_dag(),
                                            "steps": plan.steps,
                                            "dependencies": plan.dependencies,
                                            "step_intents": plan.step_intents,
                                        },
                                    )
                                else:
                                    logger.error("✗ Replanner gave up on step %d", step_idx)
                                    plan.mark_step(
                                        step_idx,
                                        step_status="blocked",
                                        step_notes="Replanner gave up",
                                    )
                                    await emit(
                                        "step_status",
                                        {
                                            "step_idx": step_idx,
                                            "status": "blocked",
                                            "error": "Replanner gave up",
                                        },
                                    )

            # Generate final result by aggregating step outputs
            progress = plan.get_progress()
            logger.info("=== AGGREGATION PHASE ===")
            logger.info(
                "Aggregating results: %d/%d steps completed, %d blocked",
                progress["completed"],
                progress["total"],
                progress["blocked"],
            )
            final_result = await self._aggregate_results(query, plan, step_outputs)

            logger.info("=== FINAL PLAN STATE ===")
            for i, step in plan.steps.items():
                status = plan.step_statuses.get(i, "unknown")
                intent = plan.step_intents.get(i, "default")
                notes = plan.step_notes.get(i, "")
                deps = plan.dependencies.get(i, [])
                tool_count = len(plan.step_tool_history.get(i, []))
                files = plan.step_files.get(i, [])
                logger.info(
                    "  Step %d [%s] intent=%s deps=%s tools=%d files=%s | %s%s",
                    i,
                    status,
                    intent,
                    deps,
                    tool_count,
                    files,
                    step,
                    " | notes: %s" % notes if notes else "",
                )

            logger.info("=== EXECUTION COMPLETE ===")
            await emit("execution_complete", {"result": final_result})
            return final_result

        finally:
            # Cleanup
            TaskManager.remove_plan(plan_id)
            if self.sandbox:
                await self.sandbox.stop()

    async def _aggregate_results(self, query: str, plan: Plan, step_outputs: dict[int, str]) -> str:
        """Aggregate step outputs into a final result using LLM."""
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        # Build context from all step outputs in order
        outputs_text = "\n\n".join(
            f"=== Step {i}: {plan.steps[i]} ===\n{step_outputs.get(i, '[No output]')}"
            for i in sorted(plan.steps.keys())
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
            instruction=(
                "You are a content synthesizer. Combine multiple step outputs into a coherent final result.\n"
                "IMPORTANT: Preserve as much detail as possible from the step outputs. "
                "Do NOT summarize aggressively or omit specific data points, numbers, product names, "
                "case studies, or reference details. Organize and restructure for readability, "
                "but relay the information faithfully — do not compress it.\n"
                "IMPORTANT: If any step output contains URLs or links, you MUST include them in the final result "
                "so the user can click and access them. Never drop or omit URLs."
            ),
        )

        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name="aggregator", user_id="aggregator")

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
            logger.debug("Aggregator event: type=%s, attrs=%s", type(event).__name__, dir(event))
            if hasattr(event, "content") and event.content:
                logger.debug("Aggregator event.content: %s", event.content)
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_output = part.text
        logger.info("Aggregator final_output length: %d", len(final_output))

        # Include execution stats and step details
        progress = plan.get_progress()
        return f"""{final_output}

---
Execution: {progress["completed"]}/{progress["total"]} steps completed.

{plan.format()}"""
