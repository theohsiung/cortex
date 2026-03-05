"""Verifier - Tool call verification and LLM-based output evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.task.plan import Plan


@dataclass
class VerifyResult:
    """Result from verification."""

    passed: bool
    notes: str = ""


class Verifier:
    """Verifies step execution via tool call checking and LLM-based output evaluation."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def verify_step(self, plan: Plan, step_idx: int) -> VerifyResult:
        """
        Verify a step's tool call status and Notes content.
        Returns VerifyResult instead of bool.
        """
        # Check Notes for [FAIL] marker first
        notes = plan.step_notes.get(step_idx, "").strip()
        if notes.startswith("[FAIL]"):
            return VerifyResult(passed=False, notes=notes)

        # Check for pending tool calls
        tool_history = plan.step_tool_history.get(step_idx, [])
        for call in tool_history:
            if call.get("status") == "pending":
                return VerifyResult(
                    passed=False, notes="Pending tool calls detected (hallucination)"
                )

        return VerifyResult(passed=True, notes=notes if notes else "")

    async def evaluate_output(
        self,
        step_description: str,
        executor_output: str,
        tool_call_count: int = 0,
    ) -> VerifyResult:
        """
        Evaluate executor output using LLM to determine success/failure.

        Args:
            step_description: What the step was supposed to do
            executor_output: Raw output from executor
            tool_call_count: Number of tool calls completed during execution

        Returns:
            VerifyResult with passed/failed and notes
        """
        if self.model is None:
            return VerifyResult(passed=True, notes=f"[SUCCESS]: {executor_output[:100]}")

        return await self._llm_evaluate(step_description, executor_output, tool_call_count)

    def build_evaluation_prompt(
        self,
        step_description: str,
        executor_output: str,
        tool_call_count: int = 0,
    ) -> str:
        """Build the evaluation prompt for LLM-based verification."""
        if tool_call_count > 0:
            tool_context = (
                f"\nThe executor completed {tool_call_count} tool call(s) during this step.\n"
            )
            criteria = (
                "- [SUCCESS] if the executor performed actions relevant to the step's goal\n"
                "  and produced a result. You are checking PROCESS, not judging the answer.\n"
                "  If the executor queried the right source, used appropriate tools, and\n"
                "  returned data — that is a success, regardless of result quantity or content.\n"
                "- [FAIL] ONLY if:\n"
                "  (a) the executor did not attempt the task (wrong tool, wrong target), OR\n"
                "  (b) the output contains an explicit error or exception, OR\n"
                "  (c) the output is empty with no usable data at all."
            )
        else:
            tool_context = ""
            criteria = (
                "- [SUCCESS] if the executor produced a relevant response that addresses\n"
                "  the step's goal. Analysis, reasoning, and text responses are valid.\n"
                "- [FAIL] only if the output is empty, completely irrelevant, or contains\n"
                "  a clear error."
            )

        return (
            "You are a process verifier. Your job is to check whether the executor"
            " performed the correct actions for this step"
            " — NOT to judge the correctness or completeness of the answer.\n\n"
            f"Step: {step_description}\n"
            f"{tool_context}"
            "Executor output:\n"
            f"{executor_output}\n\n"
            "Evaluation criteria:\n"
            f"{criteria}\n\n"
            "IMPORTANT: Do NOT evaluate the quality or completeness of the answer."
            " If the executor did the right work and got a result, it passes.\n\n"
            "Respond with EXACTLY one of:\n"
            "- [SUCCESS]: <brief description of what the executor did>\n"
            "- [FAIL]: <what went wrong with the executor's process>\n"
        )

    async def _llm_evaluate(
        self,
        step_description: str,
        executor_output: str,
        tool_call_count: int = 0,
    ) -> VerifyResult:
        """
        Use LLM to evaluate whether executor output fulfills the step.
        """
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        prompt = self.build_evaluation_prompt(step_description, executor_output, tool_call_count)
        assert self.model is not None
        agent = LlmAgent(
            name="verifier",
            model=self.model,
            instruction=(
                "You are a process verifier. Check whether the executor"
                " performed the correct actions"
                " — not whether the answer is correct. Be concise."
            ),
        )

        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name="verifier", user_id="verifier")

        runner = Runner(
            agent=agent,
            session_service=session_service,
            app_name="verifier",
        )
        content = Content(parts=[Part(text=prompt)])

        final_output = ""
        async for event in runner.run_async(
            user_id="verifier", session_id=session.id, new_message=content
        ):
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_output = part.text

        # Parse output
        if final_output.strip().startswith("[FAIL]"):
            return VerifyResult(passed=False, notes=final_output.strip())
        else:
            return VerifyResult(passed=True, notes=final_output.strip())

    # Keep existing helper methods
    def get_failed_calls(self, plan: Plan, step_idx: int) -> list[dict]:
        """Get list of failed (pending) tool calls for a step."""
        tool_history = plan.step_tool_history.get(step_idx, [])
        return [call for call in tool_history if call.get("status") == "pending"]

    def get_failure_reason(self, plan: Plan, step_idx: int) -> str:
        """Get the failure reason from Notes if [FAIL] tag present."""
        notes = plan.step_notes.get(step_idx, "").strip()
        if notes.startswith("[FAIL]"):
            if notes.startswith("[FAIL]:"):
                return notes[7:].strip()
            else:
                return notes[6:].strip()
        return ""
