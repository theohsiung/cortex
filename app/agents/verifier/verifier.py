"""Verifier - Pure Python logic for tool call verification"""

from app.task.plan import Plan


class Verifier:
    """Verifies tool call status for plan steps"""

    def verify_step(self, plan: Plan, step_idx: int) -> bool:
        """
        Verify a step's tool call status and Notes content.

        Args:
            plan: The execution plan
            step_idx: Index of the step to verify

        Returns:
            True if pass, False if fail (has [FAIL] in Notes or pending tool calls)
        """
        # Check Notes for [FAIL] marker first
        step = plan.steps[step_idx]
        notes = plan.step_notes.get(step, "").strip()
        if notes.startswith("[FAIL]"):
            return False  # LLM self-reported failure

        # Check for pending tool calls
        tool_history = plan.step_tool_history.get(step_idx, [])
        for call in tool_history:
            if call.get("status") == "pending":
                return False  # Has hallucination

        return True

    def get_failed_calls(self, plan: Plan, step_idx: int) -> list[dict]:
        """
        Get list of failed (pending) tool calls for a step.

        Args:
            plan: The execution plan
            step_idx: Index of the step

        Returns:
            List of tool call dicts that are still pending
        """
        tool_history = plan.step_tool_history.get(step_idx, [])
        return [call for call in tool_history if call.get("status") == "pending"]

    def get_failure_reason(self, plan: Plan, step_idx: int) -> str:
        """
        Get the failure reason for a step.

        Args:
            plan: The execution plan
            step_idx: Index of the step

        Returns:
            Failure reason from Notes if [FAIL] tag present, else empty string
        """
        step = plan.steps[step_idx]
        notes = plan.step_notes.get(step, "").strip()
        if notes.startswith("[FAIL]"):
            # Extract reason after [FAIL]: or [FAIL]
            if notes.startswith("[FAIL]:"):
                return notes[7:].strip()
            else:
                return notes[6:].strip()
        return ""
