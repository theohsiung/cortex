"""Verifier - Pure Python logic for tool call verification"""

from app.task.plan import Plan


class Verifier:
    """Verifies tool call status for plan steps"""

    def verify_step(self, plan: Plan, step_idx: int) -> bool:
        """
        Verify a step's tool call status.

        Args:
            plan: The execution plan
            step_idx: Index of the step to verify

        Returns:
            True if pass (no pending calls), False if fail (has pending = hallucination)
        """
        tool_history = plan.step_tool_history.get(step_idx, [])

        # Case 1: No tool calls = pass (step doesn't need tools)
        if not tool_history:
            return True

        # Case 2: Check for pending calls
        for call in tool_history:
            if call.get("status") == "pending":
                return False  # Has hallucination

        # Case 3: All success = pass
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
