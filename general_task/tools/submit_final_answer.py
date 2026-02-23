"""Submit final answer tool."""

from __future__ import annotations

from google.adk.tools import FunctionTool


def submit_final_answer(answer: str, answer_type: str = "string") -> str:
    """Submit the final answer to the task.

    This tool marks the task as completed.
    YOU MUST USE THIS TOOL TO FINISH THE TASK.

    Args:
        answer: The final answer string.
        answer_type: The type of answer (e.g., "string", "number", "date"). Default: string.
    """
    return f"[FINAL ANSWER SUBMITTED] {answer} (Type: {answer_type})"


submit_final_answer_tool = FunctionTool(submit_final_answer)
