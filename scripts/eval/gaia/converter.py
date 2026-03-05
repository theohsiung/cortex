"""Convert cortex Plan objects to LLM-planning pred format."""

from __future__ import annotations

from typing import Any


def convert_plan_to_pred(
    plan: Any,
    final_result_text: str,
) -> dict:
    """Convert a cortex Plan object to LLM-planning pred format.

    Args:
        plan: cortex Plan object with steps, dependencies, step_tool_history.
        final_result_text: The aggregated result text from cortex.execute().

    Returns:
        dict with plan_dag, tool_calls, final_answer matching LLM-planning schema.
    """
    nodes = _build_nodes(plan)
    edges = _build_edges(plan)
    tool_calls = _build_tool_calls(plan)
    final_answer = _extract_final_answer(plan, final_result_text)

    return {
        "plan_dag": {"nodes": nodes, "edges": edges},
        "tool_calls": tool_calls,
        "final_answer": final_answer,
    }


def _is_successful_call(call: dict) -> bool:
    """Check if a tool call record represents a successful execution.

    Handles both recording paths in Plan:
    - add_tool_call_pending + update_tool_result: sets status="success"
    - add_tool_call (legacy): has "result" key but no "status" key
    """
    return call.get("status") == "success" or ("result" in call and "status" not in call)


def _build_nodes(plan: Any) -> list[dict]:
    """Convert plan steps to LLM-planning node format."""
    nodes = []
    for idx in sorted(plan.steps.keys()):
        tool_history = plan.step_tool_history.get(idx, [])
        successful_calls = [c for c in tool_history if _is_successful_call(c)]
        first_tool = successful_calls[0]["tool"] if successful_calls else None

        nodes.append(
            {
                "node_id": f"n{idx}",
                "step_index": idx,
                "label": plan.steps[idx],
                "step_type": "tool" if successful_calls else "thought",
                "tool_id": first_tool,
            }
        )
    return nodes


def _build_edges(plan: Any) -> list[dict]:
    """Convert plan dependencies (child->[parents]) to edges (source->target)."""
    edges = []
    for child, parents in plan.dependencies.items():
        for parent in parents:
            if parent in plan.steps and child in plan.steps:
                edges.append(
                    {
                        "source": f"n{parent}",
                        "target": f"n{child}",
                    }
                )
    return edges


def _build_tool_calls(plan: Any) -> list[dict]:
    """Extract successful tool calls in execution order."""
    tool_calls = []
    call_idx = 0
    for step_idx in sorted(plan.steps.keys()):
        for call in plan.step_tool_history.get(step_idx, []):
            if not _is_successful_call(call):
                continue
            tool_calls.append(
                {
                    "call_index": call_idx,
                    "node_id": f"n{step_idx}",
                    "tool_id": call["tool"],
                    "alternative_tools": [],
                    "arguments": [
                        {"name": k, "value": str(v)} for k, v in call.get("args", {}).items()
                    ],
                }
            )
            call_idx += 1
    return tool_calls


def _extract_final_answer(plan: Any, final_result_text: str) -> dict:
    """Extract final answer from submit_final_answer tool call, or fallback to text."""
    # Search for submit_final_answer tool call
    for step_idx in sorted(plan.steps.keys()):
        for call in plan.step_tool_history.get(step_idx, []):
            if call.get("tool") == "submit_final_answer" and _is_successful_call(call):
                args = call.get("args", {})
                return {
                    "answer_type": args.get("answer_type", "string"),
                    "answer": args.get("answer", ""),
                }

    # Fallback: extract from aggregated text (before the "---" separator)
    text = final_result_text.split("---")[0].strip() if final_result_text else ""
    return {
        "answer_type": "string",
        "answer": text,
    }
