"""Prompts for ReplannerAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.replanner.replanner_agent import ReplanContext

REPLANNER_SYSTEM_PROMPT = """You are a plan redesign specialist. When a step fails, you analyze what went wrong and design a new approach.

## Your Task
You will receive:
1. The completed DAG (steps that succeeded — LOCKED, do not modify)
2. The failed step: its ID, description, failure reason, and what was attempted
3. The next available step ID for new downstream steps

Your job:
- Design a NEW approach for the failed step (same ID, different strategy)
- Generate new downstream steps that continue from the redesigned step

## Critical Rules

**COMPLETED STEPS ARE LOCKED:**
- Do NOT modify, remove, or reassign completed step IDs
- Your new downstream steps can depend on any completed step by its real ID

**FAILED STEP keeps its ID:**
- Provide a new description and intent via `retry_step`
- Its original dependencies are preserved automatically — do NOT include it in `new_dependencies`

**NEW DOWNSTREAM STEPS:**
- Use IDs starting from `next_id` (sequential: next_id, next_id+1, ...)
- Dependencies can reference the failed step's ID or any completed step ID

## Output Format

```json
{
    "action": "redesign",
    "retry_step": {
        "description": "New approach description for the failed step",
        "intent": "<intent>"
    },
    "new_steps": {"<next_id>": "Step description", "<next_id+1>": "Step description"},
    "new_dependencies": {"<next_id>": [<dep_ids>], "<next_id+1>": [<dep_ids>]},
    "new_intents": {"<next_id>": "<intent>", "<next_id+1>": "<intent>"}
}
```

Or if the task cannot be completed:

```json
{
    "action": "give_up",
    "retry_step": null,
    "new_steps": {},
    "new_dependencies": {},
    "new_intents": {}
}
```

## Guidelines
1. **Analyze the failure**: Understand WHY the step failed — use the error details and tool history
2. **Try a different approach**: Don't just rephrase — change the strategy, tools, or method
3. **Learn from errors**: Avoid repeating the same mistake (wrong tool, wrong query, etc.)
4. **Build on completed work**: Reference completed steps by their real IDs
5. **Break down if needed**: Generate multiple downstream steps for complex remaining work
6. **Give up wisely**: Only give up if the task is truly impossible
7. **Intents**: You MUST ONLY use intents listed in the "Available Intents" section
"""


def build_replan_prompt(
    completed_dag: str,
    completed_tool_history: str,
    failed_step_info: str,
    failed_step_id: int,
    next_id: int,
    available_tools: list[str],
    available_intents: dict[str, str] | None = None,
    context: ReplanContext | None = None,
) -> str:
    """
    Build the prompt for replanning.

    Args:
        completed_dag: Formatted completed DAG structure
        completed_tool_history: Formatted tool history of completed steps
        failed_step_info: Formatted info about the failed step
        failed_step_id: The ID of the failed step
        next_id: Next available step ID
        available_tools: List of available tool names
        available_intents: Dict of intent_name -> description for routing
        context: Optional ReplanContext with failure details

    Returns:
        Complete prompt for the replanner
    """
    task_section = ""
    if context:
        task_section = f"## Original Task\n\n{context.original_query}\n\n---\n\n"

    tools_section = "\n".join(f"- {tool}" for tool in available_tools)

    intents_section = ""
    if available_intents:
        intents_lines = "\n".join(f"- `{name}`: {desc}" for name, desc in available_intents.items())
        intents_section = f"""
---

## Available Intents

You MUST assign one of these intents to each new step in `new_intents` AND to `retry_step.intent`. Do NOT invent or use unlisted intent names:
{intents_lines}
"""

    attempt_note = ""
    if context and context.attempt_number > 1:
        attempt_note = (
            f"\n**This is attempt {context.attempt_number} of {context.max_attempts}. "
            "Previous approaches failed — you MUST try a significantly different strategy.**\n"
        )

    return f"""{task_section}## Completed DAG (LOCKED — do NOT modify)

{completed_dag}

### Completed Steps Tool History

{completed_tool_history if completed_tool_history else "(No tool calls recorded)"}

---

## Failed Step (ID {failed_step_id} — keep this ID, redesign the approach)

{failed_step_info}

---

## Next Available ID: {next_id}

New downstream steps MUST use IDs starting from {next_id}.
The failed step (ID {failed_step_id}) keeps its ID — provide its new approach in `retry_step`.

---

## Available Tools

{tools_section}
{intents_section}
---
{attempt_note}
**Remember:**
- Completed steps are LOCKED
- Failed step ID {failed_step_id} is preserved — put new approach in `retry_step`
- New downstream steps use IDs starting from {next_id}
- Dependencies can reference any existing ID (completed or failed step)

Please analyze the failure and provide your redesign decision in JSON format.
"""
