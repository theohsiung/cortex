"""Prompts for ReplannerAgent."""

from __future__ import annotations

REPLANNER_SYSTEM_PROMPT = """You are a plan redesign specialist. When a step fails, you analyze what went wrong and design a new approach.

## Your Task
You will receive:
1. The original task description
2. Completed steps (context only — you don't need to reproduce them)
3. The failed step: its ID, description, failure reason, and what was attempted
4. Available tools and intents

Your job:
- Provide a NEW description for the failed step (a different approach, but still ONE single action)
- Provide continuation steps for any remaining work needed to complete the original task

## Output Format

```json
{
    "action": "redesign",
    "failed_step_description": "New approach for the failed step",
    "failed_step_intent": "<intent>",
    "continuation_steps": {
        "0": "First continuation step",
        "1": "Second continuation step"
    },
    "continuation_dependencies": {
        "1": [0]
    },
    "continuation_intents": {
        "0": "<intent>",
        "1": "<intent>"
    }
}
```

Or if the task cannot be completed:

```json
{
    "action": "give_up"
}
```

## Key Rules
- `failed_step_description`: A NEW description for the failed step. Must use a different approach than the original. Must be ONE single action — do NOT pack multiple actions into it.
- `continuation_steps`: Use local IDs starting from 0. The system will automatically assign real IDs and connect them to the rest of the DAG. Think about what work remains to complete the original task — any downstream steps that were in the plan will be removed, so you must re-plan them here.
- `continuation_dependencies`: Dependencies between continuation steps using local IDs. Steps with no dependencies listed are "root" steps and will automatically depend on all current terminal steps.
- Only omit `continuation_steps` if the failed step is the LAST step and no further work is needed.

## Guidelines
1. **Analyze the failure**: Understand WHY the step failed — use the error details and tool history
2. **Try a different approach**: Don't just rephrase — change the strategy, tools, or method
3. **Learn from errors**: Avoid repeating the same mistake (wrong tool, wrong query, etc.)
4. **Build on partial results**: Carefully read the executor output and tool call results. If the executor found USEFUL data (coordinates, place names, URLs, partial records), design new steps that build on that data instead of starting over
5. **ONE action per step**: Each step should do ONE thing (e.g. "search for X" or "read page Y"). Do NOT combine multiple actions like "search, then browse, then parse" in a single step
6. **Give up wisely**: Only give up if the task is truly impossible
7. **Intents**: You MUST ONLY use intents listed in the "Available Intents" section
"""


def build_replan_prompt(
    original_query: str,
    completed_steps: dict[int, dict[str, str]],
    completed_tool_history: str,
    failed_step_info: str,
    failed_step_id: int,
    available_tools: list[str],
    available_intents: dict[str, str] | None = None,
    attempt: int = 1,
    max_attempts: int = 3,
) -> str:
    """Build the prompt for replanning.

    Args:
        original_query: The user's original task
        completed_steps: {id: {"description": ..., "deps": ...}} for completed steps
        completed_tool_history: Formatted tool history of completed steps
        failed_step_info: Formatted info about the failed step
        failed_step_id: The ID of the failed step
        available_tools: List of available tool names
        available_intents: Dict of intent_name -> description for routing
        attempt: Current global replan attempt number
        max_attempts: Maximum global replan attempts

    Returns:
        Complete prompt for the replanner
    """
    # Completed steps section
    if completed_steps:
        completed_lines = []
        for sid in sorted(completed_steps.keys()):
            info = completed_steps[sid]
            completed_lines.append(f"  Step {sid}: {info['description']} (deps: {info['deps']})")
        completed_section = "\n".join(completed_lines)
    else:
        completed_section = "(No completed steps)"

    tools_section = "\n".join(f"- {tool}" for tool in available_tools)

    intents_section = ""
    if available_intents:
        intents_lines = "\n".join(f"- `{name}`: {desc}" for name, desc in available_intents.items())
        intents_section = f"""
---

## Available Intents

You MUST assign one of these intents to each step. Do NOT invent or use unlisted intent names:
{intents_lines}
"""

    attempt_note = ""
    if attempt > 1:
        attempt_note = (
            f"\n**This is global replan attempt {attempt} of {max_attempts}. "
            "Previous approaches failed — you MUST try a significantly different strategy.**\n"
        )

    return f"""## Original Task

{original_query}

---

## Completed Steps (Context — do NOT reproduce these in your output)

{completed_section}

### Completed Steps Tool History

{completed_tool_history if completed_tool_history else "(No tool calls recorded)"}

---

## Failed Step

{failed_step_info}

---

## Available Tools

{tools_section}
{intents_section}
---
{attempt_note}
**Remember:**
- The failed step description must do ONE thing only (different approach from the original)
- All downstream steps will be removed — add continuation steps for any remaining work needed to complete the original task
- The system will automatically connect continuation steps to the DAG
- Each step should do ONE thing

Please analyze the failure and provide your redesigned plan in JSON format.
"""
