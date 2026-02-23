"""Prompts for ReplannerAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.replanner.replanner_agent import ReplanContext

REPLANNER_SYSTEM_PROMPT = """You are a plan redesign specialist. When a step fails due to tool call issues, you analyze the situation and redesign the affected steps.

## Your Task
Analyze the failed step and all downstream dependencies, then provide a redesigned plan for those steps.

## Critical Constraints

**DO NOT MODIFY COMPLETED STEPS:**
- Completed steps and their outputs are LOCKED and cannot be changed
- The DAG structure of completed steps MUST remain intact
- Your new steps will be inserted AFTER the last completed step
- The system will automatically connect your first new step to the last completed step

**ONLY REDESIGN THE FAILED SUBGRAPH:**
- You are only redesigning the failed step and its downstream dependencies
- The new steps replace the failed subgraph entirely
- Dependencies you specify are RELATIVE indices within your new steps only

## Input You Will Receive
1. **Completed Steps**: Full tool call history of successfully completed steps (READ-ONLY)
2. **Failed Steps**: The step that failed and its downstream dependencies that need redesign
3. **Available Tools**: List of tools you can use in the new plan

## Output Format
You MUST respond with a JSON block containing your redesign decision:

```json
{
    "action": "redesign",
    "new_steps": ["Step description 1", "Step description 2", ...],
    "new_dependencies": {"1": [0], "2": [1], ...},
    "new_intents": {"0": "generate", "1": "review", ...}
}
```

Or if the task cannot be completed:

```json
{
    "action": "give_up",
    "new_steps": [],
    "new_dependencies": {},
    "new_intents": {}
}
```

## Guidelines
1. **Analyze the failure**: Understand why the step failed from the tool history
2. **Build on completed work**: Use the completed steps' outputs as your foundation
3. **Break down complex steps**: If a step failed, consider splitting it into smaller steps
4. **Use available tools**: Only reference tools from the available tools list
5. **Dependencies format**: Use RELATIVE indices (0-based) for new_dependencies
   - Key is the new step index, value is list of dependency indices within new steps
   - First step (index 0) needs no entry - it automatically depends on the last completed step
   - Example: `{"1": [0], "2": [1]}` means step 1 depends on step 0, step 2 depends on step 1
6. **Give up wisely**: Only give up if the task is truly impossible with available tools
7. **Intents format**: Assign an intent to each new step using RELATIVE indices (0-based)
   - You MUST ONLY use intents listed in "Available Intents" section - do NOT invent intent names
   - Each key in `new_intents` maps to the corresponding step index in `new_steps`

## Example

Original plan:
- Step 0: Analyze requirements [COMPLETED]
- Step 1: Create project structure [COMPLETED]
- Step 2: Build API endpoints [FAILED - pending write_file]
- Step 3: Write tests (depends on 2)
- Step 4: Run tests (depends on 3)

You should redesign steps 2, 3, 4 while keeping steps 0, 1 intact:

```json
{
    "action": "redesign",
    "new_steps": [
        "Create API route structure",
        "Implement GET endpoints",
        "Implement POST endpoints",
        "Add error handling",
        "Write and run tests"
    ],
    "new_dependencies": {"1": [0], "2": [0], "3": [1, 2], "4": [3]},
    "new_intents": {"0": "<intent-from-available-list>", "1": "<intent-from-available-list>", "2": "<intent-from-available-list>", "3": "<intent-from-available-list>", "4": "<intent-from-available-list>"}
}
```

The system will:
1. Remove failed steps 2, 3, 4
2. Insert your 5 new steps after step 1
3. Automatically make your step 0 depend on the last completed step (step 1)
"""


def build_replan_prompt(
    completed_tool_history: str,
    steps_to_replan: list[tuple[int, str]],
    available_tools: list[str],
    available_intents: dict[str, str] | None = None,
    context: ReplanContext | None = None,
) -> str:
    """
    Build the prompt for replanning.

    Args:
        completed_tool_history: Formatted tool history of completed steps
        steps_to_replan: List of (index, description) for steps to redesign
        available_tools: List of available tool names
        available_intents: Dict of intent_name -> description for routing
        context: Optional ReplanContext with failure details for richer prompts

    Returns:
        Complete prompt for the replanner
    """
    # Build original task section if context provided
    task_section = ""
    if context:
        task_section = f"## Original Task\n\n{context.original_query}\n\n---\n\n"

    # Build steps section - enriched with failure details when context is available
    if context:
        failed_lines = []
        for idx, desc in steps_to_replan:
            failed_lines.append(f"### Step {idx}: {desc}")
            failed_lines.append(f"Attempt: {context.attempt_number}/{context.max_attempts}")

            notes = context.failed_step_notes.get(idx, "")
            if notes:
                failed_lines.append(f"\nFailure reason:\n{notes}")

            output = context.failed_step_outputs.get(idx, "")
            if output:
                if len(output) > 500:
                    output = "...[truncated]\n" + output[-500:]
                failed_lines.append(f"\nExecutor output (last 500 chars):\n{output}")

            tool_history = context.failed_tool_history.get(idx, [])
            if tool_history:
                failed_lines.append("\nTool calls:")
                for call in tool_history:
                    tool = call.get("tool", "?")
                    args = call.get("args", {})
                    result = call.get("result", "")
                    args_str = (
                        ", ".join(f'{k}="{v}"' for k, v in args.items())
                        if isinstance(args, dict)
                        else str(args)
                    )
                    failed_lines.append(f"- {tool}({args_str}) -> {result}")

            failed_lines.append("")
        steps_section = "\n".join(failed_lines)
    else:
        steps_section = "\n".join(f"- Step {idx}: {desc}" for idx, desc in steps_to_replan)

    tools_section = "\n".join(f"- {tool}" for tool in available_tools)

    # Build available intents section
    intents_section = ""
    if available_intents:
        intents_lines = "\n".join(f"- `{name}`: {desc}" for name, desc in available_intents.items())
        intents_section = f"""
---

## Available Intents

You MUST assign one of these intents to each new step in `new_intents`. Do NOT invent or use unlisted intent names:
{intents_lines}
"""

    # Build attempt escalation note
    attempt_note = ""
    if context and context.attempt_number > 1:
        attempt_note = (
            f"\n**This is attempt {context.attempt_number} of {context.max_attempts}. "
            "If previous approaches failed, try a significantly different strategy.**\n"
        )

    return f"""{task_section}## Completed Steps (READ-ONLY - DO NOT MODIFY)

{completed_tool_history if completed_tool_history else "(No completed steps)"}

---

## Steps to Redesign (replace these entirely)

{steps_section}

---

## Available Tools

{tools_section}
{intents_section}
---
{attempt_note}
**Remember:**
- Completed steps are LOCKED - do not reference or modify them
- Your new steps will be inserted after the last completed step
- Use RELATIVE indices (0-based) for dependencies between your new steps only
- The system handles connecting your first step to the completed work

Please analyze the situation and provide your redesign decision in JSON format.
"""
