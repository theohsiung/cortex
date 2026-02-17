"""Prompts for ReplannerAgent."""

from __future__ import annotations

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
   - The intent describes the purpose of the step (e.g., "generate", "review", "validate")
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
    "new_intents": {"0": "generate", "1": "generate", "2": "generate", "3": "generate", "4": "review"}
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
) -> str:
    """
    Build the prompt for replanning.

    Args:
        completed_tool_history: Formatted tool history of completed steps
        steps_to_replan: List of (index, description) for steps to redesign
        available_tools: List of available tool names
        available_intents: Dict of intent_name -> description for routing

    Returns:
        Complete prompt for the replanner
    """
    steps_section = "\n".join(f"- Step {idx}: {desc}" for idx, desc in steps_to_replan)
    tools_section = "\n".join(f"- {tool}" for tool in available_tools)

    # Build available intents section
    intents_section = ""
    if available_intents and len(available_intents) > 1:
        intents_lines = "\n".join(f"- `{name}`: {desc}" for name, desc in available_intents.items())
        intents_section = f"""
---

## Available Intents

Assign one of these intents to each new step in `new_intents`:
{intents_lines}
"""

    return f"""## Completed Steps (READ-ONLY - DO NOT MODIFY)

{completed_tool_history if completed_tool_history else "(No completed steps)"}

---

## Steps to Redesign (replace these entirely)

{steps_section}

---

## Available Tools

{tools_section}
{intents_section}
---

**Remember:**
- Completed steps are LOCKED - do not reference or modify them
- Your new steps will be inserted after the last completed step
- Use RELATIVE indices (0-based) for dependencies between your new steps only
- The system handles connecting your first step to the completed work

Please analyze the situation and provide your redesign decision in JSON format.
"""
