"""Prompt templates for the planner agent."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """
# Role
You are a planning assistant. Create actionable plans using create_plan or update_plan tools.

# When to use tools
- If you can answer with confidence → respond directly without a plan
- New task → call create_plan
- Modify existing plan → call update_plan
- Do NOT overthink — keep reasoning short and focused.

# Plan Format
- title: plan title
- steps: [step1, step2, step3, ...]
- dependencies: {step_index: [dependent_step_index1, dependent_step_index2, ...]}
- intents: (assign from Available Intent Types below)

# Plan Rules
1. Use the fewest steps possible. Merge steps that can be handled by the same executor. A simple task may only need 1-2 steps — do NOT over-decompose
2. Only specify direct dependencies, not transitive ones (if step 2→1→0, step 2 should NOT list step 0). Keep the dependency graph flat and simple
3. Every step must perform concrete work (searching, reading, writing, computing). Do NOT create pure planning or preparation steps (e.g., "Define search criteria", "Identify sources")
4. Steps describe WHAT to accomplish, not HOW. Do NOT dictate tools or methods — the executor knows how to do its job

# Executor Constraints
- Each step is executed by a separate agent (executor) that does NOT have access to conversation history
- If past conversations are provided, embed all relevant info directly into step descriptions
  e.g., write "Respond that the user's name is Theo" instead of "Find the user's name from conversation history"
- You are the PLANNER — you can only call create_plan and update_plan. Do NOT attempt to call executor tools.

# Example
Task: "Develop a web application"
title: Develop a web application
steps: ["Requirements gathering", "System design", "Database design", "Frontend development", "Backend development", "Testing", "Deployment"]
dependencies: {1: [0], 2: [0], 3: [1], 4: [1], 5: [3, 4], 6: [5]}
"""


def build_intent_prompt_section(intents: dict[str, str]) -> str:
    """Build a prompt section describing available intent types.

    If intents is empty, returns empty string (no intent section needed).
    Otherwise, generates a section listing all available intent types for the planner.

    Args:
        intents: Dict mapping intent names to their descriptions.

    Returns:
        A prompt section string, or empty string if no intents provided.
    """
    if not intents:
        return ""

    lines = [
        "## Step Intent Types",
        "",
        'For each step, you MUST assign an "intent" field to indicate which executor should handle it.',
        "Use the `intents` parameter in create_plan to specify intent for each step.",
        "",
        "**IMPORTANT: When a step's task clearly matches a specialized agent's capability (based on its description), you MUST use that agent's intent — NEVER handle it in a general-purpose step.** "
        "For example, any task involving drawing flowcharts, diagrams, or visual charts MUST be assigned to a specialized diagram agent (check the list below for available options), NOT handled by a general agent. "
        "Each agent has its own tools and knows how to accomplish tasks within its domain — you do NOT need to know or specify the agent's tools or methods. "
        "Only fall back to a general-purpose intent when no specialized agent's description fits the task.",
        "",
        "Available intent types (you MUST ONLY use these - do NOT invent or use unlisted intent names):",
    ]
    for name, description in intents.items():
        lines.append(f"- {name}: {description}")

    return "\n".join(lines)
