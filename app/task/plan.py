"""Execution plan with steps, dependencies, and status tracking.

Steps are stored as dict[int, str] where keys are stable integer IDs.
IDs are monotonically increasing, ensuring stability during replan operations.
"""

from __future__ import annotations

from typing import Any

# Constants for tool history
RESULT_MAX_LENGTH = 200


class Plan:
    """Represents an execution plan with steps, dependencies, and status tracking."""

    @staticmethod
    def _normalize_dependencies(deps: dict) -> dict[int, list[int]]:
        """Normalize dependencies dict to ensure all keys and values are integers.

        JSON parsing converts dict keys to strings, so we need to convert them back.
        """
        if not deps:
            return {}
        return {int(k): [int(v) for v in vals] for k, vals in deps.items()}

    def __init__(
        self,
        title: str = "",
        steps: list[str] | None = None,
        dependencies: dict[int, list[int]] | None = None,
        step_intents: dict[int, str] | None = None,
    ) -> None:
        self.title = title
        self._next_id = 0
        self.steps: dict[int, str] = {}
        self.step_statuses: dict[int, str] = {}
        self.step_notes: dict[int, str] = {}

        if steps:
            for desc in steps:
                sid = self._next_id
                self.steps[sid] = desc
                self.step_statuses[sid] = "not_started"
                self.step_notes[sid] = ""
                self._next_id += 1

        # Auto-generate sequential dependencies if not provided
        if dependencies is not None:
            self.dependencies = self._normalize_dependencies(dependencies)
        elif len(self.steps) > 1:
            ids = sorted(self.steps.keys())
            self.dependencies = {ids[i]: [ids[i - 1]] for i in range(1, len(ids))}
        else:
            self.dependencies = {}

        # Initialize tool history and file tracking
        self.step_tool_history: dict[int, list[dict]] = {}
        self.step_files: dict[int, list[str]] = {}

        # Initialize replan tracking
        self.replan_attempts: dict[int, int] = {}

        # Initialize step intents: fill missing indices with "default"
        self.step_intents: dict[int, str] = {}
        if step_intents:
            self.step_intents.update(step_intents)
        for sid in self.steps:
            if sid not in self.step_intents:
                self.step_intents[sid] = "default"

    def get_step_intent(self, idx: int) -> str:
        """Get the intent for a step, defaulting to 'default' if not found."""
        return self.step_intents.get(idx, "default")

    def update(
        self,
        title: str | None = None,
        steps: list[str] | None = None,
        dependencies: dict[int, list[int]] | None = None,
        step_intents: dict[int, str] | None = None,
    ) -> None:
        """Update plan properties. Used by planner for initial plan creation/modification."""
        if title is not None:
            self.title = title

        if steps is not None:
            # Reset everything for new steps
            self._next_id = 0
            self.steps = {}
            self.step_statuses = {}
            self.step_notes = {}
            self.step_tool_history = {}
            self.step_files = {}
            self.step_intents = {}

            for desc in steps:
                sid = self._next_id
                self.steps[sid] = desc
                self.step_statuses[sid] = "not_started"
                self.step_notes[sid] = ""
                self._next_id += 1

            # Set intents
            if step_intents:
                self.step_intents.update(step_intents)
            for sid in self.steps:
                if sid not in self.step_intents:
                    self.step_intents[sid] = "default"

            # Auto-generate dependencies if not provided
            if dependencies is None and len(self.steps) > 1:
                ids = sorted(self.steps.keys())
                self.dependencies = {ids[i]: [ids[i - 1]] for i in range(1, len(ids))}

        if dependencies is not None:
            self.dependencies = self._normalize_dependencies(dependencies)

    def mark_step(
        self,
        step_index: int,
        step_status: str | None = None,
        step_notes: str | None = None,
    ) -> None:
        """Mark a step with status and/or notes."""
        if step_index not in self.steps:
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_status is not None:
            self.step_statuses[step_index] = step_status

        if step_notes is not None:
            self.step_notes[step_index] = step_notes

    def add_tool_call(
        self, step_index: int, tool: str, args: dict, result: Any, timestamp: str
    ) -> None:
        """Record a tool call for a step."""
        if step_index not in self.steps:
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            self.step_tool_history[step_index] = []

        # Truncate result if too long
        result_str = str(result)
        if len(result_str) > RESULT_MAX_LENGTH:
            result_str = result_str[:RESULT_MAX_LENGTH] + "...[truncated]"

        self.step_tool_history[step_index].append(
            {"tool": tool, "args": args, "result": result_str, "timestamp": timestamp}
        )

    def add_tool_call_pending(self, step_index: int, tool: str, args: dict, call_time: str) -> None:
        """Record a pending tool call for a step."""
        if step_index not in self.steps:
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            self.step_tool_history[step_index] = []

        self.step_tool_history[step_index].append(
            {"tool": tool, "args": args, "status": "pending", "call_time": call_time}
        )

    def update_tool_result(
        self, step_index: int, tool: str, result: Any, response_time: str
    ) -> None:
        """Update a pending tool call to success with result (FIFO matching)."""
        if step_index not in self.step_tool_history:
            return

        # Find first pending call with matching tool name (FIFO)
        for call in self.step_tool_history[step_index]:
            if call["tool"] == tool and call["status"] == "pending":
                # Truncate result if too long
                result_str = str(result)
                if len(result_str) > RESULT_MAX_LENGTH:
                    result_str = result_str[:RESULT_MAX_LENGTH] + "...[truncated]"

                call["status"] = "success"
                call["result"] = result_str
                call["response_time"] = response_time
                return

    def finalize_step(self, step_index: int) -> bool:
        """
        Finalize a step and check for pending tool calls.

        Returns:
            True if all tool calls are successful (or no tool calls)
            False if there are pending tool calls (hallucination detected)
        """
        if step_index not in self.steps:
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            return True  # No tool calls = OK

        for call in self.step_tool_history[step_index]:
            if call["status"] == "pending":
                return False  # Has pending = hallucination

        return True  # All success

    def add_file(self, step_index: int, file_path: str) -> None:
        """Record a generated file for a step."""
        if step_index not in self.steps:
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_files:
            self.step_files[step_index] = []

        if file_path not in self.step_files[step_index]:
            self.step_files[step_index].append(file_path)

    def get_ready_steps(self) -> list[int]:
        """Get indices of steps ready to execute (dependencies satisfied)."""
        ready = []

        for idx in self.steps:
            # Skip if already started or completed
            if self.step_statuses[idx] != "not_started":
                continue

            # Check if all dependencies are completed
            deps = self.dependencies.get(idx, [])
            all_deps_done = all(self.step_statuses.get(dep) == "completed" for dep in deps)

            if all_deps_done:
                ready.append(idx)

        return ready

    def get_downstream_steps(self, step_idx: int) -> list[int]:
        """
        Get all downstream steps (direct + indirect dependents).

        Args:
            step_idx: The step index to find dependents for.

        Returns:
            Sorted list of step indices that depend on this step.
        """
        downstream = set()
        to_check = [step_idx]

        while to_check:
            current = to_check.pop()
            for idx, deps in self.dependencies.items():
                if idx not in self.steps:
                    continue
                if current in deps and idx not in downstream:
                    downstream.add(idx)
                    to_check.append(idx)

        return sorted(downstream)

    def get_progress(self) -> dict[str, int]:
        """Get progress statistics."""
        statuses = list(self.step_statuses.values())
        return {
            "total": len(self.steps),
            "completed": statuses.count("completed"),
            "in_progress": statuses.count("in_progress"),
            "blocked": statuses.count("blocked"),
            "not_started": statuses.count("not_started"),
        }

    def format_dag(self) -> str:
        """Format DAG structure for debugging.

        Returns:
            Compact DAG representation showing steps and dependencies.
        """
        lines = []
        for idx in sorted(self.steps.keys()):
            step = self.steps[idx]
            status = self.step_statuses.get(idx, "?")
            status_char = {"completed": "✓", "blocked": "!", "in_progress": "→"}.get(status, " ")
            deps = self.dependencies.get(idx, [])
            dep_str = f" ← {deps}" if deps else ""
            # Truncate step description for readability
            step_short = step[:30] + "..." if len(step) > 30 else step
            lines.append(f"  [{status_char}] {idx}: {step_short}{dep_str}")
        return "\n".join(lines)

    def format(self) -> str:
        """Format plan for display."""
        lines = [f"Plan: {self.title}", "=" * 40, ""]

        progress = self.get_progress()
        pct = (progress["completed"] / progress["total"] * 100) if progress["total"] > 0 else 0
        lines.append(f"Progress: {progress['completed']}/{progress['total']} ({pct:.1f}%)")
        lines.append("")
        lines.append("Steps:")

        status_symbols = {
            "not_started": "[ ]",
            "in_progress": "[→]",
            "completed": "[✓]",
            "blocked": "[!]",
        }

        for idx in sorted(self.steps.keys()):
            step = self.steps[idx]
            symbol = status_symbols.get(self.step_statuses.get(idx, "not_started"), "[ ]")
            deps = self.dependencies.get(idx, [])
            dep_str = f" (depends on: {deps})" if deps else ""
            lines.append(f"  {idx}: {symbol} {step}{dep_str}")

            notes = self.step_notes.get(idx, "")
            if notes:
                lines.append(f"      Notes: {notes}")

            # Show tool history summary
            if idx in self.step_tool_history:
                tool_counts: dict[str, int] = {}
                for call in self.step_tool_history[idx]:
                    tool_name = call["tool"]
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                tools_str = ", ".join(f"{t} ({c})" for t, c in tool_counts.items())
                lines.append(f"      Tools: {tools_str}")

            # Show files
            if idx in self.step_files and self.step_files[idx]:
                files_str = ", ".join(self.step_files[idx])
                lines.append(f"      Files: {files_str}")

        return "\n".join(lines)

    def replan(
        self,
        remove_ids: list[int],
        new_steps: dict[int, str],
        new_dependencies: dict[int, list[int]],
        new_intents: dict[int, str] | None = None,
    ) -> None:
        """Atomic replan: remove failed steps and add new steps.

        Args:
            remove_ids: Step IDs to remove (failed + downstream).
            new_steps: New steps as {id: description}. IDs assigned by LLM
                starting from _next_id.
            new_dependencies: Dependencies for new steps, using real IDs.
            new_intents: Optional intents for new steps.
        """
        remove_set = set(remove_ids)

        # Delete removed steps
        for sid in remove_ids:
            self.steps.pop(sid, None)
            self.step_statuses.pop(sid, None)
            self.step_notes.pop(sid, None)
            self.step_tool_history.pop(sid, None)
            self.step_files.pop(sid, None)
            self.step_intents.pop(sid, None)
            self.replan_attempts.pop(sid, None)

        # Clean stale deps: remove entries for deleted steps
        for sid in list(self.dependencies):
            if sid in remove_set:
                del self.dependencies[sid]
            else:
                cleaned = [d for d in self.dependencies[sid] if d not in remove_set]
                if cleaned:
                    self.dependencies[sid] = cleaned
                else:
                    del self.dependencies[sid]

        # Add new steps
        normalized_steps = {int(k): v for k, v in new_steps.items()}
        for sid, desc in normalized_steps.items():
            self.steps[sid] = desc
            self.step_statuses[sid] = "not_started"
            self.step_notes[sid] = ""
            self._next_id = max(self._next_id, sid + 1)

        # Add new dependencies
        normalized_deps = self._normalize_dependencies(new_dependencies)
        for sid, deps in normalized_deps.items():
            self.dependencies[sid] = deps

        # Add new intents
        if new_intents:
            for sid, intent in new_intents.items():
                self.step_intents[int(sid)] = intent
        # Default intents for new steps without explicit intent
        for sid in normalized_steps:
            if sid not in self.step_intents:
                self.step_intents[sid] = "default"

    def drop_steps(self, step_ids: list[int]) -> None:
        """Remove specified steps and clean up all references."""
        remove_set = set(step_ids)

        for sid in step_ids:
            self.steps.pop(sid, None)
            self.step_statuses.pop(sid, None)
            self.step_notes.pop(sid, None)
            self.step_tool_history.pop(sid, None)
            self.step_files.pop(sid, None)
            self.step_intents.pop(sid, None)
            self.replan_attempts.pop(sid, None)

        # Clean dependencies
        for sid in list(self.dependencies):
            if sid in remove_set:
                del self.dependencies[sid]
            else:
                cleaned = [d for d in self.dependencies[sid] if d not in remove_set]
                if cleaned:
                    self.dependencies[sid] = cleaned
                else:
                    del self.dependencies[sid]

    def reset_step(
        self,
        step_idx: int,
        new_description: str | None = None,
        new_intent: str | None = None,
    ) -> None:
        """Reset a step to not_started, optionally updating description and intent.

        Clears notes, tool history, and files. Preserves dependencies.
        """
        if step_idx not in self.steps:
            raise ValueError(f"Invalid step_index: {step_idx}")

        self.step_statuses[step_idx] = "not_started"
        self.step_notes[step_idx] = ""
        self.step_tool_history[step_idx] = []
        self.step_files.pop(step_idx, None)

        if new_description is not None:
            self.steps[step_idx] = new_description
        if new_intent is not None:
            self.step_intents[step_idx] = new_intent

    def recalculate_next_id(self) -> None:
        """Recalculate _next_id based on remaining step IDs."""
        if self.steps:
            self._next_id = max(self.steps.keys()) + 1
        else:
            self._next_id = 0

    def format_completed_dag(self) -> str:
        """Format the completed portion of the DAG for the replanner LLM.

        Returns a structured representation of completed steps with their
        IDs, descriptions, and dependencies.
        """
        lines = []
        for idx in sorted(self.steps.keys()):
            if self.step_statuses.get(idx) != "completed":
                continue
            step = self.steps[idx]
            deps = self.dependencies.get(idx, [])
            dep_str = f" (depends on: {deps})" if deps else ""
            lines.append(f"  {idx}: {step}{dep_str}")
        return "\n".join(lines) if lines else "(No completed steps)"

    def format_tool_history(self, step_indices: list[int]) -> str:
        """
        Format tool history for specified steps.

        Args:
            step_indices: List of step indices to format.

        Returns:
            Formatted string with tool call details.
        """
        lines = []

        for idx in step_indices:
            if idx not in self.steps:
                continue

            step_desc = self.steps[idx]
            lines.append(f"### Step {idx}: {step_desc}")

            if idx in self.step_tool_history and self.step_tool_history[idx]:
                lines.append("Tool calls:")
                for call in self.step_tool_history[idx]:
                    tool = call["tool"]
                    args = call.get("args", {})
                    result = call.get("result", "")

                    # Format args as key=value pairs
                    args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                    lines.append(f"- {tool}({args_str}) → {result}")
            else:
                lines.append("(No tool calls)")

            lines.append("")

        return "\n".join(lines)
