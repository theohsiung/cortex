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

        # Global replan counter (shared across all steps)
        self.global_replan_count: int = 0

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

    def _get_downstream(self, step_id: int) -> set[int]:
        """找出所有直接或間接依賴 step_id 的步驟。

        透過 BFS 遍歷 dependency graph 的子節點方向。
        dependencies[child] = [parents]，所以 child of X = 任何 Y where X in dependencies[Y]。
        """
        downstream: set[int] = set()
        queue = [step_id]
        while queue:
            current = queue.pop(0)
            for sid, deps in self.dependencies.items():
                if current in deps and sid not in downstream:
                    downstream.add(sid)
                    queue.append(sid)
        return downstream

    def _get_terminal_nodes(self) -> set[int]:
        """找出沒有子節點的終端步驟（不被任何其他步驟依賴的）。"""
        parents_set: set[int] = set()
        for deps in self.dependencies.values():
            parents_set.update(deps)
        return set(self.steps.keys()) - parents_set

    def apply_replan(
        self,
        failed_step_id: int,
        new_description: str,
        new_intent: str,
        continuation_steps: dict[int, str] | None = None,
        continuation_dependencies: dict[int, list[int]] | None = None,
        continuation_intents: dict[int, str] | None = None,
    ) -> None:
        """Apply a replan after a step failure.

        1. 刪除 failed step 的所有 downstream steps
        2. 重設 failed step（新 description、清 tool history）
        3. 如果有 continuation steps，map local IDs → actual IDs，
           並自動連接 continuation root nodes 到 DAG 的 terminal nodes
        """
        # 1. 刪除 downstream steps
        downstream = self._get_downstream(failed_step_id)
        for sid in downstream:
            self.steps.pop(sid, None)
            self.step_statuses.pop(sid, None)
            self.step_notes.pop(sid, None)
            self.step_tool_history.pop(sid, None)
            self.step_files.pop(sid, None)
            self.step_intents.pop(sid, None)
            self.dependencies.pop(sid, None)

        # 2. 重設 failed step
        self.steps[failed_step_id] = new_description
        self.step_statuses[failed_step_id] = "not_started"
        self.step_notes[failed_step_id] = ""
        self.step_tool_history.pop(failed_step_id, None)
        self.step_files.pop(failed_step_id, None)
        self.step_intents[failed_step_id] = new_intent or "default"

        # 3. 加入 continuation steps
        if continuation_steps:
            continuation_dependencies = continuation_dependencies or {}
            continuation_intents = continuation_intents or {}

            # Map local IDs → actual IDs
            base_id = max(self.steps.keys()) + 1
            sorted_local_ids = sorted(continuation_steps.keys())
            id_map = {lid: base_id + i for i, lid in enumerate(sorted_local_ids)}

            # 找 terminal nodes
            terminal_nodes = sorted(self._get_terminal_nodes())

            # 找 continuation root nodes（沒有 internal deps 的）
            norm_cont_deps = self._normalize_dependencies(continuation_dependencies)
            root_local_ids = set()
            for lid in sorted_local_ids:
                if not norm_cont_deps.get(lid, []):
                    root_local_ids.add(lid)

            # 加入每個 continuation step
            for lid in sorted_local_ids:
                actual_id = id_map[lid]
                self.steps[actual_id] = continuation_steps[lid]
                self.step_statuses[actual_id] = "not_started"
                self.step_notes[actual_id] = ""
                self.step_intents[actual_id] = continuation_intents.get(lid, "default")

                if lid in root_local_ids:
                    self.dependencies[actual_id] = terminal_nodes
                else:
                    self.dependencies[actual_id] = [id_map[d] for d in norm_cont_deps[lid]]

        # Update _next_id
        if self.steps:
            self._next_id = max(self.steps.keys()) + 1
        else:
            self._next_id = 0

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
