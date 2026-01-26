from typing import Optional, Any


# Constants for tool history
RESULT_MAX_LENGTH = 200


class Plan:
    """Represents an execution plan with steps, dependencies, and status tracking"""

    @staticmethod
    def _normalize_dependencies(deps: dict) -> dict[int, list[int]]:
        """Normalize dependencies dict to ensure all keys and values are integers.

        JSON parsing converts dict keys to strings, so we need to convert them back.
        """
        if not deps:
            return {}
        return {
            int(k): [int(v) for v in vals]
            for k, vals in deps.items()
        }

    def __init__(
        self,
        title: str = "",
        steps: list[str] = None,
        dependencies: dict[int, list[int]] = None
    ):
        self.title = title
        self.steps = steps if steps else []

        # Auto-generate sequential dependencies if not provided
        if dependencies is not None:
            self.dependencies = self._normalize_dependencies(dependencies)
        elif len(self.steps) > 1:
            self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))}
        else:
            self.dependencies = {}

        # Initialize status tracking
        self.step_statuses: dict[str, str] = {step: "not_started" for step in self.steps}
        self.step_notes: dict[str, str] = {step: "" for step in self.steps}

        # Initialize tool history and file tracking
        self.step_tool_history: dict[int, list[dict]] = {}
        self.step_files: dict[int, list[str]] = {}

        # Initialize replan tracking
        self.replan_attempts: dict[int, int] = {}

    def update(
        self,
        title: Optional[str] = None,
        steps: Optional[list[str]] = None,
        dependencies: Optional[dict[int, list[int]]] = None
    ) -> None:
        """Update plan properties"""
        if title is not None:
            self.title = title

        if steps is not None:
            self.steps = steps
            # Reinitialize status tracking for new steps
            self.step_statuses = {step: "not_started" for step in self.steps}
            self.step_notes = {step: "" for step in self.steps}

            # Reinitialize tool history and file tracking
            self.step_tool_history = {}
            self.step_files = {}

            # Auto-generate dependencies if not provided
            if dependencies is None and len(self.steps) > 1:
                self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))}

        if dependencies is not None:
            self.dependencies = self._normalize_dependencies(dependencies)

    def mark_step(
        self,
        step_index: int,
        step_status: Optional[str] = None,
        step_notes: Optional[str] = None
    ) -> None:
        """Mark a step with status and/or notes"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        step = self.steps[step_index]

        if step_status is not None:
            self.step_statuses[step] = step_status

        if step_notes is not None:
            self.step_notes[step] = step_notes

    def add_tool_call(
        self,
        step_index: int,
        tool: str,
        args: dict,
        result: Any,
        timestamp: str
    ) -> None:
        """Record a tool call for a step"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            self.step_tool_history[step_index] = []

        # Truncate result if too long
        result_str = str(result)
        if len(result_str) > RESULT_MAX_LENGTH:
            result_str = result_str[:RESULT_MAX_LENGTH] + "...[truncated]"

        self.step_tool_history[step_index].append({
            "tool": tool,
            "args": args,
            "result": result_str,
            "timestamp": timestamp
        })

    def add_tool_call_pending(
        self,
        step_index: int,
        tool: str,
        args: dict,
        call_time: str
    ) -> None:
        """Record a pending tool call for a step"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            self.step_tool_history[step_index] = []

        self.step_tool_history[step_index].append({
            "tool": tool,
            "args": args,
            "status": "pending",
            "call_time": call_time
        })

    def update_tool_result(
        self,
        step_index: int,
        tool: str,
        result: Any,
        response_time: str
    ) -> None:
        """Update a pending tool call to success with result (FIFO matching)"""
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
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_tool_history:
            return True  # No tool calls = OK

        for call in self.step_tool_history[step_index]:
            if call["status"] == "pending":
                return False  # Has pending = hallucination

        return True  # All success

    def add_file(self, step_index: int, file_path: str) -> None:
        """Record a generated file for a step"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        if step_index not in self.step_files:
            self.step_files[step_index] = []

        if file_path not in self.step_files[step_index]:
            self.step_files[step_index].append(file_path)

    def get_ready_steps(self) -> list[int]:
        """Get indices of steps ready to execute (dependencies satisfied)"""
        ready = []

        for idx, step in enumerate(self.steps):
            # Skip if already started or completed
            if self.step_statuses[step] != "not_started":
                continue

            # Check if all dependencies are completed
            deps = self.dependencies.get(idx, [])
            all_deps_done = all(
                self.step_statuses[self.steps[dep]] == "completed"
                for dep in deps
            )

            if all_deps_done:
                ready.append(idx)

        return ready

    def get_downstream_steps(self, step_idx: int) -> list[int]:
        """
        Get all downstream steps (direct + indirect dependents).

        Args:
            step_idx: The step index to find dependents for

        Returns:
            Sorted list of step indices that depend on this step
        """
        downstream = set()
        to_check = [step_idx]
        max_valid_idx = len(self.steps) - 1

        while to_check:
            current = to_check.pop()
            for idx, deps in self.dependencies.items():
                # Skip stale indices that are out of range
                if idx > max_valid_idx:
                    continue
                if current in deps and idx not in downstream:
                    downstream.add(idx)
                    to_check.append(idx)

        return sorted(downstream)

    def get_progress(self) -> dict[str, int]:
        """Get progress statistics"""
        statuses = list(self.step_statuses.values())
        return {
            "total": len(self.steps),
            "completed": statuses.count("completed"),
            "in_progress": statuses.count("in_progress"),
            "blocked": statuses.count("blocked"),
            "not_started": statuses.count("not_started")
        }

    def format(self) -> str:
        """Format plan for display"""
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
            "blocked": "[!]"
        }

        for idx, step in enumerate(self.steps):
            symbol = status_symbols.get(self.step_statuses[step], "[ ]")
            deps = self.dependencies.get(idx, [])
            dep_str = f" (depends on: {deps})" if deps else ""
            lines.append(f"  {idx}: {symbol} {step}{dep_str}")

            if self.step_notes[step]:
                lines.append(f"      Notes: {self.step_notes[step]}")

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

    def remove_steps(self, step_indices: list[int]) -> None:
        """
        Remove steps and update DAG structure.

        Args:
            step_indices: List of step indices to remove (must be sorted desc internally)
        """
        # Filter out invalid indices (out of range)
        valid_indices = [idx for idx in step_indices if 0 <= idx < len(self.steps)]
        if not valid_indices:
            return

        # Sort indices in descending order to remove from end first
        indices_to_remove = sorted(valid_indices, reverse=True)

        # Build mapping from old index to new index
        removed_set = set(valid_indices)
        index_map: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(len(self.steps)):
            if old_idx not in removed_set:
                index_map[old_idx] = new_idx
                new_idx += 1

        # Remove steps (from end to preserve indices)
        for idx in indices_to_remove:
            step = self.steps[idx]
            del self.steps[idx]
            if step in self.step_statuses:
                del self.step_statuses[step]
            if step in self.step_notes:
                del self.step_notes[step]
            if idx in self.step_tool_history:
                del self.step_tool_history[idx]
            if idx in self.step_files:
                del self.step_files[idx]

        # Update dependencies with new indices
        # When a dep points to a removed step, inherit that step's dependencies
        new_dependencies: dict[int, list[int]] = {}
        for old_idx, deps in self.dependencies.items():
            if old_idx in removed_set:
                continue

            resolved_deps: set[int] = set()
            for dep in deps:
                if dep in removed_set:
                    # Inherit the removed step's dependencies
                    inherited = self.dependencies.get(dep, [])
                    for inherited_dep in inherited:
                        if inherited_dep not in removed_set:
                            resolved_deps.add(inherited_dep)
                else:
                    resolved_deps.add(dep)

            new_idx = index_map[old_idx]
            new_deps = sorted([index_map[d] for d in resolved_deps])
            if new_deps:
                new_dependencies[new_idx] = new_deps

        self.dependencies = new_dependencies

        # Update tool history indices
        new_tool_history: dict[int, list[dict]] = {}
        for old_idx, history in list(self.step_tool_history.items()):
            if old_idx in index_map:
                new_tool_history[index_map[old_idx]] = history
        self.step_tool_history = new_tool_history

        # Update file indices
        new_files: dict[int, list[str]] = {}
        for old_idx, files in list(self.step_files.items()):
            if old_idx in index_map:
                new_files[index_map[old_idx]] = files
        self.step_files = new_files

    def add_steps(
        self,
        new_steps: list[str],
        new_dependencies: dict[int, list[int]],
        insert_after: int
    ) -> None:
        """
        Add new steps to the plan after a specified position.

        Args:
            new_steps: List of new step descriptions
            new_dependencies: Dependencies between new steps (relative indices)
            insert_after: Index after which to insert new steps
        """
        base_idx = insert_after + 1

        # Add new steps
        for i, step in enumerate(new_steps):
            self.steps.insert(base_idx + i, step)
            self.step_statuses[step] = "not_started"
            self.step_notes[step] = ""

        # Convert relative dependencies to absolute indices
        normalized_deps = self._normalize_dependencies(new_dependencies)
        for rel_idx, rel_deps in normalized_deps.items():
            abs_idx = base_idx + rel_idx
            # Map relative deps to absolute, and add connection to insert_after
            abs_deps = []
            if rel_idx == 0:
                # First new step depends on insert_after
                abs_deps.append(insert_after)
            for rel_dep in rel_deps:
                abs_deps.append(base_idx + rel_dep)
            if abs_deps:
                self.dependencies[abs_idx] = abs_deps

    def format_tool_history(self, step_indices: list[int]) -> str:
        """
        Format tool history for specified steps.

        Args:
            step_indices: List of step indices to format

        Returns:
            Formatted string with tool call details
        """
        lines = []

        for idx in step_indices:
            if idx >= len(self.steps):
                continue

            step_desc = self.steps[idx]
            lines.append(f"### Step {idx}: {step_desc}")

            if idx in self.step_tool_history and self.step_tool_history[idx]:
                lines.append("Tool calls:")
                for call in self.step_tool_history[idx]:
                    tool = call["tool"]
                    args = call.get("args", {})
                    result = call.get("result", "")
                    status = call.get("status", "unknown")

                    # Format args as key=value pairs
                    args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                    lines.append(f"- {tool}({args_str}) → {result}")
            else:
                lines.append("(No tool calls)")

            lines.append("")

        return "\n".join(lines)
