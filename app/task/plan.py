from typing import Optional, Any


# Constants for tool history
RESULT_MAX_LENGTH = 200


class Plan:
    """Represents an execution plan with steps, dependencies, and status tracking"""

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
            self.dependencies = dependencies
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
            self.dependencies = dependencies

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
