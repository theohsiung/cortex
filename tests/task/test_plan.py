import pytest
from app.task.plan import Plan


class TestPlan:
    def test_create_empty_plan(self):
        """Should create plan with empty defaults"""
        plan = Plan()

        assert plan.title == ""
        assert plan.steps == []
        assert plan.dependencies == {}

    def test_create_plan_with_title_and_steps(self):
        """Should create plan with provided values"""
        plan = Plan(
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"]
        )

        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        # Default sequential dependencies
        assert plan.dependencies == {1: [0], 2: [1]}

    def test_create_plan_with_custom_dependencies(self):
        """Should use custom dependencies when provided"""
        plan = Plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]}  # C depends on both A and B
        )

        assert plan.dependencies == {2: [0, 1]}

    def test_update_plan(self):
        """Should update plan properties"""
        plan = Plan()

        plan.update(
            title="Updated",
            steps=["New Step 1", "New Step 2"],
            dependencies={1: [0]}
        )

        assert plan.title == "Updated"
        assert plan.steps == ["New Step 1", "New Step 2"]

    def test_step_statuses_initialized(self):
        """Should initialize all steps as not_started"""
        plan = Plan(steps=["A", "B", "C"])

        assert plan.step_statuses["A"] == "not_started"
        assert plan.step_statuses["B"] == "not_started"
        assert plan.step_statuses["C"] == "not_started"

    def test_mark_step_status(self):
        """Should update step status"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="in_progress")

        assert plan.step_statuses["A"] == "in_progress"

    def test_mark_step_with_notes(self):
        """Should store step notes"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="completed", step_notes="Done successfully")

        assert plan.step_notes["A"] == "Done successfully"

    def test_get_ready_steps_initial(self):
        """Should return first step when no dependencies completed"""
        plan = Plan(steps=["A", "B", "C"])

        ready = plan.get_ready_steps()

        assert ready == [0]  # Only first step is ready

    def test_get_ready_steps_after_completion(self):
        """Should return next steps after dependencies complete"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")

        ready = plan.get_ready_steps()

        assert ready == [1]  # Second step is now ready

    def test_get_ready_steps_parallel(self):
        """Should return multiple ready steps for parallel execution"""
        plan = Plan(
            steps=["A", "B", "C", "D"],
            dependencies={2: [0], 3: [1]}  # C depends on A, D depends on B
        )

        ready = plan.get_ready_steps()

        assert set(ready) == {0, 1}  # A and B can run in parallel

    def test_get_progress(self):
        """Should return progress statistics"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="in_progress")

        progress = plan.get_progress()

        assert progress["total"] == 3
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["not_started"] == 1

    def test_format_plan(self):
        """Should return formatted string representation"""
        plan = Plan(title="Test", steps=["A", "B"])

        output = plan.format()

        assert "Test" in output
        assert "A" in output
        assert "B" in output


class TestPlanToolHistory:
    def test_add_tool_call(self):
        """Should record tool call for a step"""
        plan = Plan(steps=["A", "B"])

        plan.add_tool_call(
            step_index=0,
            tool="run_python",
            args={"code": "print('hello')"},
            result={"exit_code": 0, "stdout": "hello\n"},
            timestamp="2026-01-20T10:00:00"
        )

        assert 0 in plan.step_tool_history
        assert len(plan.step_tool_history[0]) == 1
        assert plan.step_tool_history[0][0]["tool"] == "run_python"

    def test_add_tool_call_truncates_long_result(self):
        """Should truncate result exceeding max length"""
        plan = Plan(steps=["A"])

        long_result = "x" * 300
        plan.add_tool_call(
            step_index=0,
            tool="test_tool",
            args={},
            result=long_result,
            timestamp="2026-01-20T10:00:00"
        )

        result_str = plan.step_tool_history[0][0]["result"]
        assert len(result_str) < 300
        assert result_str.endswith("...[truncated]")

    def test_add_tool_call_invalid_step(self):
        """Should raise error for invalid step index"""
        plan = Plan(steps=["A"])

        with pytest.raises(ValueError):
            plan.add_tool_call(5, "tool", {}, {}, "ts")  # step_index out of range

    def test_add_file(self):
        """Should record file for a step"""
        plan = Plan(steps=["A", "B"])

        plan.add_file(0, "/workspace/output.txt")
        plan.add_file(0, "/workspace/data.csv")

        assert 0 in plan.step_files
        assert len(plan.step_files[0]) == 2
        assert "/workspace/output.txt" in plan.step_files[0]

    def test_add_file_no_duplicates(self):
        """Should not add duplicate file paths"""
        plan = Plan(steps=["A"])

        plan.add_file(0, "/workspace/output.txt")
        plan.add_file(0, "/workspace/output.txt")

        assert len(plan.step_files[0]) == 1

    def test_add_file_invalid_step(self):
        """Should raise error for invalid step index"""
        plan = Plan(steps=["A"])

        with pytest.raises(ValueError):
            plan.add_file(5, "/workspace/file.txt")

    def test_format_includes_tools_and_files(self):
        """Should include tool and file info in format output"""
        plan = Plan(title="Test", steps=["A", "B"])
        plan.add_tool_call(0, "run_python", {}, {}, "ts")
        plan.add_tool_call(0, "run_python", {}, {}, "ts")
        plan.add_file(0, "/workspace/out.txt")

        output = plan.format()

        assert "Tools: run_python (2)" in output
        assert "Files: /workspace/out.txt" in output

    def test_tool_history_reset_on_update_steps(self):
        """Should reset tool history when steps are updated"""
        plan = Plan(steps=["A"])
        plan.add_tool_call(0, "tool", {}, {}, "ts")
        plan.add_file(0, "/file.txt")

        plan.update(steps=["New A", "New B"])

        assert plan.step_tool_history == {}
        assert plan.step_files == {}
