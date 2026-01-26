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


class TestPlanToolCallWithStatus:
    """Tests for new tool call tracking with pending/success status"""

    def test_add_tool_call_pending(self):
        """Should record tool call with pending status"""
        plan = Plan(steps=["A", "B"])

        plan.add_tool_call_pending(
            step_index=0,
            tool="write_file",
            args={"path": "main.py", "content": "..."},
            call_time="2026-01-26T10:00:00"
        )

        assert 0 in plan.step_tool_history
        assert len(plan.step_tool_history[0]) == 1
        call = plan.step_tool_history[0][0]
        assert call["tool"] == "write_file"
        assert call["args"] == {"path": "main.py", "content": "..."}
        assert call["status"] == "pending"
        assert call["call_time"] == "2026-01-26T10:00:00"
        assert "result" not in call
        assert "response_time" not in call

    def test_add_tool_call_pending_multiple(self):
        """Should record multiple pending calls for same step"""
        plan = Plan(steps=["A"])

        plan.add_tool_call_pending(0, "tool_a", {"arg": 1}, "ts1")
        plan.add_tool_call_pending(0, "tool_b", {"arg": 2}, "ts2")

        assert len(plan.step_tool_history[0]) == 2
        assert plan.step_tool_history[0][0]["tool"] == "tool_a"
        assert plan.step_tool_history[0][1]["tool"] == "tool_b"

    def test_update_tool_result_success(self):
        """Should update pending call to success with result"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "write_file", {"path": "x.py"}, "2026-01-26T10:00:00")

        plan.update_tool_result(
            step_index=0,
            tool="write_file",
            result="File written successfully",
            response_time="2026-01-26T10:00:01"
        )

        call = plan.step_tool_history[0][0]
        assert call["status"] == "success"
        assert call["result"] == "File written successfully"
        assert call["response_time"] == "2026-01-26T10:00:01"

    def test_update_tool_result_fifo_matching(self):
        """Should match first pending call with same tool name (FIFO)"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "write_file", {"path": "a.py"}, "ts1")
        plan.add_tool_call_pending(0, "write_file", {"path": "b.py"}, "ts2")

        plan.update_tool_result(0, "write_file", "Result A", "ts1_resp")

        # First call should be updated
        assert plan.step_tool_history[0][0]["status"] == "success"
        assert plan.step_tool_history[0][0]["result"] == "Result A"
        # Second call still pending
        assert plan.step_tool_history[0][1]["status"] == "pending"

    def test_update_tool_result_truncates_long_result(self):
        """Should truncate result exceeding max length"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "tool", {}, "ts")

        long_result = "x" * 300
        plan.update_tool_result(0, "tool", long_result, "ts_resp")

        result = plan.step_tool_history[0][0]["result"]
        assert len(result) < 300
        assert result.endswith("...[truncated]")

    def test_update_tool_result_no_matching_pending(self):
        """Should handle case when no matching pending call exists"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "tool_a", {}, "ts")

        # Try to update a different tool - should not crash
        plan.update_tool_result(0, "tool_b", "result", "ts_resp")

        # Original call still pending
        assert plan.step_tool_history[0][0]["status"] == "pending"

    def test_finalize_step_all_success(self):
        """Should return True when all tool calls are successful"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "tool", {}, "ts")
        plan.update_tool_result(0, "tool", "done", "ts_resp")

        result = plan.finalize_step(0)

        assert result is True

    def test_finalize_step_has_pending(self):
        """Should return False when there are pending tool calls"""
        plan = Plan(steps=["A"])
        plan.add_tool_call_pending(0, "tool_a", {}, "ts1")
        plan.add_tool_call_pending(0, "tool_b", {}, "ts2")
        plan.update_tool_result(0, "tool_a", "done", "ts1_resp")
        # tool_b is still pending

        result = plan.finalize_step(0)

        assert result is False

    def test_finalize_step_no_tool_calls(self):
        """Should return True when step has no tool calls"""
        plan = Plan(steps=["A"])

        result = plan.finalize_step(0)

        assert result is True

    def test_finalize_step_invalid_index(self):
        """Should raise error for invalid step index"""
        plan = Plan(steps=["A"])

        with pytest.raises(ValueError):
            plan.finalize_step(5)


class TestPlanDownstreamSteps:
    """Tests for get_downstream_steps method"""

    def test_get_downstream_steps_no_dependents(self):
        """Should return empty list when step has no dependents"""
        # 0 → 1 → 2
        plan = Plan(steps=["A", "B", "C"])  # default sequential deps

        downstream = plan.get_downstream_steps(2)  # Last step

        assert downstream == []

    def test_get_downstream_steps_direct_dependency(self):
        """Should return directly dependent steps"""
        # 0 → 1 → 2
        plan = Plan(steps=["A", "B", "C"])

        downstream = plan.get_downstream_steps(0)

        assert 1 in downstream
        assert 2 in downstream

    def test_get_downstream_steps_indirect_dependency(self):
        """Should return indirectly dependent steps"""
        # 0 → 1 → 2 → 3
        plan = Plan(steps=["A", "B", "C", "D"])

        downstream = plan.get_downstream_steps(0)

        assert set(downstream) == {1, 2, 3}

    def test_get_downstream_steps_parallel_branches(self):
        """Should return all downstream in parallel DAG"""
        # 0 → 1
        #   ↘ 2
        plan = Plan(
            steps=["A", "B", "C"],
            dependencies={1: [0], 2: [0]}
        )

        downstream = plan.get_downstream_steps(0)

        assert set(downstream) == {1, 2}

    def test_get_downstream_steps_complex_dag(self):
        """Should handle complex DAG with multiple paths"""
        # Example from design doc: {1:[0], 2:[0], 3:[1], 4:[2], 5:[3,4], 6:[5], 7:[5]}
        # 0 → 1 → 3 ─┐
        #   ↘ 2 → 4 ─┴→ 5 → 6
        #               ↘ 7
        plan = Plan(
            steps=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            dependencies={1: [0], 2: [0], 3: [1], 4: [2], 5: [3, 4], 6: [5], 7: [5]}
        )

        # Step 5 should have 6, 7 as downstream
        downstream_5 = plan.get_downstream_steps(5)
        assert set(downstream_5) == {6, 7}

        # Step 0 should have all others as downstream
        downstream_0 = plan.get_downstream_steps(0)
        assert set(downstream_0) == {1, 2, 3, 4, 5, 6, 7}

        # Step 3 should have 5, 6, 7 as downstream
        downstream_3 = plan.get_downstream_steps(3)
        assert set(downstream_3) == {5, 6, 7}

    def test_get_downstream_steps_returns_sorted(self):
        """Should return downstream steps in sorted order"""
        plan = Plan(
            steps=["A", "B", "C", "D"],
            dependencies={1: [0], 2: [0], 3: [0]}
        )

        downstream = plan.get_downstream_steps(0)

        assert downstream == sorted(downstream)


class TestPlanDAGOperations:
    """Tests for remove_steps and add_steps methods"""

    def test_remove_steps_single(self):
        """Should remove a single step and update indices"""
        # 0 → 1 → 2
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")

        plan.remove_steps([1])

        assert plan.steps == ["A", "C"]
        # C (now index 1) should have dependency updated
        assert plan.dependencies == {1: [0]}

    def test_remove_steps_multiple(self):
        """Should remove multiple steps"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3", "S4"],
            dependencies={1: [0], 2: [0], 3: [1, 2], 4: [3]}
        )
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="completed")
        plan.mark_step(2, step_status="completed")

        # Remove steps 3, 4 (the subgraph after completed steps)
        plan.remove_steps([3, 4])

        assert plan.steps == ["S0", "S1", "S2"]
        assert 3 not in plan.dependencies
        assert 4 not in plan.dependencies

    def test_remove_steps_preserves_completed_status(self):
        """Should preserve status of remaining steps"""
        plan = Plan(steps=["A", "B", "C", "D"])
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="completed")

        plan.remove_steps([2, 3])

        assert plan.step_statuses["A"] == "completed"
        assert plan.step_statuses["B"] == "completed"

    def test_remove_steps_clears_tool_history(self):
        """Should clear tool history for removed steps"""
        plan = Plan(steps=["A", "B", "C"])
        plan.add_tool_call(0, "tool", {}, "result", "ts")
        plan.add_tool_call(1, "tool", {}, "result", "ts")

        plan.remove_steps([1])

        assert 0 in plan.step_tool_history
        assert 1 not in plan.step_tool_history

    def test_add_steps_basic(self):
        """Should add new steps after specified position"""
        plan = Plan(steps=["A", "B"])
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="completed")

        plan.add_steps(
            new_steps=["C", "D"],
            new_dependencies={0: [], 1: [0]},  # Relative to new steps
            insert_after=1
        )

        assert plan.steps == ["A", "B", "C", "D"]
        # New step C (index 2) should depend on last completed (index 1)
        # New step D (index 3) should depend on C (index 2)
        assert 2 in plan.dependencies
        assert 3 in plan.dependencies
        assert plan.dependencies[3] == [2]

    def test_add_steps_initializes_status(self):
        """Should initialize new steps as not_started"""
        plan = Plan(steps=["A"])
        plan.mark_step(0, step_status="completed")

        plan.add_steps(["B", "C"], {0: [], 1: [0]}, insert_after=0)

        assert plan.step_statuses["B"] == "not_started"
        assert plan.step_statuses["C"] == "not_started"

    def test_add_steps_connects_to_completed(self):
        """Should connect first new step to last completed step"""
        plan = Plan(
            steps=["S0", "S1", "S2"],
            dependencies={1: [0], 2: [1]}
        )
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="completed")

        # Replace S2 with new steps
        plan.remove_steps([2])
        plan.add_steps(
            new_steps=["New1", "New2"],
            new_dependencies={0: [], 1: [0]},
            insert_after=1
        )

        # New1 (index 2) should depend on S1 (index 1)
        assert 1 in plan.dependencies.get(2, [])


class TestPlanFormatToolHistory:
    """Tests for format_tool_history method"""

    def test_format_tool_history_single_step(self):
        """Should format tool history for a single step"""
        plan = Plan(steps=["A", "B"])
        plan.add_tool_call(0, "write_file", {"path": "x.py"}, "done", "ts")

        output = plan.format_tool_history([0])

        assert "Step 0" in output
        assert "write_file" in output
        assert "path" in output

    def test_format_tool_history_multiple_steps(self):
        """Should format tool history for multiple steps"""
        plan = Plan(steps=["A", "B", "C"])
        plan.add_tool_call(0, "tool_a", {}, "result_a", "ts")
        plan.add_tool_call(1, "tool_b", {}, "result_b", "ts")

        output = plan.format_tool_history([0, 1])

        assert "Step 0" in output
        assert "Step 1" in output
        assert "tool_a" in output
        assert "tool_b" in output

    def test_format_tool_history_empty_step(self):
        """Should handle step with no tool calls"""
        plan = Plan(steps=["A", "B"])
        plan.add_tool_call(0, "tool", {}, "result", "ts")
        # Step 1 has no tool calls

        output = plan.format_tool_history([0, 1])

        assert "Step 0" in output
        assert "Step 1" in output

    def test_format_tool_history_includes_result(self):
        """Should include tool call results"""
        plan = Plan(steps=["A"])
        plan.add_tool_call(0, "run_command", {"cmd": "ls"}, "file1.py\nfile2.py", "ts")

        output = plan.format_tool_history([0])

        assert "file1.py" in output or "file2.py" in output


class TestPlanReplanAttempts:
    """Tests for replan_attempts tracking"""

    def test_replan_attempts_initialized(self):
        """Should initialize replan_attempts as empty dict"""
        plan = Plan(steps=["A", "B"])

        assert plan.replan_attempts == {}

    def test_replan_attempts_increment(self):
        """Should track replan attempts per step"""
        plan = Plan(steps=["A", "B"])

        plan.replan_attempts[0] = 1
        plan.replan_attempts[0] += 1

        assert plan.replan_attempts[0] == 2
        assert 1 not in plan.replan_attempts
