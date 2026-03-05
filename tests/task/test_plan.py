import pytest

from app.task.plan import Plan


class TestPlan:
    def test_create_empty_plan(self):
        """Should create plan with empty defaults"""
        plan = Plan()

        assert plan.title == ""
        assert plan.steps == {}
        assert plan.dependencies == {}

    def test_create_plan_with_title_and_steps(self):
        """Should create plan with provided values"""
        plan = Plan(title="Test Plan", steps=["Step 1", "Step 2", "Step 3"])

        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        assert plan.steps == {0: "Step 1", 1: "Step 2", 2: "Step 3"}
        # Default sequential dependencies
        assert plan.dependencies == {1: [0], 2: [1]}

    def test_create_plan_with_custom_dependencies(self):
        """Should use custom dependencies when provided"""
        plan = Plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]},  # C depends on both A and B
        )

        assert plan.dependencies == {2: [0, 1]}

    def test_create_plan_normalizes_string_keys(self):
        """Should convert string keys to integers (JSON parsing produces strings)"""
        # Simulate JSON-parsed dependencies (keys are strings)
        plan = Plan(title="Test", steps=["A", "B", "C"], dependencies={"1": ["0"], "2": ["1"]})

        # Keys and values should be normalized to integers
        assert plan.dependencies == {1: [0], 2: [1]}
        assert all(isinstance(k, int) for k in plan.dependencies.keys())
        assert all(isinstance(v, int) for vals in plan.dependencies.values() for v in vals)

    def test_update_plan_normalizes_string_keys(self):
        """Should normalize string keys on update"""
        plan = Plan(steps=["A", "B"])

        plan.update(dependencies={"1": ["0"]})

        assert plan.dependencies == {1: [0]}

    def test_update_plan(self):
        """Should update plan properties"""
        plan = Plan()

        plan.update(title="Updated", steps=["New Step 1", "New Step 2"], dependencies={1: [0]})

        assert plan.title == "Updated"
        assert plan.steps == {0: "New Step 1", 1: "New Step 2"}

    def test_step_statuses_initialized(self):
        """Should initialize all steps as not_started"""
        plan = Plan(steps=["A", "B", "C"])

        assert plan.step_statuses[0] == "not_started"
        assert plan.step_statuses[1] == "not_started"
        assert plan.step_statuses[2] == "not_started"

    def test_mark_step_status(self):
        """Should update step status"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="in_progress")

        assert plan.step_statuses[0] == "in_progress"

    def test_mark_step_with_notes(self):
        """Should store step notes"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="completed", step_notes="Done successfully")

        assert plan.step_notes[0] == "Done successfully"

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
            dependencies={2: [0], 3: [1]},  # C depends on A, D depends on B
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
            timestamp="2026-01-20T10:00:00",
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
            timestamp="2026-01-20T10:00:00",
        )

        result_str = plan.step_tool_history[0][0]["result"]
        assert len(result_str) < 300
        assert result_str.endswith("...[truncated]")

    def test_add_tool_call_invalid_step(self):
        """Should raise error for invalid step index"""
        plan = Plan(steps=["A"])

        with pytest.raises(ValueError):
            plan.add_tool_call(5, "tool", {}, {}, "ts")  # step_index not in steps

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
            call_time="2026-01-26T10:00:00",
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
            response_time="2026-01-26T10:00:01",
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


class TestPlanIntents:
    """Tests for step intent tracking"""

    def test_default_intents_are_default(self):
        """All steps should default to 'default' intent"""
        plan = Plan(steps=["Step A", "Step B"])
        assert plan.step_intents == {0: "default", 1: "default"}

    def test_custom_intents(self):
        """Should accept custom intents"""
        plan = Plan(
            steps=["Generate code", "Review code"], step_intents={0: "generate", 1: "review"}
        )
        assert plan.step_intents[0] == "generate"
        assert plan.step_intents[1] == "review"

    def test_partial_intents_fills_default(self):
        """Missing intents should default to 'default'"""
        plan = Plan(steps=["Step A", "Step B", "Step C"], step_intents={1: "generate"})
        assert plan.step_intents[0] == "default"
        assert plan.step_intents[1] == "generate"
        assert plan.step_intents[2] == "default"

    def test_update_with_intents(self):
        """update() should accept and set intents"""
        plan = Plan(steps=["Old step"])
        plan.update(steps=["New A", "New B"], step_intents={0: "generate", 1: "review"})
        assert plan.step_intents == {0: "generate", 1: "review"}

    def test_update_without_intents_resets_to_default(self):
        """update() with new steps but no intents should default all to 'default'"""
        plan = Plan(steps=["A"], step_intents={0: "generate"})
        plan.update(steps=["B", "C"])
        assert plan.step_intents == {0: "default", 1: "default"}

    def test_get_step_intent(self):
        """get_step_intent() should return intent for a step"""
        plan = Plan(steps=["A", "B"], step_intents={0: "generate", 1: "review"})
        assert plan.get_step_intent(0) == "generate"
        assert plan.get_step_intent(1) == "review"

    def test_get_step_intent_default_fallback(self):
        """get_step_intent() should return 'default' for missing index"""
        plan = Plan(steps=["A"])
        assert plan.get_step_intent(0) == "default"


class TestPlanGetDownstream:
    """Tests for _get_downstream method"""

    def test_linear_chain(self):
        """Should find all downstream steps in a linear chain"""
        plan = Plan(steps=["S0", "S1", "S2", "S3"], dependencies={1: [0], 2: [1], 3: [2]})

        assert plan._get_downstream(0) == {1, 2, 3}
        assert plan._get_downstream(1) == {2, 3}
        assert plan._get_downstream(3) == set()

    def test_diamond_dag(self):
        """Should find all downstream in a diamond DAG: 0->{1,2}->3"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [0], 3: [1, 2]},
        )

        assert plan._get_downstream(0) == {1, 2, 3}
        assert plan._get_downstream(1) == {3}

    def test_no_downstream(self):
        """Leaf node should have empty downstream set"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})

        assert plan._get_downstream(2) == set()

    def test_branching_dag(self):
        """Should handle branching: 0->{1,2}, 1->3"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [0], 3: [1]},
        )

        assert plan._get_downstream(0) == {1, 2, 3}
        assert plan._get_downstream(2) == set()


class TestPlanGetTerminalNodes:
    """Tests for _get_terminal_nodes method"""

    def test_linear_chain(self):
        """Terminal node is the last step in a linear chain"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})

        assert plan._get_terminal_nodes() == {2}

    def test_branching_dag(self):
        """Multiple terminal nodes in branching DAG"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [0], 3: [1]},
        )

        assert plan._get_terminal_nodes() == {2, 3}

    def test_after_deletion(self):
        """Terminal nodes should change after manually deleting steps"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [1], 3: [2]},
        )

        assert plan._get_terminal_nodes() == {3}

        # Manually delete step 3
        del plan.steps[3]
        del plan.step_statuses[3]
        del plan.dependencies[3]

        assert plan._get_terminal_nodes() == {2}

    def test_single_step(self):
        """Single step is terminal"""
        plan = Plan(steps=["S0"])

        assert plan._get_terminal_nodes() == {0}


class TestPlanApplyReplan:
    """Tests for apply_replan method"""

    def test_basic_no_continuation(self):
        """Should reset failed step and delete downstream without continuation"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="failed")

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry S1",
            new_intent="fix",
        )

        # Failed step reset
        assert plan.steps[1] == "Retry S1"
        assert plan.step_statuses[1] == "not_started"
        assert plan.step_intents[1] == "fix"
        # Downstream deleted
        assert 2 not in plan.steps
        # Completed step untouched
        assert plan.steps[0] == "S0"
        assert plan.step_statuses[0] == "completed"

    def test_with_continuation(self):
        """Should add continuation steps with proper ID mapping"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="failed")

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry S1",
            new_intent="fix",
            continuation_steps={0: "Cont A", 1: "Cont B"},
            continuation_dependencies={1: [0]},
            continuation_intents={0: "generate", 1: "review"},
        )

        # Failed step reset
        assert plan.steps[1] == "Retry S1"
        assert plan.step_statuses[1] == "not_started"
        # Original step 2 was downstream and deleted; continuation steps now occupy IDs 2 and 3
        # (max key after deleting downstream and resetting is 1, so base = 2)
        # local 0 -> 2, local 1 -> 3
        assert plan.steps[2] == "Cont A"
        assert plan.steps[3] == "Cont B"
        assert plan.step_intents[2] == "generate"
        assert plan.step_intents[3] == "review"
        # Root cont step (local 0 -> actual 2) depends on terminal node (1)
        assert set(plan.dependencies[2]) == {1}
        # Non-root cont step (local 1 -> actual 3) depends on remapped local 0 -> actual 2
        assert plan.dependencies[3] == [2]

    def test_preserves_completed(self):
        """Completed steps should be completely untouched"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed", step_notes="All good")
        plan.add_tool_call(0, "tool", {}, "result", "ts")
        plan.add_file(0, "/file.txt")
        plan.mark_step(1, step_status="failed")

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry S1",
            new_intent="default",
        )

        assert plan.steps[0] == "S0"
        assert plan.step_statuses[0] == "completed"
        assert plan.step_notes[0] == "All good"
        assert 0 in plan.step_tool_history
        assert 0 in plan.step_files

    def test_clears_downstream(self):
        """Downstream steps should be fully cleaned from all dicts"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [1], 3: [2]},
        )
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="failed")
        plan.add_tool_call(2, "tool", {}, "res", "ts")
        plan.add_file(2, "/some.txt")
        plan.step_notes[2] = "note"

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry S1",
            new_intent="default",
        )

        for sid in [2, 3]:
            assert sid not in plan.steps
            assert sid not in plan.step_statuses
            assert sid not in plan.step_notes
            assert sid not in plan.step_tool_history
            assert sid not in plan.step_files
            assert sid not in plan.step_intents
            assert sid not in plan.dependencies

    def test_clears_failed_step_tool_history(self):
        """Failed step's tool history and files should be cleared on reset"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="failed")
        plan.add_tool_call(1, "run_python", {"code": "x"}, "err", "ts")
        plan.add_file(1, "/output.txt")

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry S1",
            new_intent="default",
        )

        assert 1 not in plan.step_tool_history
        assert 1 not in plan.step_files

    def test_updates_next_id(self):
        """_next_id should be max(steps.keys()) + 1 after replan"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="failed")

        plan.apply_replan(
            failed_step_id=1,
            new_description="Retry",
            new_intent="default",
            continuation_steps={0: "Cont A"},
            continuation_dependencies={},
        )

        # After replan: steps 0, 1, 2 (mapped cont). max = 2, so _next_id = 3
        assert plan._next_id == max(plan.steps.keys()) + 1

    def test_terminal_node_auto_connect(self):
        """Continuation root steps should depend on ALL terminal nodes"""
        plan = Plan(
            steps=["S0", "S1", "S2", "S3"],
            dependencies={1: [0], 2: [0], 3: [1]},
        )
        # Complete steps 0, 1, 2, 3
        for sid in [0, 1, 2, 3]:
            plan.mark_step(sid, step_status="completed")
        # Now fail step 3
        plan.mark_step(3, step_status="failed")

        plan.apply_replan(
            failed_step_id=3,
            new_description="Retry S3",
            new_intent="default",
            continuation_steps={0: "Final"},
            continuation_dependencies={},
        )

        # Terminal nodes after reset: 2 (leaf, not depended on) and 3 (reset, not depended on)
        # Continuation root (local 0 -> actual 4) should depend on all terminals
        assert set(plan.dependencies[4]) == {2, 3}

    def test_spec_example(self):
        """Full example from the design spec"""
        # Initial: steps 0-7
        plan = Plan(
            steps=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            dependencies={
                1: [0],
                2: [1],
                3: [2],
                4: [3],
                5: [3],
                6: [5],
                7: [4],
            },
        )

        # Completed: {0,1,2,3,4,7}, step 5 fails
        for sid in [0, 1, 2, 3, 4, 7]:
            plan.mark_step(sid, step_status="completed")
        plan.mark_step(5, step_status="failed")

        plan.apply_replan(
            failed_step_id=5,
            new_description="Retry S5",
            new_intent="fix",
            continuation_steps={0: "Next action", 1: "Final merge"},
            continuation_dependencies={1: [0]},
        )

        # Step 6 deleted (downstream of 5)
        assert 6 not in plan.steps

        # Step 5 reset
        assert plan.steps[5] == "Retry S5"
        assert plan.step_statuses[5] == "not_started"
        assert plan.step_intents[5] == "fix"

        # Local 0 -> 8, local 1 -> 9 (max key was 7 before cont, so base = 8)
        assert plan.steps[8] == "Next action"
        assert plan.steps[9] == "Final merge"

        # Terminal nodes after deleting 6 and resetting 5: {5, 7}
        # (4 is depended on by 7, 3 by 4 and 5, etc.)
        # Root continuation (local 0 -> 8) depends on all terminals
        assert set(plan.dependencies[8]) == {5, 7}
        # Non-root (local 1 -> 9) depends on remapped local 0 -> 8
        assert plan.dependencies[9] == [8]

        # _next_id updated
        assert plan._next_id == 10


class TestPlanGlobalReplanCount:
    """Tests for global_replan_count attribute"""

    def test_global_replan_count_initialized(self):
        """Should initialize global_replan_count to 0"""
        plan = Plan(steps=["A"])

        assert plan.global_replan_count == 0

    def test_global_replan_count_increment(self):
        """Should be incrementable"""
        plan = Plan(steps=["A"])

        plan.global_replan_count += 1
        assert plan.global_replan_count == 1

        plan.global_replan_count += 1
        assert plan.global_replan_count == 2

    def test_replanned_steps_initialized_empty(self):
        """New plan should have empty replanned_steps set"""
        plan = Plan(steps=["A", "B"])
        assert plan.replanned_steps == set()

    def test_apply_replan_tracks_replanned_steps(self):
        """apply_replan should add failed_step_id to replanned_steps"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")

        plan.apply_replan(failed_step_id=1, new_description="Retry S1", new_intent="default")

        assert 1 in plan.replanned_steps
