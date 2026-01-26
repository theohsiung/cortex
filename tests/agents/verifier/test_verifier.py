"""Tests for Verifier - tool call verification logic"""

import pytest
from app.task.plan import Plan
from app.agents.verifier.verifier import Verifier


class TestVerifierVerifyStep:
    """Tests for verify_step method"""

    def test_verify_pass_when_no_tool_calls(self):
        """Should pass when step has no tool calls (doesn't need tools)"""
        plan = Plan(steps=["Step A", "Step B"])
        verifier = Verifier()

        # Step 0 has no tool calls
        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_pass_when_all_success(self):
        """Should pass when all tool calls are successful"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # Add pending then update to success
        plan.add_tool_call_pending(0, "read_file", {"path": "test.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "file content", "2026-01-26T10:00:01")

        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_pass_when_multiple_all_success(self):
        """Should pass when multiple tool calls are all successful"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # Multiple successful calls
        plan.add_tool_call_pending(0, "read_file", {"path": "a.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content a", "2026-01-26T10:00:01")
        plan.add_tool_call_pending(0, "write_file", {"path": "b.py"}, "2026-01-26T10:00:02")
        plan.update_tool_result(0, "write_file", "written", "2026-01-26T10:00:03")

        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_fail_when_has_pending(self):
        """Should fail when step has pending tool calls (hallucination)"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # Add pending but don't update (simulating hallucination)
        plan.add_tool_call_pending(0, "write_file", {"path": "test.py"}, "2026-01-26T10:00:00")

        result = verifier.verify_step(plan, 0)

        assert result is False

    def test_verify_fail_when_some_pending(self):
        """Should fail when some tool calls are pending even if others succeeded"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # One success, one pending
        plan.add_tool_call_pending(0, "read_file", {"path": "a.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-26T10:00:01")
        plan.add_tool_call_pending(0, "write_file", {"path": "b.py"}, "2026-01-26T10:00:02")
        # Not updating write_file - simulating hallucination

        result = verifier.verify_step(plan, 0)

        assert result is False


class TestVerifierGetFailedCalls:
    """Tests for get_failed_calls method"""

    def test_get_failed_calls_returns_pending(self):
        """Should return list of pending (failed) tool calls"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        plan.add_tool_call_pending(0, "write_file", {"path": "test.py", "content": "..."}, "2026-01-26T10:00:00")

        failed = verifier.get_failed_calls(plan, 0)

        assert len(failed) == 1
        assert failed[0]["tool"] == "write_file"
        assert failed[0]["status"] == "pending"

    def test_get_failed_calls_empty_when_all_success(self):
        """Should return empty list when all calls succeeded"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        plan.add_tool_call_pending(0, "read_file", {"path": "test.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-26T10:00:01")

        failed = verifier.get_failed_calls(plan, 0)

        assert failed == []

    def test_get_failed_calls_empty_when_no_history(self):
        """Should return empty list when step has no tool history"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        failed = verifier.get_failed_calls(plan, 0)

        assert failed == []

    def test_get_failed_calls_returns_only_pending(self):
        """Should only return pending calls, not successful ones"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        plan.add_tool_call_pending(0, "read_file", {"path": "a.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-26T10:00:01")
        plan.add_tool_call_pending(0, "write_file", {"path": "b.py"}, "2026-01-26T10:00:02")
        plan.add_tool_call_pending(0, "run_cmd", {"cmd": "test"}, "2026-01-26T10:00:03")

        failed = verifier.get_failed_calls(plan, 0)

        assert len(failed) == 2
        tool_names = [f["tool"] for f in failed]
        assert "write_file" in tool_names
        assert "run_cmd" in tool_names
        assert "read_file" not in tool_names


class TestVerifierNotesDetection:
    """Tests for Notes-based failure detection"""

    def test_verify_fail_when_notes_has_fail_tag(self):
        """Should fail verification when Notes starts with [FAIL]"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[FAIL]: Unable to access Facebook API")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is False

    def test_verify_fail_when_notes_has_fail_tag_no_colon(self):
        """Should fail verification when Notes starts with [FAIL] (no colon)"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[FAIL] Cannot create image")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is False

    def test_verify_pass_when_notes_has_success_tag(self):
        """Should pass verification when Notes starts with [SUCCESS]"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[SUCCESS]: Task completed")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_pass_when_notes_empty(self):
        """Should pass verification when Notes is empty (backward compatible)"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_pass_when_notes_has_no_tag(self):
        """Should pass verification when Notes has no status tag (backward compatible)"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="Some notes without tag")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is True

    def test_verify_fail_notes_takes_priority_over_tool_calls(self):
        """Notes [FAIL] should fail even if no pending tool calls"""
        plan = Plan(steps=["Step A"])
        plan.add_tool_call_pending(0, "read_file", {"path": "test.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-26T10:00:01")
        plan.mark_step(0, step_notes="[FAIL]: Could not complete task")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is False

    def test_verify_fail_with_whitespace_before_tag(self):
        """Should handle whitespace before [FAIL] tag"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="  [FAIL]: Unable to access API")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result is False


class TestVerifierGetFailureReason:
    """Tests for get_failure_reason method"""

    def test_get_failure_reason_with_colon(self):
        """Should extract reason after [FAIL]:"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[FAIL]: Unable to access Facebook API")
        verifier = Verifier()

        reason = verifier.get_failure_reason(plan, 0)

        assert reason == "Unable to access Facebook API"

    def test_get_failure_reason_without_colon(self):
        """Should extract reason after [FAIL] without colon"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[FAIL] Cannot create image")
        verifier = Verifier()

        reason = verifier.get_failure_reason(plan, 0)

        assert reason == "Cannot create image"

    def test_get_failure_reason_empty_when_no_fail_tag(self):
        """Should return empty string when no [FAIL] tag"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[SUCCESS]: Task completed")
        verifier = Verifier()

        reason = verifier.get_failure_reason(plan, 0)

        assert reason == ""

    def test_get_failure_reason_empty_when_no_notes(self):
        """Should return empty string when no notes"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        reason = verifier.get_failure_reason(plan, 0)

        assert reason == ""
