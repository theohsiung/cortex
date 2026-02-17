"""Tests for Verifier - tool call verification logic"""

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.verifier.verifier import Verifier
from app.task.plan import Plan


class TestVerifierVerifyStep:
    """Tests for verify_step method"""

    def test_verify_pass_when_no_tool_calls(self):
        """Should pass when step has no tool calls (doesn't need tools)"""
        plan = Plan(steps=["Step A", "Step B"])
        verifier = Verifier()

        # Step 0 has no tool calls
        result = verifier.verify_step(plan, 0)

        assert result.passed is True

    def test_verify_pass_when_all_success(self):
        """Should pass when all tool calls are successful"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # Add pending then update to success
        plan.add_tool_call_pending(0, "read_file", {"path": "test.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "file content", "2026-01-26T10:00:01")

        result = verifier.verify_step(plan, 0)

        assert result.passed is True

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

        assert result.passed is True

    def test_verify_fail_when_has_pending(self):
        """Should fail when step has pending tool calls (hallucination)"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        # Add pending but don't update (simulating hallucination)
        plan.add_tool_call_pending(0, "write_file", {"path": "test.py"}, "2026-01-26T10:00:00")

        result = verifier.verify_step(plan, 0)

        assert result.passed is False

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

        assert result.passed is False


class TestVerifierGetFailedCalls:
    """Tests for get_failed_calls method"""

    def test_get_failed_calls_returns_pending(self):
        """Should return list of pending (failed) tool calls"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        plan.add_tool_call_pending(
            0, "write_file", {"path": "test.py", "content": "..."}, "2026-01-26T10:00:00"
        )

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

        assert result.passed is False

    def test_verify_fail_when_notes_has_fail_tag_no_colon(self):
        """Should fail verification when Notes starts with [FAIL] (no colon)"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[FAIL] Cannot create image")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is False

    def test_verify_pass_when_notes_has_success_tag(self):
        """Should pass verification when Notes starts with [SUCCESS]"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="[SUCCESS]: Task completed")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is True

    def test_verify_pass_when_notes_empty(self):
        """Should pass verification when Notes is empty (backward compatible)"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is True

    def test_verify_pass_when_notes_has_no_tag(self):
        """Should pass verification when Notes has no status tag (backward compatible)"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="Some notes without tag")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is True

    def test_verify_fail_notes_takes_priority_over_tool_calls(self):
        """Notes [FAIL] should fail even if no pending tool calls"""
        plan = Plan(steps=["Step A"])
        plan.add_tool_call_pending(0, "read_file", {"path": "test.py"}, "2026-01-26T10:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-26T10:00:01")
        plan.mark_step(0, step_notes="[FAIL]: Could not complete task")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is False

    def test_verify_fail_with_whitespace_before_tag(self):
        """Should handle whitespace before [FAIL] tag"""
        plan = Plan(steps=["Step A"])
        plan.mark_step(0, step_notes="  [FAIL]: Unable to access API")
        verifier = Verifier()

        result = verifier.verify_step(plan, 0)

        assert result.passed is False


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


class TestVerifierLLMBased:
    """Tests for LLM-based verification"""

    def test_verify_result_dataclass(self):
        """VerifyResult should have passed and notes fields"""
        from app.agents.verifier.verifier import VerifyResult

        result = VerifyResult(passed=True, notes="[SUCCESS]: Done")
        assert result.passed is True
        assert result.notes == "[SUCCESS]: Done"

    def test_verify_step_returns_verify_result(self):
        """verify_step should return VerifyResult"""
        from app.agents.verifier.verifier import VerifyResult

        plan = Plan(steps=["Step A"])
        verifier = Verifier()
        result = verifier.verify_step(plan, 0)
        assert isinstance(result, VerifyResult)

    def test_verify_step_pending_calls_fail(self):
        """Should fail when pending tool calls exist (hallucination)"""
        plan = Plan(steps=["Step A"])
        plan.add_tool_call_pending(0, "write_file", {"path": "x.py"}, "2026-01-01T00:00:00")
        verifier = Verifier()
        result = verifier.verify_step(plan, 0)
        assert result.passed is False

    def test_verify_step_all_calls_success_passes(self):
        """Should pass when all tool calls succeeded"""
        plan = Plan(steps=["Step A"])
        plan.add_tool_call_pending(0, "read_file", {"path": "x.py"}, "2026-01-01T00:00:00")
        plan.update_tool_result(0, "read_file", "content", "2026-01-01T00:00:01")
        verifier = Verifier()
        result = verifier.verify_step(plan, 0)
        assert result.passed is True

    def test_verify_step_no_calls_passes(self):
        """Should pass when no tool calls were made"""
        plan = Plan(steps=["Step A"])
        verifier = Verifier()
        result = verifier.verify_step(plan, 0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_output_success(self):
        """evaluate_output should detect success from executor output"""
        from app.agents.verifier.verifier import VerifyResult

        verifier = Verifier(model=Mock())
        # Mock the LLM call to return success
        verifier._llm_evaluate = AsyncMock(
            return_value=VerifyResult(passed=True, notes="[SUCCESS]: Code generated successfully")
        )
        result = await verifier.evaluate_output(
            step_description="Generate parser code",
            executor_output="Here is the parser code:\ndef parse(text): ...",
        )
        assert result.passed is True
        assert "[SUCCESS]" in result.notes

    @pytest.mark.asyncio
    async def test_evaluate_output_failure(self):
        """evaluate_output should detect failure from executor output"""
        from app.agents.verifier.verifier import VerifyResult

        verifier = Verifier(model=Mock())
        verifier._llm_evaluate = AsyncMock(
            return_value=VerifyResult(passed=False, notes="[FAIL]: No code was generated")
        )
        result = await verifier.evaluate_output(
            step_description="Generate parser code",
            executor_output="A parser typically works by...",
        )
        assert result.passed is False
        assert "[FAIL]" in result.notes

    @pytest.mark.asyncio
    async def test_evaluate_output_without_model(self):
        """evaluate_output without model should return pass with generic notes"""
        verifier = Verifier()  # No model
        result = await verifier.evaluate_output(
            step_description="Do something", executor_output="Done"
        )
        assert result.passed is True


class TestVerifierEvaluationPrompt:
    """Tests for the LLM evaluation prompt content."""

    def test_prompt_instructs_lenient_success_criteria(self):
        """Prompt should tell LLM to SUCCESS when output is relevant, even without tools."""
        verifier = Verifier()
        prompt = verifier.build_evaluation_prompt("Analyze input", "The input looks correct")
        assert "relevant" in prompt.lower() or "addresses" in prompt.lower()

    def test_prompt_instructs_fail_only_on_clear_issues(self):
        """Prompt should tell LLM to only FAIL on empty, irrelevant, or erroneous output."""
        verifier = Verifier()
        prompt = verifier.build_evaluation_prompt("Analyze input", "The input looks correct")
        assert "empty" in prompt.lower() or "irrelevant" in prompt.lower()
