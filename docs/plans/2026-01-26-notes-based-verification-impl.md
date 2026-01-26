# Notes-Based Step Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect LLM-reported failures via structured `[SUCCESS]/[FAIL]` format in Notes field.

**Architecture:** Extend Verifier to check Notes prefix before tool call verification. Update Executor prompt to require status tags. Defer replan to batch-end.

**Tech Stack:** Python, pytest

---

## Task 1: Verifier Notes Detection

**Files:**
- Modify: `app/agents/verifier/verifier.py:9-32`
- Test: `tests/agents/verifier/test_verifier.py`

**Step 1: Write failing test for [FAIL] detection**

Add to `tests/agents/verifier/test_verifier.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest tests/agents/verifier/test_verifier.py::TestVerifierNotesDetection -v`

Expected: FAIL (tests not found or failing assertions)

**Step 3: Implement Notes checking in Verifier**

Replace `app/agents/verifier/verifier.py`:

```python
"""Verifier - Pure Python logic for tool call verification"""

from app.task.plan import Plan


class Verifier:
    """Verifies tool call status for plan steps"""

    def verify_step(self, plan: Plan, step_idx: int) -> bool:
        """
        Verify a step's tool call status and Notes content.

        Args:
            plan: The execution plan
            step_idx: Index of the step to verify

        Returns:
            True if pass, False if fail (has [FAIL] in Notes or pending tool calls)
        """
        # Check Notes for [FAIL] marker first
        step = plan.steps[step_idx]
        notes = plan.step_notes.get(step, "").strip()
        if notes.startswith("[FAIL]"):
            return False  # LLM self-reported failure

        # Check for pending tool calls
        tool_history = plan.step_tool_history.get(step_idx, [])
        for call in tool_history:
            if call.get("status") == "pending":
                return False  # Has hallucination

        return True

    def get_failed_calls(self, plan: Plan, step_idx: int) -> list[dict]:
        """
        Get list of failed (pending) tool calls for a step.

        Args:
            plan: The execution plan
            step_idx: Index of the step

        Returns:
            List of tool call dicts that are still pending
        """
        tool_history = plan.step_tool_history.get(step_idx, [])
        return [call for call in tool_history if call.get("status") == "pending"]

    def get_failure_reason(self, plan: Plan, step_idx: int) -> str:
        """
        Get the failure reason for a step.

        Args:
            plan: The execution plan
            step_idx: Index of the step

        Returns:
            Failure reason from Notes if [FAIL] tag present, else empty string
        """
        step = plan.steps[step_idx]
        notes = plan.step_notes.get(step, "").strip()
        if notes.startswith("[FAIL]"):
            # Extract reason after [FAIL]: or [FAIL]
            if notes.startswith("[FAIL]:"):
                return notes[7:].strip()
            else:
                return notes[6:].strip()
        return ""
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest tests/agents/verifier/test_verifier.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
cd /home/theo/projects/cortex/.worktrees/dev-scoring && git add app/agents/verifier/verifier.py tests/agents/verifier/test_verifier.py && git commit -m "feat(verifier): add Notes-based failure detection

Verifier now checks Notes for [FAIL] tag before tool call verification.
This detects LLM self-reported failures.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Executor Prompt Update

**Files:**
- Modify: `app/agents/executor/prompts.py`
- No tests needed (prompt-only change)

**Step 1: Update executor prompt**

Replace `app/agents/executor/prompts.py`:

```python
EXECUTOR_SYSTEM_PROMPT = """You are a task execution agent. Your job is to execute assigned steps and report results.

When executing a step:
1. Understand what the step requires
2. Perform the necessary actions
3. Use mark_step to report completion status

## Step Completion Reporting

When you complete a step, you MUST call mark_step with a notes field that starts with a status tag:

- [SUCCESS]: <brief description of what was accomplished>
- [FAIL]: <reason why the step could not be completed>

Examples:
- [SUCCESS]: Posted joke to Facebook successfully
- [FAIL]: Unable to access Facebook API - no credentials available
- [SUCCESS]: Generated Chinese joke about programmers
- [FAIL]: Cannot create image - no image generation tool available

Be HONEST about failures. If you cannot complete a step due to missing tools,
permissions, or external dependencies, report [FAIL] with a clear reason.

Status options for step_status parameter:
- in_progress: Currently working on step
- completed: Step finished successfully (use with [SUCCESS] notes)
- blocked: Step cannot be completed (use with [FAIL] notes)

Always provide notes describing what was done or why it was blocked."""
```

**Step 2: Commit**

```bash
cd /home/theo/projects/cortex/.worktrees/dev-scoring && git add app/agents/executor/prompts.py && git commit -m "feat(executor): add [SUCCESS]/[FAIL] format requirement to prompt

Executor must now report step status with structured notes format.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Deferred Batch Replan in Cortex

**Files:**
- Modify: `cortex.py:173-255`
- Test: `tests/test_cortex.py`

**Step 1: Write failing test for deferred replan**

Add to `tests/test_cortex.py` in `TestCortexVerificationAndReplan`:

```python
    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_batch_replan_collects_multiple_failures(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls,
        mock_verifier_cls, mock_replanner_cls, mock_aggregate
    ):
        """Multiple failures in same batch should trigger single replan with all steps"""
        # Independent parallel steps: 0, 1, 2 (no dependencies)
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Steps 0 and 2 fail verification, step 1 passes
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(side_effect=[False, True, False, True, True])
        mock_verifier_cls.return_value = mock_verifier

        # Track replan calls
        from app.agents.replanner.replanner_agent import ReplanResult
        replan_calls = []

        async def capture_replan(steps_to_replan, available_tools):
            replan_calls.append(sorted(steps_to_replan))
            return ReplanResult(
                action="redesign",
                new_steps=["New Step"],
                new_dependencies={}
            )

        mock_replanner = AsyncMock()
        mock_replanner.replan_subgraph = AsyncMock(side_effect=capture_replan)
        mock_replanner_cls.return_value = mock_replanner
        mock_replanner_cls.MAX_REPLAN_ATTEMPTS = 2

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Should have called replan with both failed steps combined
        assert len(replan_calls) >= 1
        # First replan should include steps 0 and 2 (the failed ones)
        assert 0 in replan_calls[0] or 2 in replan_calls[0]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest tests/test_cortex.py::TestCortexVerificationAndReplan::test_batch_replan_collects_multiple_failures -v`

Expected: FAIL (current code replans immediately per step)

**Step 3: Refactor cortex.py execution loop for batch replan**

In `cortex.py`, replace the results processing loop (lines ~173-255) with:

```python
                for step_idx, result in results:
                    if isinstance(result, Exception):
                        logger.error("✗ Step %d failed with exception: %s", step_idx, result)
                        plan.mark_step(
                            step_idx, step_status="blocked", step_notes=str(result)
                        )
                    else:
                        # Finalize step and verify tool calls
                        plan.finalize_step(step_idx)

                        if verifier.verify_step(plan, step_idx):
                            # Verification passed - mark completed
                            logger.info("✓ Step %d verified - all tool calls confirmed", step_idx)
                            step_outputs[step_idx] = result
                            plan.mark_step(step_idx, step_status="completed")
                        else:
                            # Verification failed - mark blocked (defer replan)
                            failed_calls = verifier.get_failed_calls(plan, step_idx)
                            failure_reason = verifier.get_failure_reason(plan, step_idx)

                            if failure_reason:
                                logger.warning(
                                    "⚠ Step %d verification FAILED - LLM reported: %s",
                                    step_idx, failure_reason
                                )
                            elif failed_calls:
                                logger.warning(
                                    "⚠ Step %d verification FAILED - %d pending tool calls detected",
                                    step_idx, len(failed_calls)
                                )
                                for call in failed_calls:
                                    logger.warning("  - Pending: %s(%s)", call["tool"], call.get("args", {}))

                            plan.mark_step(step_idx, step_status="blocked")

                # Batch replan: collect all blocked steps after processing results
                blocked_steps = [
                    idx for idx in ready_steps
                    if plan.step_statuses[plan.steps[idx]] == "blocked"
                ]

                if blocked_steps:
                    # Collect all steps to replan (blocked + downstream)
                    all_to_replan = set()
                    for idx in blocked_steps:
                        attempts = plan.replan_attempts.get(idx, 0)
                        if attempts < ReplannerAgent.MAX_REPLAN_ATTEMPTS:
                            all_to_replan.add(idx)
                            all_to_replan.update(plan.get_downstream_steps(idx))
                        else:
                            logger.error(
                                "✗ Step %d blocked - max replan attempts (%d) reached",
                                idx, ReplannerAgent.MAX_REPLAN_ATTEMPTS
                            )
                            plan.mark_step(
                                idx,
                                step_status="blocked",
                                step_notes="Max replan attempts reached"
                            )

                    if all_to_replan:
                        steps_to_replan = sorted(all_to_replan)
                        logger.info(
                            "🔄 BATCH REPLANNING: %d steps",
                            len(steps_to_replan)
                        )
                        logger.info("  Steps to replan: %s", steps_to_replan)

                        replan_result = await replanner.replan_subgraph(
                            steps_to_replan=steps_to_replan,
                            available_tools=available_tools
                        )

                        # Update replan attempts for all blocked steps
                        for idx in blocked_steps:
                            plan.replan_attempts[idx] = plan.replan_attempts.get(idx, 0) + 1

                        if replan_result.action == "redesign":
                            logger.info(
                                "✓ Replanner redesigned with %d new steps",
                                len(replan_result.new_steps)
                            )
                            for i, new_step in enumerate(replan_result.new_steps):
                                logger.info("  New step %d: %s", i, new_step)

                            # Find last completed step index
                            completed_indices = [
                                i for i, step in enumerate(plan.steps)
                                if plan.step_statuses[step] == "completed"
                            ]
                            insert_after = max(completed_indices) if completed_indices else -1

                            # Update plan DAG
                            plan.remove_steps(steps_to_replan)
                            plan.add_steps(
                                replan_result.new_steps,
                                replan_result.new_dependencies,
                                insert_after=insert_after
                            )
                            logger.info("  Plan updated: now %d total steps", len(plan.steps))
                        else:
                            # Replanner gave up
                            logger.error("✗ Replanner gave up on steps %s", steps_to_replan)
                            for idx in blocked_steps:
                                plan.mark_step(
                                    idx,
                                    step_status="blocked",
                                    step_notes="Replanner gave up"
                                )
```

**Step 4: Run all verification tests**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest tests/test_cortex.py::TestCortexVerificationAndReplan -v`

Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest -v`

Expected: All PASS

**Step 6: Commit**

```bash
cd /home/theo/projects/cortex/.worktrees/dev-scoring && git add cortex.py tests/test_cortex.py && git commit -m "feat(cortex): implement deferred batch replan

Blocked steps are now collected after batch execution and replanned together.
This improves efficiency by allowing parallel independent steps to complete.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Final Integration Test

**Files:**
- No new files, just run existing tests

**Step 1: Run full test suite**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run pytest -v`

Expected: All PASS

**Step 2: Manual verification (optional)**

Run: `cd /home/theo/projects/cortex/.worktrees/dev-scoring && uv run python example.py`

Check that:
- Steps with `[FAIL]` in Notes are marked blocked
- Replan is triggered after batch completion
- Logging shows correct behavior
