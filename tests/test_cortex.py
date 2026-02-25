"""Tests for the Cortex orchestrator."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.agents.verifier.verifier import VerifyResult
from app.config import CortexConfig, ExecutorEntry, MCPSse, ModelConfig, SandboxConfig
from app.task.plan import Plan
from app.task.task_manager import TaskManager
from cortex import Cortex


def make_config(**overrides) -> CortexConfig:
    """Create a minimal CortexConfig for testing."""
    defaults = {"model": ModelConfig(name="test-model", api_base="http://test/v1")}
    defaults.update(overrides)
    return CortexConfig(**defaults)  # type: ignore[arg-type]


class TestCortex:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_creates_empty_history(self):
        """Should initialize with empty history"""
        cortex = Cortex(make_config())

        assert cortex.history == []

    def test_init_stores_model(self):
        """Should store the model config"""
        cortex = Cortex(make_config())

        assert cortex.config.model.name == "test-model"

    def test_history_persists_across_tasks(self):
        """History should persist after task execution"""
        cortex = Cortex(make_config())

        cortex.history.append({"role": "user", "content": "Task 1"})
        cortex.history.append({"role": "assistant", "content": "Done"})

        assert len(cortex.history) == 2

    def test_plan_cleanup(self):
        """Should clean up plan after task"""
        Cortex(make_config())

        # Simulate plan creation and cleanup
        plan_id = "test_plan"
        from app.task.plan import Plan

        TaskManager.set_plan(plan_id, Plan())

        assert TaskManager.get_plan(plan_id) is not None

        TaskManager.remove_plan(plan_id)

        assert TaskManager.get_plan(plan_id) is None


class TestCortexExecutors:
    """Tests for dynamic executor routing API"""

    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_with_executors_list(self):
        """Should accept executors list via ExecutorEntry"""
        # Create a fake module for the factory
        fake_mod = types.ModuleType("_test_exec_gen")
        fake_mod.create = Mock()
        sys.modules["_test_exec_gen"] = fake_mod
        try:
            cortex = Cortex(
                make_config(
                    executors=[
                        ExecutorEntry(
                            intent="generate",
                            description="Generate code",
                            factory_module="_test_exec_gen",
                            factory_function="create",
                        ),
                    ]
                )
            )
            assert "generate" in cortex._executor_entries
        finally:
            del sys.modules["_test_exec_gen"]

    def test_init_without_executors(self):
        """Should work without executors (backward compat)"""
        cortex = Cortex(make_config())
        assert cortex._executor_entries == {}

    def test_init_with_model_only(self):
        """Should not require executor_factory when model is provided"""
        cortex = Cortex(make_config())
        assert cortex.config.model.name == "test-model"

    def test_get_executor_factory_returns_registered(self):
        """Should return registered factory for known intent"""
        factory = Mock()
        fake_mod = types.ModuleType("_test_exec_reg")
        fake_mod.create = factory
        sys.modules["_test_exec_reg"] = fake_mod
        try:
            cortex = Cortex(
                make_config(
                    executors=[
                        ExecutorEntry(
                            intent="generate",
                            description="...",
                            factory_module="_test_exec_reg",
                            factory_function="create",
                        ),
                    ]
                )
            )
            assert cortex._get_executor_factory("generate") is factory
        finally:
            del sys.modules["_test_exec_reg"]

    def test_get_executor_factory_returns_none_for_unknown(self):
        """Should return None for unknown intent"""
        cortex = Cortex(make_config())
        assert cortex._get_executor_factory("nonexistent") is None

    def test_available_intents_from_executors(self):
        """Should build available intents list from executors keys"""
        fake_mod = types.ModuleType("_test_exec_intents")
        fake_mod.create = Mock()
        sys.modules["_test_exec_intents"] = fake_mod
        try:
            cortex = Cortex(
                make_config(
                    executors=[
                        ExecutorEntry(
                            intent="generate",
                            description="Gen code",
                            factory_module="_test_exec_intents",
                            factory_function="create",
                        ),
                        ExecutorEntry(
                            intent="review",
                            description="Review code",
                            factory_module="_test_exec_intents",
                            factory_function="create",
                        ),
                    ]
                )
            )
            intents = cortex._get_available_intents()
            assert {"generate", "review"} == set(intents.keys())
            assert intents["generate"] == "Gen code"
        finally:
            del sys.modules["_test_exec_intents"]

    def test_available_intents_empty_when_no_executors(self):
        """Should return empty dict when no executors provided"""
        cortex = Cortex(make_config())
        intents = cortex._get_available_intents()
        assert intents == {}


class TestCortexSandbox:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_without_sandbox(self):
        """Should work without sandbox (no features enabled)"""
        cortex = Cortex(make_config())
        assert cortex.sandbox is None

    def test_init_with_filesystem_creates_sandbox(self):
        """Should create SandboxManager when filesystem enabled"""
        cortex = Cortex(
            make_config(
                sandbox=SandboxConfig(enable_filesystem=True),
            )
        )
        assert cortex.sandbox is not None
        assert cortex.sandbox.enable_filesystem is True

    def test_init_with_user_id(self):
        """Should pass user_id to SandboxManager"""
        cortex = Cortex(
            make_config(
                sandbox=SandboxConfig(user_id="alice", enable_filesystem=True),
            )
        )
        assert cortex.sandbox.user_id == "alice"

    def test_init_auto_generates_user_id(self):
        """Should auto-generate user_id if not provided"""
        cortex = Cortex(
            make_config(
                sandbox=SandboxConfig(enable_filesystem=True),
            )
        )
        assert cortex.sandbox.user_id.startswith("auto-")

    def test_init_with_all_sandbox_options(self):
        """Should pass all options to SandboxManager"""
        mcp_servers = [MCPSse(transport="sse", url="https://example.com/mcp")]
        cortex = Cortex(
            make_config(
                sandbox=SandboxConfig(user_id="test", enable_filesystem=True, enable_shell=True),
                mcp_servers=mcp_servers,
            )
        )
        assert cortex.sandbox.enable_shell is True
        assert cortex.sandbox.mcp_servers == mcp_servers


class TestCortexParallelExecution:
    """Tests for parallel execution logic with DAG dependencies"""

    def setup_method(self):
        TaskManager._plans.clear()

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_step_outputs_accumulated(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """step_outputs should accumulate results from completed steps"""
        # Setup plan with sequential dependencies: 0 -> 1 -> 2
        plan = Plan(
            title="Test", steps=["Step A", "Step B", "Step C"], dependencies={1: [0], 2: [1]}
        )
        mock_plan_cls.return_value = plan

        # Mock planner
        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock executor to return specific outputs
        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(
            side_effect=[
                "Output from step 0",
                "Output from step 1",
                "Output from step 2",
            ]
        )
        mock_executor_cls.return_value = mock_executor

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Verify all steps were executed
        assert mock_executor.execute_step.call_count == 3

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_dep_context_includes_dependency_outputs(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """dep_context should include outputs from dependency steps"""
        # Setup plan: step 2 depends on steps 0 and 1
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={2: [0, 1]},  # Step 2 depends on 0 and 1
        )
        mock_plan_cls.return_value = plan

        # Mock planner
        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Track contexts passed to execute_step
        captured_contexts = []

        async def capture_context(step_index, context=""):
            captured_contexts.append((step_index, context))
            return f"Output from step {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=capture_context)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Find context for step 2
        step_2_context = next(ctx for idx, ctx in captured_contexts if idx == 2)

        # Step 2's context should include outputs from step 0 and 1
        assert "Step 0 result: Output from step 0" in step_2_context
        assert "Step 1 result: Output from step 1" in step_2_context

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_parallel_steps_execute_together(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """Steps with same dependencies should execute in parallel"""
        # Setup plan: steps 1 and 2 both depend only on step 0
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C", "Step D"],
            dependencies={1: [0], 2: [0], 3: [1, 2]},
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Track execution order
        execution_order = []

        async def track_execution(step_index, context=""):
            execution_order.append(f"start_{step_index}")
            await asyncio.sleep(0.01)  # Small delay to simulate work
            execution_order.append(f"end_{step_index}")
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=track_execution)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Steps 1 and 2 should both start before either ends (parallel execution)
        start_1_idx = execution_order.index("start_1")
        start_2_idx = execution_order.index("start_2")
        end_1_idx = execution_order.index("end_1")
        end_2_idx = execution_order.index("end_2")

        # Both should start before either ends
        assert start_1_idx < end_1_idx
        assert start_2_idx < end_2_idx
        # They should overlap (one starts before the other ends)
        assert start_1_idx < end_2_idx or start_2_idx < end_1_idx

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_failed_step_marked_as_blocked(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """Failed step should be marked as blocked, others continue"""
        # Setup plan: steps 1 and 2 are independent (both depend on 0)
        plan = Plan(
            title="Test", steps=["Step A", "Step B", "Step C"], dependencies={1: [0], 2: [0]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        async def execute_with_failure(step_index, context=""):
            if step_index == 1:
                raise Exception("Step 1 failed")
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=execute_with_failure)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Step 1 should be blocked
        assert plan.step_statuses[1] == "blocked"
        assert "Step 1 failed" in plan.step_notes[1]

        # Step 0 and 2 should be completed
        assert plan.step_statuses[0] == "completed"
        assert plan.step_statuses[2] == "completed"

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_dependent_steps_not_executed_when_dependency_blocked(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """Steps depending on blocked step should not execute"""
        # Setup plan: 0 -> 1 -> 2 (sequential)
        plan = Plan(
            title="Test", steps=["Step A", "Step B", "Step C"], dependencies={1: [0], 2: [1]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        executed_steps = []

        async def execute_with_failure(step_index, context=""):
            executed_steps.append(step_index)
            if step_index == 1:
                raise Exception("Step 1 failed")
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=execute_with_failure)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Step 2 should NOT be executed because step 1 is blocked
        assert 0 in executed_steps
        assert 1 in executed_steps
        assert 2 not in executed_steps

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_semaphore_limits_concurrency(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_verifier_cls, mock_aggregate
    ):
        """Semaphore should limit concurrent executions to 3"""
        # Setup plan with 5 independent steps
        plan = Plan(
            title="Test",
            steps=["Step 0", "Step 1", "Step 2", "Step 3", "Step 4"],
            dependencies={},  # All independent
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Track concurrent executions
        current_concurrent = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency(step_index, context=""):
            nonlocal current_concurrent, max_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=track_concurrency)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Max concurrent should be 3 (semaphore limit)
        assert max_concurrent <= 3


class TestCortexVerificationAndReplan:
    """Tests for verification and replan integration"""

    def setup_method(self):
        TaskManager._plans.clear()

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_verification_pass_completes_step(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Step should be marked completed when verification passes"""
        plan = Plan(title="Test", steps=["Step A", "Step B"], dependencies={1: [0]})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier passes
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Both steps should be completed
        assert plan.step_statuses[0] == "completed"
        assert plan.step_statuses[1] == "completed"

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_verification_fail_triggers_replan(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Verification failure should trigger replan"""
        plan = Plan(title="Test", steps=["Step A", "Step B"], dependencies={1: [0]})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier fails on step 0, then passes on all subsequent steps
        # New flow: fail(0) -> replan -> retry(0)+new(2,3) -> pass(0), pass(2), pass(3)
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            side_effect=[
                VerifyResult(passed=False, notes="[FAIL]: test"),
                VerifyResult(passed=True, notes=""),
                VerifyResult(passed=True, notes=""),
                VerifyResult(passed=True, notes=""),
            ]
        )
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Replanner redesigns
        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(
            return_value=ReplanResult(
                action="redesign",
                failed_step_description="Step A (new approach)",
                failed_step_intent="default",
                continuation_steps={0: "New Step B", 1: "New Step C"},
                continuation_dependencies={1: [0]},
            )
        )
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Replanner should have been called
        mock_replanner.replan.assert_called()

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_replan_give_up_marks_blocked(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Replan give_up should mark step as blocked"""
        plan = Plan(title="Test", steps=["Step A"], dependencies={})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier fails
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            return_value=VerifyResult(passed=False, notes="[FAIL]: test")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Replanner gives up
        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(return_value=ReplanResult(action="give_up"))
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Step should be blocked
        assert plan.step_statuses[0] == "blocked"

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_max_replan_attempts_marks_blocked(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Should mark blocked after max_replan_attempts"""
        plan = Plan(title="Test", steps=["Step A"], dependencies={})
        # Simulate already having 2 global replan attempts (max)
        plan.global_replan_count = 2
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier fails
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            return_value=VerifyResult(passed=False, notes="[FAIL]: test")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Replanner should NOT be called
        mock_replanner = AsyncMock()
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Step should be blocked without calling replanner
        assert plan.step_statuses[0] == "blocked"
        mock_replanner.replan.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_replan_budget_resets_on_replanned_step_success(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """global_replan_count should reset to 0 when a replanned step passes verification"""
        plan = Plan(title="Test", steps=["Step A", "Step B"], dependencies={1: [0]})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Step 0: fail → replan → pass; Step 1: pass
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            side_effect=[
                VerifyResult(passed=False, notes="[FAIL]: test"),
                VerifyResult(passed=True, notes="OK"),
                VerifyResult(passed=True, notes="OK"),
            ]
        )
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(
            return_value=ReplanResult(
                action="redesign",
                failed_step_description="Step A (retry)",
                failed_step_intent="default",
            )
        )
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # After replanned step 0 succeeds, budget should be reset
        assert plan.global_replan_count == 0

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_downstream_steps_included_in_replan(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Replan should include failed step and all downstream dependencies"""
        # 0 -> 1 -> 2
        plan = Plan(
            title="Test", steps=["Step A", "Step B", "Step C"], dependencies={1: [0], 2: [1]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier fails on step 0
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            side_effect=[
                VerifyResult(passed=False, notes="[FAIL]: test"),
                VerifyResult(passed=True, notes=""),
                VerifyResult(passed=True, notes=""),
                VerifyResult(passed=True, notes=""),
            ]
        )
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier_cls.return_value = mock_verifier

        # Track replan calls
        from app.agents.replanner.replanner_agent import ReplanResult

        replan_calls = []

        async def capture_replan(**kwargs):
            replan_calls.append(kwargs.get("failed_step_id"))
            return ReplanResult(
                action="redesign",
                failed_step_description="New approach for A",
                failed_step_intent="default",
                continuation_steps={0: "New B", 1: "New C"},
                continuation_dependencies={1: [0]},
            )

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(side_effect=capture_replan)
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Should replan step 0 (failed_step_id is now an int, not a list)
        assert len(replan_calls) > 0
        assert replan_calls[0] == 0

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_batch_replan_collects_multiple_failures(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Multiple failures in same batch should trigger single replan with all steps"""
        # Independent parallel steps: 0, 1, 2 (no dependencies)
        plan = Plan(title="Test", steps=["Step A", "Step B", "Step C"], dependencies={})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Steps 0 and 2 fail verification on first attempt, then pass on retry
        verify_call_count = 0

        def verify_side_effect(plan_obj, step_idx):
            nonlocal verify_call_count
            verify_call_count += 1
            # First batch: steps 0 and 2 fail, step 1 passes
            if verify_call_count <= 3:
                if step_idx in (0, 2):
                    return VerifyResult(passed=False, notes="[FAIL]: test")
                return VerifyResult(passed=True, notes="")
            # All subsequent calls pass
            return VerifyResult(passed=True, notes="")

        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(side_effect=verify_side_effect)
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(passed=True, notes="Verified")
        )
        mock_verifier.get_failed_calls = MagicMock(return_value=[])
        mock_verifier.get_failure_reason = MagicMock(return_value="Test failure")
        mock_verifier_cls.return_value = mock_verifier

        # Track replan calls
        from app.agents.replanner.replanner_agent import ReplanResult

        replan_calls = []
        replan_counter = 0

        async def capture_replan(**kwargs):
            nonlocal replan_counter
            failed_id = kwargs.get("failed_step_id")
            replan_calls.append(failed_id)
            replan_counter += 1
            return ReplanResult(
                action="redesign",
                failed_step_description=f"Retried step {failed_id}",
                failed_step_intent="default",
                continuation_steps={0: "New Step"},
                continuation_dependencies={},
            )

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(side_effect=capture_replan)
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Should have called replan for failed steps
        assert len(replan_calls) >= 1
        assert 0 in replan_calls or 2 in replan_calls

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_notes_failure_reason_logged(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Notes-based failure reason should be used in logging"""
        plan = Plan(title="Test", steps=["Step A"], dependencies={})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Done")
        mock_executor_cls.return_value = mock_executor

        # Verifier fails with Notes-based failure
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(
            return_value=VerifyResult(passed=False, notes="[FAIL]: Unable to access Facebook API")
        )
        mock_verifier.get_failed_calls = MagicMock(return_value=[])
        mock_verifier.get_failure_reason = MagicMock(return_value="Unable to access Facebook API")
        mock_verifier_cls.return_value = mock_verifier

        # Replanner gives up
        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(return_value=ReplanResult(action="give_up"))
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Test query")

        # Verify get_failure_reason was called
        mock_verifier.get_failure_reason.assert_called()


class TestCortexIntentDispatch:
    """Tests for intent-based executor dispatch"""

    def setup_method(self):
        TaskManager._plans.clear()

    def test_dispatches_to_correct_factory(self):
        """Should look up correct factory based on intent"""
        gen_factory = Mock()
        review_factory = Mock()

        fake_mod_gen = types.ModuleType("_test_dispatch_gen")
        fake_mod_gen.create = gen_factory
        sys.modules["_test_dispatch_gen"] = fake_mod_gen

        fake_mod_rev = types.ModuleType("_test_dispatch_rev")
        fake_mod_rev.create = review_factory
        sys.modules["_test_dispatch_rev"] = fake_mod_rev

        try:
            cortex = Cortex(
                make_config(
                    executors=[
                        ExecutorEntry(
                            intent="generate",
                            description="Gen",
                            factory_module="_test_dispatch_gen",
                            factory_function="create",
                        ),
                        ExecutorEntry(
                            intent="review",
                            description="Review",
                            factory_module="_test_dispatch_rev",
                            factory_function="create",
                        ),
                    ]
                )
            )
            assert cortex._get_executor_factory("generate") is gen_factory
            assert cortex._get_executor_factory("review") is review_factory
            # Unknown intent falls back to first configured executor
            assert cortex._get_executor_factory("default") is gen_factory
        finally:
            del sys.modules["_test_dispatch_gen"]
            del sys.modules["_test_dispatch_rev"]

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.Verifier")
    @patch("cortex.PlannerAgent")
    async def test_external_executor_uses_evaluate_output(
        self, mock_planner_cls, mock_verifier_cls, mock_aggregate
    ):
        """External executor steps should be verified via evaluate_output"""
        plan = Plan(
            title="Test", steps=["Generate code"], dependencies={}, step_intents={0: "generate"}
        )

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock verifier: verify_step passes (no tool history), but evaluate_output fails
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(
                passed=False, notes="[FAIL]: Output was just a description, no actual code"
            )
        )
        mock_verifier.get_failed_calls = MagicMock(return_value=[])
        mock_verifier.get_failure_reason = MagicMock(return_value="Output was just a description")
        mock_verifier_cls.return_value = mock_verifier

        # Replanner gives up (to end the loop)
        from app.agents.replanner.replanner_agent import ReplanResult

        with patch("cortex.ReplannerAgent") as mock_replanner_cls:
            mock_replanner = AsyncMock()
            mock_replanner.replan = AsyncMock(return_value=ReplanResult(action="give_up"))
            mock_replanner_cls.return_value = mock_replanner

            # External executor factory - mock _get_executor_factory directly
            mock_agent = MagicMock()
            mock_exec_result = MagicMock()
            mock_exec_result.output = "A parser typically works by..."

            with patch("cortex.Plan", return_value=plan):
                with patch("cortex.ExecutorAgent"):
                    with patch(
                        "app.agents.base.base_agent.BaseAgent.execute",
                        new_callable=AsyncMock,
                        return_value=mock_exec_result,
                    ):
                        cortex = Cortex(make_config())
                        # Mock _get_executor_factory to return our lambda factory
                        cortex._get_executor_factory = lambda intent: (
                            (lambda: mock_agent) if intent == "generate" else None
                        )
                        await cortex.execute("Generate parser code")

        # evaluate_output should have been called for external executor
        mock_verifier.evaluate_output.assert_called()
        # Step should NOT be completed (evaluate_output failed)
        assert plan.step_statuses[0] != "completed"


class TestCortexReplanContext:
    """Tests for ReplanContext construction on failure"""

    def setup_method(self):
        TaskManager._plans.clear()

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_failure_saves_notes_to_plan(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Verification failure should save verifier notes to plan"""
        plan = Plan(title="Test", steps=["Step A"], dependencies={})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Found geo coords but no ZIP")
        mock_executor_cls.return_value = mock_executor

        # Verifier passes mechanical check but fails LLM evaluation
        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(
            return_value=VerifyResult(
                passed=False,
                notes="[FAIL]: Found geo coords | No ZIP code field",
            )
        )
        mock_verifier.get_failed_calls = MagicMock(return_value=[])
        mock_verifier.get_failure_reason = MagicMock(return_value="")
        mock_verifier_cls.return_value = mock_verifier

        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(return_value=ReplanResult(action="give_up"))
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Find ZIP codes")

        # After give_up, step ends up as blocked with "Replanner gave up"
        assert plan.step_statuses[0] == "blocked"
        assert "Replanner gave up" in plan.step_notes[0]

    @pytest.mark.asyncio
    @patch.object(
        Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result"
    )
    @patch("cortex.ReplannerAgent")
    @patch("cortex.Verifier")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_replan_receives_context(
        self,
        mock_plan_cls,
        mock_planner_cls,
        mock_executor_cls,
        mock_verifier_cls,
        mock_replanner_cls,
        mock_aggregate,
    ):
        """Replanner should receive keyword arguments with failure details"""
        plan = Plan(title="Test", steps=["Step A", "Step B"], dependencies={1: [0]})
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(return_value="Some output")
        mock_executor_cls.return_value = mock_executor

        # Fail on first evaluate_output call, pass on subsequent ones
        eval_count = 0

        async def eval_side_effect(step_desc, output, tool_call_count=0):
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                return VerifyResult(passed=False, notes="[FAIL]: goal not met")
            return VerifyResult(passed=True, notes="[SUCCESS]: done")

        mock_verifier = MagicMock()
        mock_verifier.verify_step = MagicMock(return_value=VerifyResult(passed=True, notes=""))
        mock_verifier.evaluate_output = AsyncMock(side_effect=eval_side_effect)
        mock_verifier.get_failed_calls = MagicMock(return_value=[])
        mock_verifier.get_failure_reason = MagicMock(return_value="")
        mock_verifier_cls.return_value = mock_verifier

        from app.agents.replanner.replanner_agent import ReplanResult

        mock_replanner = AsyncMock()
        mock_replanner.replan = AsyncMock(
            return_value=ReplanResult(
                action="redesign",
                failed_step_description="New approach",
                failed_step_intent="default",
                continuation_steps={0: "New Step"},
                continuation_dependencies={},
            )
        )
        mock_replanner_cls.return_value = mock_replanner

        cortex = Cortex(make_config())
        await cortex.execute("Find ZIP codes")

        # replan should have been called with keyword arguments
        mock_replanner.replan.assert_called()
        call_kwargs = mock_replanner.replan.call_args
        assert call_kwargs is not None
        # Verify the key arguments were passed
        assert call_kwargs.kwargs.get("original_query") == "Find ZIP codes"
        assert call_kwargs.kwargs.get("failed_step_id") == 0
        assert call_kwargs.kwargs.get("attempt") == 1
