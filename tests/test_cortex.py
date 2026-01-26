import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from cortex import Cortex
from app.task.task_manager import TaskManager
from app.task.plan import Plan
from pathlib import Path


class TestCortex:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_creates_empty_history(self):
        """Should initialize with empty history"""
        cortex = Cortex(model=Mock())

        assert cortex.history == []

    def test_init_stores_model(self):
        """Should store the model"""
        model = Mock()
        cortex = Cortex(model=model)

        assert cortex.model is model

    def test_history_persists_across_tasks(self):
        """History should persist after task execution"""
        cortex = Cortex(model=Mock())

        cortex.history.append({"role": "user", "content": "Task 1"})
        cortex.history.append({"role": "assistant", "content": "Done"})

        assert len(cortex.history) == 2

    def test_plan_cleanup(self):
        """Should clean up plan after task"""
        cortex = Cortex(model=Mock())

        # Simulate plan creation and cleanup
        plan_id = "test_plan"
        from app.task.plan import Plan
        TaskManager.set_plan(plan_id, Plan())

        assert TaskManager.get_plan(plan_id) is not None

        TaskManager.remove_plan(plan_id)

        assert TaskManager.get_plan(plan_id) is None

    def test_init_with_custom_factories(self):
        """Should accept custom planner and executor factories"""
        planner_factory = Mock()
        executor_factory = Mock()
        cortex = Cortex(planner_factory=planner_factory, executor_factory=executor_factory)

        assert cortex.planner_factory is planner_factory
        assert cortex.executor_factory is executor_factory
        assert cortex.model is None

    def test_init_with_model_and_custom_planner_factory(self):
        """Should accept model with custom planner factory"""
        model = Mock()
        planner_factory = Mock()
        cortex = Cortex(model=model, planner_factory=planner_factory)

        assert cortex.model is model
        assert cortex.planner_factory is planner_factory
        assert cortex.executor_factory is None

    def test_init_requires_model_or_planner_factory(self):
        """Should raise error if no model and no planner_factory"""
        with pytest.raises(ValueError, match="planner_factory"):
            Cortex(executor_factory=Mock())

    def test_init_requires_model_or_executor_factory(self):
        """Should raise error if no model and no executor_factory"""
        with pytest.raises(ValueError, match="executor_factory"):
            Cortex(planner_factory=Mock())


class TestCortexSandbox:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_without_sandbox(self):
        """Should work without sandbox (no features enabled)"""
        cortex = Cortex(model=Mock())
        assert cortex.sandbox is None

    def test_init_with_filesystem_creates_sandbox(self):
        """Should create SandboxManager when filesystem enabled"""
        cortex = Cortex(
            model=Mock(),
            enable_filesystem=True,
        )
        assert cortex.sandbox is not None
        assert cortex.sandbox.enable_filesystem is True

    def test_init_with_user_id(self):
        """Should pass user_id to SandboxManager"""
        cortex = Cortex(
            model=Mock(),
            user_id="alice",
            enable_filesystem=True,
        )
        assert cortex.sandbox.user_id == "alice"

    def test_init_auto_generates_user_id(self):
        """Should auto-generate user_id if not provided"""
        cortex = Cortex(
            model=Mock(),
            enable_filesystem=True,
        )
        assert cortex.sandbox.user_id.startswith("auto-")

    def test_init_with_all_sandbox_options(self):
        """Should pass all options to SandboxManager"""
        mcp_servers = [{"url": "https://example.com/mcp"}]
        cortex = Cortex(
            model=Mock(),
            user_id="test",
            enable_filesystem=True,
            enable_shell=True,
            mcp_servers=mcp_servers,
        )
        assert cortex.sandbox.enable_shell is True
        assert cortex.sandbox.mcp_servers == mcp_servers


class TestCortexParallelExecution:
    """Tests for parallel execution logic with DAG dependencies"""

    def setup_method(self):
        TaskManager._plans.clear()

    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_step_outputs_accumulated(self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate):
        """step_outputs should accumulate results from completed steps"""
        # Setup plan with sequential dependencies: 0 -> 1 -> 2
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={1: [0], 2: [1]}
        )
        mock_plan_cls.return_value = plan

        # Mock planner
        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Mock executor to return specific outputs
        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=[
            "Output from step 0",
            "Output from step 1",
            "Output from step 2",
        ])
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Verify all steps were executed
        assert mock_executor.execute_step.call_count == 3

    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_dep_context_includes_dependency_outputs(self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate):
        """dep_context should include outputs from dependency steps"""
        # Setup plan: step 2 depends on steps 0 and 1
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={2: [0, 1]}  # Step 2 depends on 0 and 1
        )
        mock_plan_cls.return_value = plan

        # Mock planner
        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        # Track contexts passed to execute_step
        captured_contexts = []

        async def capture_context(step_index, context=""):
            captured_contexts.append((step_index, context))
            return f"Output from step {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=capture_context)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Find context for step 2
        step_2_context = next(ctx for idx, ctx in captured_contexts if idx == 2)

        # Step 2's context should include outputs from step 0 and 1
        assert "Step 0 result: Output from step 0" in step_2_context
        assert "Step 1 result: Output from step 1" in step_2_context

    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_parallel_steps_execute_together(self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate):
        """Steps with same dependencies should execute in parallel"""
        # Setup plan: steps 1 and 2 both depend only on step 0
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C", "Step D"],
            dependencies={1: [0], 2: [0], 3: [1, 2]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

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

        cortex = Cortex(model=Mock())
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
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_failed_step_marked_as_blocked(self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate):
        """Failed step should be marked as blocked, others continue"""
        # Setup plan: steps 1 and 2 are independent (both depend on 0)
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={1: [0], 2: [0]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        async def execute_with_failure(step_index, context=""):
            if step_index == 1:
                raise Exception("Step 1 failed")
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=execute_with_failure)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Step 1 should be blocked
        assert plan.step_statuses["Step B"] == "blocked"
        assert "Step 1 failed" in plan.step_notes["Step B"]

        # Step 0 and 2 should be completed
        assert plan.step_statuses["Step A"] == "completed"
        assert plan.step_statuses["Step C"] == "completed"

    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_dependent_steps_not_executed_when_dependency_blocked(
        self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate
    ):
        """Steps depending on blocked step should not execute"""
        # Setup plan: 0 -> 1 -> 2 (sequential)
        plan = Plan(
            title="Test",
            steps=["Step A", "Step B", "Step C"],
            dependencies={1: [0], 2: [1]}
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

        executed_steps = []

        async def execute_with_failure(step_index, context=""):
            executed_steps.append(step_index)
            if step_index == 1:
                raise Exception("Step 1 failed")
            return f"Output {step_index}"

        mock_executor = AsyncMock()
        mock_executor.execute_step = AsyncMock(side_effect=execute_with_failure)
        mock_executor_cls.return_value = mock_executor

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Step 2 should NOT be executed because step 1 is blocked
        assert 0 in executed_steps
        assert 1 in executed_steps
        assert 2 not in executed_steps

    @pytest.mark.asyncio
    @patch.object(Cortex, "_aggregate_results", new_callable=AsyncMock, return_value="Aggregated result")
    @patch("cortex.ExecutorAgent")
    @patch("cortex.PlannerAgent")
    @patch("cortex.Plan")
    async def test_semaphore_limits_concurrency(self, mock_plan_cls, mock_planner_cls, mock_executor_cls, mock_aggregate):
        """Semaphore should limit concurrent executions to 3"""
        # Setup plan with 5 independent steps
        plan = Plan(
            title="Test",
            steps=["Step 0", "Step 1", "Step 2", "Step 3", "Step 4"],
            dependencies={}  # All independent
        )
        mock_plan_cls.return_value = plan

        mock_planner = AsyncMock()
        mock_planner.create_plan = AsyncMock(return_value=None)
        mock_planner_cls.return_value = mock_planner

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

        cortex = Cortex(model=Mock())
        await cortex.execute("Test query")

        # Max concurrent should be 3 (semaphore limit)
        assert max_concurrent <= 3
