import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from cortex import Cortex
from app.task.task_manager import TaskManager
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
        """Should work without sandbox (no workspace)"""
        cortex = Cortex(model=Mock())
        assert cortex.sandbox is None

    def test_init_with_workspace_creates_sandbox(self):
        """Should create SandboxManager when workspace provided"""
        cortex = Cortex(
            model=Mock(),
            workspace="/tmp/test",
            enable_filesystem=True,
        )
        assert cortex.sandbox is not None
        assert "/tmp/test" in cortex.sandbox.workspace  # resolved path
        assert cortex.sandbox.enable_filesystem is True

    def test_init_with_all_sandbox_options(self):
        """Should pass all options to SandboxManager"""
        mcp_servers = [{"url": "https://example.com/mcp"}]
        cortex = Cortex(
            model=Mock(),
            workspace="/tmp/test",
            enable_filesystem=True,
            enable_shell=True,
            mcp_servers=mcp_servers,
        )
        assert cortex.sandbox.enable_shell is True
        assert cortex.sandbox.mcp_servers == mcp_servers
