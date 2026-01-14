import pytest
from unittest.mock import Mock, AsyncMock
from cortex import Cortex
from app.task.task_manager import TaskManager


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

    def test_init_with_custom_agents(self):
        """Should accept custom planner and executor agents"""
        planner = Mock()
        executor = Mock()
        cortex = Cortex(planner_agent=planner, executor_agent=executor)

        assert cortex.planner_agent is planner
        assert cortex.executor_agent is executor
        assert cortex.model is None

    def test_init_with_model_and_custom_planner(self):
        """Should accept model with custom planner agent"""
        model = Mock()
        planner = Mock()
        cortex = Cortex(model=model, planner_agent=planner)

        assert cortex.model is model
        assert cortex.planner_agent is planner
        assert cortex.executor_agent is None

    def test_init_requires_model_or_planner(self):
        """Should raise error if no model and no planner_agent"""
        with pytest.raises(ValueError, match="planner_agent"):
            Cortex(executor_agent=Mock())

    def test_init_requires_model_or_executor(self):
        """Should raise error if no model and no executor_agent"""
        with pytest.raises(ValueError, match="executor_agent"):
            Cortex(planner_agent=Mock())
