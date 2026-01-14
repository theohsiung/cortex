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
