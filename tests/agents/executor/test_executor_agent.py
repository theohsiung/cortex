import pytest
from unittest.mock import Mock
from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.executor.executor_agent import ExecutorAgent


class TestExecutorAgent:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan(title="Test", steps=["Step A", "Step B"])
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_gets_plan_from_task_manager(self):
        """Should get plan from TaskManager"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert agent.plan is self.plan

    def test_has_mark_step_tool(self):
        """Should have mark_step tool"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert "mark_step" in agent.tool_functions

    def test_mark_step_modifies_plan(self):
        """mark_step should modify the TaskManager plan"""
        agent = ExecutorAgent(
            plan_id="plan_1",
            model=Mock()
        )

        agent.tool_functions["mark_step"](
            step_index=0,
            status="completed",
            notes="Done"
        )

        plan = TaskManager.get_plan("plan_1")
        assert plan.step_statuses["Step A"] == "completed"
        assert plan.step_notes["Step A"] == "Done"

    def test_plan_not_found_raises_error(self):
        """Should raise ValueError when plan_id not found"""
        with pytest.raises(ValueError):
            ExecutorAgent(
                plan_id="nonexistent",
                model=Mock()
            )
