import pytest
from unittest.mock import Mock
from app.agents.planner.planner_agent import PlannerAgent
from app.task.task_manager import TaskManager
from app.task.plan import Plan


class TestPlannerAgent:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan()
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_gets_plan_from_task_manager(self):
        """Should get plan from TaskManager"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert agent.plan is self.plan

    def test_has_plan_tools(self):
        """Should have create_plan and update_plan tools"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        assert "create_plan" in agent.tool_functions
        assert "update_plan" in agent.tool_functions

    def test_tools_modify_same_plan(self):
        """Tools should modify the TaskManager plan"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock()
        )

        agent.tool_functions["create_plan"](
            title="Test Plan",
            steps=["Step 1", "Step 2"]
        )

        # Verify the TaskManager plan was modified
        plan = TaskManager.get_plan("plan_1")
        assert plan.title == "Test Plan"
        assert plan.steps == ["Step 1", "Step 2"]
