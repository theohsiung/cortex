import pytest
from unittest.mock import Mock
from app.task.task_manager import TaskManager
from app.task.plan import Plan
from app.agents.planner.planner_agent import PlannerAgent


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

    def test_plan_not_found_raises_error(self):
        """Should raise ValueError when plan_id not found"""
        with pytest.raises(ValueError):
            PlannerAgent(
                plan_id="nonexistent",
                model=Mock()
            )

    def test_agent_factory_receives_tools(self):
        """agent_factory should receive toolkit tools"""
        received_tools = []

        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()

        PlannerAgent(
            plan_id="plan_1",
            agent_factory=my_factory
        )

        # Factory should receive create_plan and update_plan tools
        assert len(received_tools) == 2
