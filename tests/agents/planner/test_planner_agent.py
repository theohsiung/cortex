from unittest.mock import Mock

import pytest

from app.agents.planner.planner_agent import PlannerAgent
from app.task.plan import Plan
from app.task.task_manager import TaskManager


class TestPlannerAgent:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan()
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_gets_plan_from_task_manager(self):
        """Should get plan from TaskManager"""
        agent = PlannerAgent(plan_id="plan_1", model=Mock())

        assert agent.plan is self.plan

    def test_has_plan_tools(self):
        """Should have create_plan and update_plan tools"""
        agent = PlannerAgent(plan_id="plan_1", model=Mock())

        assert "create_plan" in agent.tool_functions
        assert "update_plan" in agent.tool_functions

    def test_tools_modify_same_plan(self):
        """Tools should modify the TaskManager plan"""
        agent = PlannerAgent(plan_id="plan_1", model=Mock())

        agent.tool_functions["create_plan"](title="Test Plan", steps=["Step 1", "Step 2"])

        # Verify the TaskManager plan was modified
        plan = TaskManager.get_plan("plan_1")
        assert plan.title == "Test Plan"
        assert plan.steps == ["Step 1", "Step 2"]

    def test_plan_not_found_raises_error(self):
        """Should raise ValueError when plan_id not found"""
        with pytest.raises(ValueError):
            PlannerAgent(plan_id="nonexistent", model=Mock())

    def test_agent_factory_receives_tools(self):
        """agent_factory should receive toolkit tools"""
        received_tools = []

        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()

        PlannerAgent(plan_id="plan_1", agent_factory=my_factory)

        # Factory should receive 2 tools (no aliases by default)
        assert len(received_tools) == 2

        # Should include create_plan and update_plan
        func_names = [f.__name__ for f in received_tools]
        assert "create_plan" in func_names
        assert "update_plan" in func_names

    def test_extra_tools_included(self):
        """extra_tools should be included in agent tools"""
        extra_tool = Mock(name="extra_tool")
        received_tools = []

        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()

        PlannerAgent(
            plan_id="plan_1",
            agent_factory=my_factory,
            extra_tools=[extra_tool],
        )

        assert extra_tool in received_tools
        # Should have 2 toolkit tools + 1 extra tool
        assert len(received_tools) == 3


class TestPlannerAgentIntents:
    def setup_method(self):
        TaskManager._plans.clear()
        self.plan = Plan()
        TaskManager.set_plan("plan_1", self.plan)

    def test_init_with_available_intents(self):
        """Should accept available_intents parameter"""
        intents = {
            "generate": "Generate new code",
            "review": "Review code quality",
            "default": "General tasks",
        }
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock(),
            available_intents=intents,
        )
        assert agent is not None

    def test_init_without_available_intents(self):
        """Should work without available_intents (backward compat)"""
        agent = PlannerAgent(
            plan_id="plan_1",
            model=Mock(),
        )
        assert agent is not None

    def test_intent_prompt_injection(self):
        """Planner prompt should include available intents"""
        from app.agents.planner.prompts import build_intent_prompt_section

        intents = {
            "generate": "Generate new code",
            "default": "General tasks",
        }
        prompt = build_intent_prompt_section(intents)
        assert "generate" in prompt
        assert "Generate new code" in prompt
        assert "default" in prompt

    def test_intent_prompt_not_injected_when_only_default(self):
        """Should not inject intent section when only 'default' intent exists"""
        from app.agents.planner.prompts import build_intent_prompt_section

        intents = {"default": "General tasks"}
        prompt = build_intent_prompt_section(intents)
        assert prompt == ""  # No need to inject when only default
