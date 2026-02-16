import pytest
from unittest.mock import Mock, AsyncMock
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
        agent = ExecutorAgent(plan_id="plan_1", model=Mock())
        assert agent.plan is self.plan

    def test_no_mark_step_tool(self):
        """Executor should NOT have mark_step tool (decoupled)"""
        agent = ExecutorAgent(plan_id="plan_1", model=Mock())
        assert "mark_step" not in agent.tool_functions

    def test_plan_not_found_raises_error(self):
        """Should raise ValueError when plan_id not found"""
        with pytest.raises(ValueError):
            ExecutorAgent(plan_id="nonexistent", model=Mock())

    def test_external_agent_factory_no_tools_injected(self):
        """External agent factory (no args) should work"""
        called = []
        def my_factory():
            called.append(True)
            return Mock()
        ExecutorAgent(plan_id="plan_1", agent_factory=my_factory)
        assert len(called) == 1

    def test_legacy_agent_factory_with_tools(self):
        """Legacy agent factory (with tools arg) should still work"""
        received_tools = []
        def my_factory(tools: list):
            received_tools.extend(tools)
            return Mock()
        ExecutorAgent(plan_id="plan_1", agent_factory=my_factory, extra_tools=[Mock()])
        # Should receive extra_tools (no mark_step)
        assert len(received_tools) >= 1

    def test_extra_tools_supported_for_default(self):
        """Default executor should accept extra_tools"""
        extra = Mock(name="sandbox_tool")
        agent = ExecutorAgent(plan_id="plan_1", model=Mock(), extra_tools=[extra])
        assert agent is not None
