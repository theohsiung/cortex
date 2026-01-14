import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.agents.base.base_agent import BaseAgent, AgentResult
from app.task.task_manager import TaskManager
from app.task.plan import Plan


class TestBaseAgent:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_without_plan(self):
        """Should initialize without plan when plan_id not provided"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test instruction"
        )

        assert agent.plan is None
        assert agent.plan_id is None

    def test_init_with_plan(self):
        """Should get plan from TaskManager when plan_id provided"""
        plan = Plan(title="Test")
        TaskManager.set_plan("plan_1", plan)

        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test",
            plan_id="plan_1"
        )

        assert agent.plan is plan
        assert agent.plan_id == "plan_1"

    def test_tool_events_initialized_empty(self):
        """Should initialize with empty tool events"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        assert agent._tool_events == []

    def test_track_tool_event(self):
        """Should track tool call events"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        agent._track_tool_event({
            "type": "call",
            "name": "test_tool",
            "args": {"arg1": "value1"}
        })

        assert len(agent._tool_events) == 1
        assert agent._tool_events[0]["name"] == "test_tool"

    def test_get_tool_summary(self):
        """Should return tool usage statistics"""
        agent = BaseAgent(
            name="test",
            model=Mock(),
            tool_declarations=[],
            tool_functions={},
            instruction="Test"
        )

        agent._tool_events = [
            {"type": "call", "name": "tool_a"},
            {"type": "response", "name": "tool_a"},
            {"type": "call", "name": "tool_b"},
            {"type": "response", "name": "tool_b"},
            {"type": "call", "name": "tool_a"},
            {"type": "response", "name": "tool_a"},
        ]

        summary = agent.get_tool_summary()

        assert summary["total_calls"] == 3
        assert summary["total_responses"] == 3
        assert set(summary["tools_used"]) == {"tool_a", "tool_b"}


class TestAgentResult:
    def test_agent_result_creation(self):
        """Should create AgentResult with all fields"""
        result = AgentResult(
            events=[{"event": 1}],
            output="Test output",
            is_complete=True
        )

        assert result.events == [{"event": 1}]
        assert result.output == "Test output"
        assert result.is_complete is True
