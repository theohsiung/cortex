import json

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.agents.base.base_agent import BaseAgent, AgentResult
from app.task.task_manager import TaskManager
from app.task.plan import Plan


def create_mock_agent(name="test"):
    """Helper to create a mock ADK agent"""
    mock = Mock()
    mock.name = name
    return mock


class TestBaseAgent:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_init_without_plan(self):
        """Should initialize without plan when plan_id not provided"""
        mock_agent = create_mock_agent()

        agent = BaseAgent(
            agent=mock_agent,
            tool_functions={}
        )

        assert agent.plan is None
        assert agent.plan_id is None

    def test_init_with_plan(self):
        """Should get plan from TaskManager when plan_id provided"""
        plan = Plan(title="Test")
        TaskManager.set_plan("plan_1", plan)
        mock_agent = create_mock_agent()

        agent = BaseAgent(
            agent=mock_agent,
            tool_functions={},
            plan_id="plan_1"
        )

        assert agent.plan is plan
        assert agent.plan_id == "plan_1"

    def test_agent_stored_directly(self):
        """Should store the passed agent directly"""
        mock_agent = create_mock_agent("my_agent")

        agent = BaseAgent(agent=mock_agent)

        assert agent.agent is mock_agent
        assert agent.agent.name == "my_agent"

    def test_tool_events_initialized_empty(self):
        """Should initialize with empty tool events"""
        mock_agent = create_mock_agent()

        agent = BaseAgent(agent=mock_agent)

        assert agent._tool_events == []

    def test_track_tool_event(self):
        """Should track tool call events"""
        mock_agent = create_mock_agent()
        agent = BaseAgent(agent=mock_agent)

        agent._track_tool_event({
            "type": "call",
            "name": "test_tool",
            "args": {"arg1": "value1"}
        })

        assert len(agent._tool_events) == 1
        assert agent._tool_events[0]["name"] == "test_tool"

    def test_get_tool_summary(self):
        """Should return tool usage statistics"""
        mock_agent = create_mock_agent()
        agent = BaseAgent(agent=mock_agent)

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

    def test_tool_functions_default_empty(self):
        """Should default tool_functions to empty dict"""
        mock_agent = create_mock_agent()

        agent = BaseAgent(agent=mock_agent)

        assert agent.tool_functions == {}

    def test_should_include_aliases_none_model(self):
        """Should return False when model is None"""
        assert BaseAgent.should_include_aliases(None) is False

    def test_should_include_aliases_gemini(self):
        """Should return False for Gemini models"""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="gemini/gemini-2.5-flash")
        assert BaseAgent.should_include_aliases(mock_model) is False

    def test_should_include_aliases_openai(self):
        """Should return True for OpenAI models"""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="openai/gpt-oss-20b")
        assert BaseAgent.should_include_aliases(mock_model) is True

    def test_should_include_aliases_unknown(self):
        """Should return False for unknown models"""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="some-other-model")
        assert BaseAgent.should_include_aliases(mock_model) is False


class TestExecuteRetryOnJsonDecodeError:
    """Tests for retry on JSONDecodeError during execute."""

    def setup_method(self):
        TaskManager._plans.clear()

    @pytest.mark.asyncio
    async def test_retries_on_json_decode_error(self):
        """Should retry when _run_once raises JSONDecodeError and succeed."""
        mock_agent = create_mock_agent()
        agent = BaseAgent(agent=mock_agent)

        good_result = AgentResult(events=[], output="ok", is_complete=True)
        agent._run_once = AsyncMock(
            side_effect=[
                json.JSONDecodeError("bad json", "", 0),
                good_result,
            ]
        )
        agent._get_session_service = Mock(
            return_value=AsyncMock(
                create_session=AsyncMock(return_value=Mock(id="s1"))
            )
        )

        result = await agent.execute("test query")

        assert result.output == "ok"
        assert agent._run_once.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        """Should raise JSONDecodeError after max_iteration retries."""
        mock_agent = create_mock_agent()
        agent = BaseAgent(agent=mock_agent)

        agent._run_once = AsyncMock(
            side_effect=json.JSONDecodeError("bad json", "", 0)
        )
        agent._get_session_service = Mock(
            return_value=AsyncMock(
                create_session=AsyncMock(return_value=Mock(id="s1"))
            )
        )

        with pytest.raises(json.JSONDecodeError):
            await agent.execute("test query", max_iteration=3)

        assert agent._run_once.call_count == 3


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
