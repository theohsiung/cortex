"""Tests for ReplannerAgent - subgraph redesign after verification failure"""

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from app.task.plan import Plan
from app.task.task_manager import TaskManager
from app.agents.replanner.replanner_agent import ReplannerAgent, ReplanResult


class TestReplanResult:
    """Tests for ReplanResult dataclass"""

    def test_replan_result_redesign(self):
        """Should create ReplanResult with redesign action"""
        result = ReplanResult(
            action="redesign",
            new_steps=["Step A", "Step B"],
            new_dependencies={1: [0]}
        )

        assert result.action == "redesign"
        assert result.new_steps == ["Step A", "Step B"]
        assert result.new_dependencies == {1: [0]}

    def test_replan_result_give_up(self):
        """Should create ReplanResult with give_up action"""
        result = ReplanResult(
            action="give_up",
            new_steps=[],
            new_dependencies={}
        )

        assert result.action == "give_up"
        assert result.new_steps == []


class TestReplannerAgentConstants:
    """Tests for ReplannerAgent constants"""

    def test_max_replan_attempts(self):
        """Should have MAX_REPLAN_ATTEMPTS = 2"""
        assert ReplannerAgent.MAX_REPLAN_ATTEMPTS == 2


class TestReplannerAgentInit:
    """Tests for ReplannerAgent initialization"""

    def setup_method(self):
        """Set up test fixtures"""
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        """Clean up after tests"""
        TaskManager.remove_plan("test_plan")

    def test_init_with_model(self):
        """Should initialize with model"""
        plan = Plan(steps=["A", "B"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        assert agent.plan == plan

    def test_init_raises_without_plan(self):
        """Should raise error if plan not found"""
        with pytest.raises(ValueError, match="Plan not found"):
            ReplannerAgent(plan_id="nonexistent", model=MagicMock())

    def test_init_raises_without_model_or_factory(self):
        """Should raise error if neither model nor agent_factory provided"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with pytest.raises(ValueError, match="Either 'model' or 'agent_factory'"):
            ReplannerAgent(plan_id="test_plan")


class TestReplannerAgentBuildPrompt:
    """Tests for _build_replan_prompt method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def test_build_prompt_includes_completed_steps(self):
        """Should include completed steps tool history"""
        plan = Plan(
            steps=["S0", "S1", "S2"],
            dependencies={1: [0], 2: [1]}
        )
        plan.mark_step(0, step_status="completed")
        plan.add_tool_call(0, "read_file", {"path": "x.py"}, "content", "ts")
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1, 2],
            available_tools=["tool_a", "tool_b"]
        )

        assert "Step 0" in prompt
        assert "read_file" in prompt
        assert "completed" in prompt.lower() or "已完成" in prompt.lower()

    def test_build_prompt_includes_failed_steps(self):
        """Should include failed steps info"""
        plan = Plan(steps=["S0", "S1", "S2"])
        plan.mark_step(0, step_status="completed")
        plan.add_tool_call_pending(1, "write_file", {"path": "x.py"}, "ts")
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1, 2],
            available_tools=["tool_a"]
        )

        assert "S1" in prompt
        assert "S2" in prompt

    def test_build_prompt_includes_available_tools(self):
        """Should include available tools list"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1],
            available_tools=["write_file", "run_command", "read_file"]
        )

        assert "write_file" in prompt
        assert "run_command" in prompt


class TestReplannerAgentParseResponse:
    """Tests for _parse_replan_response method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def test_parse_redesign_response(self):
        """Should parse redesign response with steps and dependencies"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = """
        Based on the failure, I'll redesign the steps.

        ```json
        {
            "action": "redesign",
            "new_steps": ["Build API framework", "Implement endpoints", "Write tests"],
            "new_dependencies": {"1": [0], "2": [1]}
        }
        ```
        """

        result = agent._parse_replan_response(response)

        assert result.action == "redesign"
        assert len(result.new_steps) == 3
        assert result.new_dependencies == {1: [0], 2: [1]}

    def test_parse_give_up_response(self):
        """Should parse give_up response"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = """
        The task cannot be completed with available tools.

        ```json
        {
            "action": "give_up",
            "new_steps": [],
            "new_dependencies": {}
        }
        ```
        """

        result = agent._parse_replan_response(response)

        assert result.action == "give_up"
        assert result.new_steps == []

    def test_parse_response_handles_malformed_json(self):
        """Should return give_up on malformed response"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = "This response has no valid JSON"

        result = agent._parse_replan_response(response)

        assert result.action == "give_up"


class TestReplannerAgentReplanSubgraph:
    """Tests for replan_subgraph method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    @pytest.mark.asyncio
    async def test_replan_subgraph_calls_execute(self):
        """Should call execute with built prompt"""
        plan = Plan(steps=["S0", "S1", "S2"])
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {"action": "redesign", "new_steps": ["New S1"], "new_dependencies": {}}
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan_subgraph(
            steps_to_replan=[1, 2],
            available_tools=["tool_a"]
        )

        agent.execute.assert_called_once()
        assert result.action == "redesign"

    @pytest.mark.asyncio
    async def test_replan_subgraph_returns_parsed_result(self):
        """Should return parsed ReplanResult"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {
            "action": "redesign",
            "new_steps": ["Step X", "Step Y"],
            "new_dependencies": {"1": [0]}
        }
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan_subgraph(
            steps_to_replan=[1],
            available_tools=[]
        )

        assert isinstance(result, ReplanResult)
        assert result.new_steps == ["Step X", "Step Y"]


class TestReplannerIntents:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_replan_result_includes_intents(self):
        """ReplanResult should include intents for new steps"""
        result = ReplanResult(
            action="redesign",
            new_steps=["Gen code", "Review code"],
            new_dependencies={1: [0]},
            new_intents={0: "generate", 1: "review"}
        )
        assert result.new_intents[0] == "generate"
        assert result.new_intents[1] == "review"

    def test_replan_result_default_intents(self):
        """ReplanResult should default intents to empty dict"""
        result = ReplanResult(
            action="redesign",
            new_steps=["Step A"],
            new_dependencies={}
        )
        assert result.new_intents == {}

    def test_parse_response_extracts_intents(self):
        """Should parse intents from JSON response"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p1", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p1", model=MagicMock())

        response = '''```json
        {
            "action": "redesign",
            "new_steps": ["Gen code", "Review"],
            "new_dependencies": {"1": [0]},
            "new_intents": {"0": "generate", "1": "review"}
        }
        ```'''
        result = replanner._parse_replan_response(response)
        assert result.new_intents == {0: "generate", 1: "review"}

    def test_parse_response_missing_intents(self):
        """Should handle missing intents in JSON response"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p2", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p2", model=MagicMock())

        response = '''```json
        {
            "action": "redesign",
            "new_steps": ["Step A"],
            "new_dependencies": {}
        }
        ```'''
        result = replanner._parse_replan_response(response)
        assert result.new_intents == {}

    def test_available_intents_stored(self):
        """Should store available_intents on init"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p3", plan)

        intents = {"default": "General", "generate": "Gen code"}
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(
                plan_id="p3", model=MagicMock(),
                available_intents=intents
            )
        assert replanner.available_intents == intents

    def test_available_intents_in_prompt(self):
        """Should include available intents in replan prompt"""
        plan = Plan(steps=["A", "B"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("p4", plan)

        intents = {"default": "General", "generate": "Gen code", "review": "Review code"}
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(
                plan_id="p4", model=MagicMock(),
                available_intents=intents
            )

        prompt = replanner._build_replan_prompt(
            steps_to_replan=[1],
            available_tools=["tool_a"]
        )

        assert "generate" in prompt
        assert "review" in prompt
        assert "Available Intents" in prompt
