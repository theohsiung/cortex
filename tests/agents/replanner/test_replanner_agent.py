"""Tests for ReplannerAgent - subgraph redesign after verification failure"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.replanner.replanner_agent import (
    ReplanContext,
    ReplannerAgent,
    ReplanResult,
    RetryStepInfo,
)
from app.task.plan import Plan
from app.task.task_manager import TaskManager


class TestReplanResult:
    """Tests for ReplanResult dataclass"""

    def test_replan_result_redesign(self):
        """Should create ReplanResult with redesign action"""
        result = ReplanResult(
            action="redesign", new_steps={0: "Step A", 1: "Step B"}, new_dependencies={1: [0]}
        )

        assert result.action == "redesign"
        assert result.new_steps == {0: "Step A", 1: "Step B"}
        assert result.new_dependencies == {1: [0]}

    def test_replan_result_give_up(self):
        """Should create ReplanResult with give_up action"""
        result = ReplanResult(action="give_up", new_steps={}, new_dependencies={})

        assert result.action == "give_up"
        assert result.new_steps == {}

    def test_replan_result_with_retry_step(self):
        """ReplanResult should include retry_step info"""
        result = ReplanResult(
            action="redesign",
            retry_step=RetryStepInfo(description="Try different approach", intent="research"),
            new_steps={5: "Downstream step"},
            new_dependencies={5: [3]},
        )

        assert result.retry_step is not None
        assert result.retry_step.description == "Try different approach"
        assert result.retry_step.intent == "research"

    def test_replan_result_retry_step_defaults_none(self):
        """ReplanResult.retry_step should default to None"""
        result = ReplanResult(
            action="redesign",
            new_steps={5: "Step"},
            new_dependencies={},
        )

        assert result.retry_step is None


class TestReplanContext:
    """Tests for ReplanContext dataclass"""

    def test_replan_context_creation(self):
        """Should create ReplanContext with all fields"""
        ctx = ReplanContext(
            original_query="Find the ZIP code for clownfish sightings",
            failed_step_notes={3: "[FAIL]: Found geo coords | No ZIP code field"},
            failed_step_outputs={3: "Downloaded 1 record from USGS..."},
            failed_tool_history={3: [{"tool": "download_file", "status": "success"}]},
            attempt_number=1,
            max_attempts=2,
        )
        assert ctx.original_query == "Find the ZIP code for clownfish sightings"
        assert ctx.attempt_number == 1
        assert ctx.max_attempts == 2
        assert 3 in ctx.failed_step_notes
        assert 3 in ctx.failed_step_outputs
        assert 3 in ctx.failed_tool_history

    def test_replan_context_defaults_empty(self):
        """Should work with empty dicts"""
        ctx = ReplanContext(
            original_query="task",
            failed_step_notes={},
            failed_step_outputs={},
            failed_tool_history={},
            attempt_number=1,
            max_attempts=2,
        )
        assert ctx.failed_step_notes == {}


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
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")
        plan.add_tool_call(0, "read_file", {"path": "x.py"}, "content", "ts")
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1, 2], available_tools=["tool_a", "tool_b"]
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

        prompt = agent._build_replan_prompt(steps_to_replan=[1, 2], available_tools=["tool_a"])

        assert "S1" in prompt
        assert "S2" in prompt

    def test_build_prompt_includes_available_tools(self):
        """Should include available tools list"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1], available_tools=["write_file", "run_command", "read_file"]
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
            "new_steps": {
                "1": "Build API framework",
                "2": "Implement endpoints",
                "3": "Write tests"
            },
            "new_dependencies": {"2": [1], "3": [2]}
        }
        ```
        """

        result = agent._parse_replan_response(response)

        assert result.action == "redesign"
        assert len(result.new_steps) == 3
        assert result.new_dependencies == {2: [1], 3: [2]}

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
            "new_steps": {},
            "new_dependencies": {}
        }
        ```
        """

        result = agent._parse_replan_response(response)

        assert result.action == "give_up"
        assert result.new_steps == {}

    def test_parse_response_handles_malformed_json(self):
        """Should return None on malformed response (caller retries)"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = "This response has no valid JSON"

        result = agent._parse_replan_response(response)

        assert result is None

    def test_parse_response_extracts_retry_step(self):
        """Should extract retry_step from response"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = """```json
    {
        "action": "redesign",
        "retry_step": {
            "description": "Use local file instead of web search",
            "intent": "research"
        },
        "new_steps": {"3": "Process the local data"},
        "new_dependencies": {"3": [2]},
        "new_intents": {"3": "generate"}
    }
    ```"""

        result = agent._parse_replan_response(response)

        assert result is not None
        assert result.retry_step is not None
        assert result.retry_step.description == "Use local file instead of web search"
        assert result.retry_step.intent == "research"

    def test_parse_response_missing_retry_step(self):
        """Should handle missing retry_step gracefully"""
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        response = """```json
    {
        "action": "redesign",
        "new_steps": {"3": "New step"},
        "new_dependencies": {}
    }
    ```"""

        result = agent._parse_replan_response(response)

        assert result is not None
        assert result.retry_step is None


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
        {"action": "redesign", "new_steps": {"1": "New S1"}, "new_dependencies": {}}
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan_subgraph(steps_to_replan=[1, 2], available_tools=["tool_a"])

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
            "new_steps": {"2": "Step X", "3": "Step Y"},
            "new_dependencies": {"3": [2]}
        }
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan_subgraph(steps_to_replan=[1], available_tools=[])

        assert isinstance(result, ReplanResult)
        assert result.new_steps == {2: "Step X", 3: "Step Y"}


class TestReplannerIntents:
    def setup_method(self):
        TaskManager._plans.clear()

    def test_replan_result_includes_intents(self):
        """ReplanResult should include intents for new steps"""
        result = ReplanResult(
            action="redesign",
            new_steps={0: "Gen code", 1: "Review code"},
            new_dependencies={1: [0]},
            new_intents={0: "generate", 1: "review"},
        )
        assert result.new_intents[0] == "generate"
        assert result.new_intents[1] == "review"

    def test_replan_result_default_intents(self):
        """ReplanResult should default intents to empty dict"""
        result = ReplanResult(action="redesign", new_steps={0: "Step A"}, new_dependencies={})
        assert result.new_intents == {}

    def test_parse_response_extracts_intents(self):
        """Should parse intents from JSON response"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p1", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p1", model=MagicMock())

        response = """```json
        {
            "action": "redesign",
            "new_steps": {"1": "Gen code", "2": "Review"},
            "new_dependencies": {"2": [1]},
            "new_intents": {"1": "generate", "2": "review"}
        }
        ```"""
        result = replanner._parse_replan_response(response)
        assert result.new_intents == {1: "generate", 2: "review"}

    def test_parse_response_missing_intents(self):
        """Should handle missing intents in JSON response"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p2", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p2", model=MagicMock())

        response = """```json
        {
            "action": "redesign",
            "new_steps": {"1": "Step A"},
            "new_dependencies": {}
        }
        ```"""
        result = replanner._parse_replan_response(response)
        assert result.new_intents == {}

    def test_available_intents_stored(self):
        """Should store available_intents on init"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p3", plan)

        intents = {"default": "General", "generate": "Gen code"}
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p3", model=MagicMock(), available_intents=intents)
        assert replanner.available_intents == intents

    def test_available_intents_in_prompt(self):
        """Should include available intents in replan prompt"""
        plan = Plan(steps=["A", "B"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("p4", plan)

        intents = {"default": "General", "generate": "Gen code", "review": "Review code"}
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p4", model=MagicMock(), available_intents=intents)

        prompt = replanner._build_replan_prompt(steps_to_replan=[1], available_tools=["tool_a"])

        assert "generate" in prompt
        assert "review" in prompt
        assert "Available Intents" in prompt


class TestReplannerWithContext:
    """Tests for replanner with ReplanContext"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def test_build_prompt_includes_original_query(self):
        """Should include original task query when context provided"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        ctx = ReplanContext(
            original_query="Find the ZIP code for clownfish sightings in Florida",
            failed_step_notes={1: "[FAIL]: Found geo | No ZIP"},
            failed_step_outputs={1: "Downloaded 1 record..."},
            failed_tool_history={
                1: [{"tool": "download_file", "args": {}, "status": "success", "result": "ok"}]
            },
            attempt_number=1,
            max_attempts=2,
        )

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(
            steps_to_replan=[1], available_tools=["tool_a"], context=ctx
        )

        assert "Find the ZIP code for clownfish sightings in Florida" in prompt
        assert "Original Task" in prompt

    def test_build_prompt_includes_failure_notes(self):
        """Should include failure notes for failed steps"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        ctx = ReplanContext(
            original_query="task",
            failed_step_notes={1: "[FAIL]: Found geo coords | No ZIP code field"},
            failed_step_outputs={1: "output"},
            failed_tool_history={},
            attempt_number=1,
            max_attempts=2,
        )

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(steps_to_replan=[1], available_tools=[], context=ctx)

        assert "Found geo coords" in prompt
        assert "No ZIP code field" in prompt

    def test_build_prompt_includes_attempt_info(self):
        """Should include attempt number and max attempts"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        ctx = ReplanContext(
            original_query="task",
            failed_step_notes={},
            failed_step_outputs={},
            failed_tool_history={},
            attempt_number=2,
            max_attempts=3,
        )

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(steps_to_replan=[1], available_tools=[], context=ctx)

        assert "2" in prompt and "3" in prompt
        assert "different strategy" in prompt.lower() or "attempt" in prompt.lower()

    def test_build_prompt_truncates_long_output(self):
        """Should truncate executor output to last 500 chars"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        long_output = "x" * 300 + "IMPORTANT_ENDING"
        ctx = ReplanContext(
            original_query="task",
            failed_step_notes={},
            failed_step_outputs={1: long_output},
            failed_tool_history={},
            attempt_number=1,
            max_attempts=2,
        )

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        prompt = agent._build_replan_prompt(steps_to_replan=[1], available_tools=[], context=ctx)

        assert "IMPORTANT_ENDING" in prompt

    def test_build_prompt_without_context_backward_compatible(self):
        """Should work without context (backward compatible)"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())

        # No context param - should not raise
        prompt = agent._build_replan_prompt(steps_to_replan=[1], available_tools=["tool_a"])

        assert "S1" in prompt
        assert "Original Task" not in prompt

    @pytest.mark.asyncio
    async def test_replan_subgraph_accepts_context(self):
        """replan_subgraph should accept optional context parameter"""
        plan = Plan(steps=["S0", "S1"])
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = (
            '```json\n{"action": "redesign",'
            ' "new_steps": {"1": "New S1"},'
            ' "new_dependencies": {}}\n```'
        )

        ctx = ReplanContext(
            original_query="task",
            failed_step_notes={},
            failed_step_outputs={},
            failed_tool_history={},
            attempt_number=1,
            max_attempts=2,
        )

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan_subgraph(
            steps_to_replan=[1], available_tools=["tool_a"], context=ctx
        )

        assert result.action == "redesign"
        agent.execute.assert_called_once()
