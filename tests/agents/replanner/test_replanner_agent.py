"""Tests for ReplannerAgent - redesigns the remaining plan when a step fails."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.replanner.replanner_agent import (
    FailureRecord,
    ReplannerAgent,
    ReplanResult,
)
from app.task.plan import Plan
from app.task.task_manager import TaskManager


class TestReplanResult:
    """Tests for ReplanResult dataclass"""

    def test_replan_result_redesign(self):
        """Should create ReplanResult with redesign action and all fields populated"""
        result = ReplanResult(
            action="redesign",
            failed_step_description="New approach to data extraction",
            failed_step_intent="generate",
            continuation_steps={0: "Step A", 1: "Step B"},
            continuation_dependencies={1: [0]},
            continuation_intents={0: "generate", 1: "review"},
        )

        assert result.action == "redesign"
        assert result.failed_step_description == "New approach to data extraction"
        assert result.failed_step_intent == "generate"
        assert result.continuation_steps == {0: "Step A", 1: "Step B"}
        assert result.continuation_dependencies == {1: [0]}
        assert result.continuation_intents == {0: "generate", 1: "review"}

    def test_replan_result_give_up(self):
        """Should create ReplanResult with give_up action and empty defaults"""
        result = ReplanResult(action="give_up")

        assert result.action == "give_up"
        assert result.failed_step_description == ""
        assert result.failed_step_intent == ""
        assert result.continuation_steps == {}
        assert result.continuation_dependencies == {}
        assert result.continuation_intents == {}

    def test_replan_result_defaults(self):
        """ReplanResult should default all optional fields to empty values"""
        result = ReplanResult(action="redesign")

        assert result.failed_step_description == ""
        assert result.failed_step_intent == ""
        assert result.continuation_steps == {}
        assert result.continuation_dependencies == {}
        assert result.continuation_intents == {}


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
    """Tests for _build_prompt method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def _make_agent(self, plan):
        TaskManager.set_plan("test_plan", plan)
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            return ReplannerAgent(plan_id="test_plan", model=MagicMock())

    def _default_prompt_kwargs(self, **overrides):
        """Return default kwargs for _build_prompt with optional overrides."""
        defaults = dict(
            original_query="Find the ZIP code for clownfish sightings",
            failed_step_id=1,
            failed_step_desc="Download data from USGS",
            failed_output="Downloaded 1 record but no ZIP field",
            failed_reason="Executor did not find ZIP code",
            failed_tool_history=[],
            attempt=1,
            max_attempts=3,
            available_tools=["tool_a", "tool_b"],
        )
        defaults.update(overrides)
        return defaults

    def test_build_prompt_includes_original_query(self):
        """Should include the original task query"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(**self._default_prompt_kwargs())

        assert "Find the ZIP code for clownfish sightings" in prompt
        assert "Original Task" in prompt

    def test_build_prompt_includes_completed_steps(self):
        """Should include completed steps and their tool history"""
        plan = Plan(steps=["S0", "S1", "S2"], dependencies={1: [0], 2: [1]})
        plan.mark_step(0, step_status="completed")
        plan.add_tool_call(0, "read_file", {"path": "x.py"}, "content", "ts")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(**self._default_prompt_kwargs(failed_step_id=2))

        assert "Step 0" in prompt
        assert "read_file" in prompt

    def test_build_prompt_includes_failed_step_info(self):
        """Should include failed step description and reason"""
        plan = Plan(steps=["S0", "S1", "S2"])
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(
                failed_step_id=2,
                failed_step_desc="Extract data from table",
                failed_reason="Executor did not call any tools",
            )
        )

        assert "Step 2" in prompt
        assert "Extract data from table" in prompt
        assert "Executor did not call any tools" in prompt

    def test_build_prompt_includes_available_tools(self):
        """Should include available tools list"""
        plan = Plan(steps=["S0", "S1"])
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(
                available_tools=["write_file", "run_command", "read_file"]
            )
        )

        assert "write_file" in prompt
        assert "run_command" in prompt
        assert "read_file" in prompt

    def test_build_prompt_includes_attempt_info_when_retry(self):
        """Should include attempt number when attempt > 1"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(**self._default_prompt_kwargs(attempt=2, max_attempts=3))

        assert "2" in prompt and "3" in prompt
        assert "different strategy" in prompt.lower() or "attempt" in prompt.lower()

    def test_build_prompt_includes_failure_notes(self):
        """Should include the failure reason text"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(
                failed_reason="Found geo coords but no ZIP code field",
            )
        )

        assert "Found geo coords" in prompt
        assert "no ZIP code field" in prompt

    def test_build_prompt_truncates_long_output(self):
        """Should truncate executor output that exceeds 2000 chars"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        long_output = "x" * 2500 + "IMPORTANT_ENDING"
        prompt = agent._build_prompt(**self._default_prompt_kwargs(failed_output=long_output))

        assert "IMPORTANT_ENDING" in prompt

    def test_build_prompt_includes_tool_history(self):
        """Should include failed step tool history when provided"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(
                failed_tool_history=[
                    {"tool": "download_file", "args": {"url": "http://example.com"}, "result": "ok"}
                ],
            )
        )

        assert "download_file" in prompt


class TestReplannerAgentParseResponse:
    """Tests for _parse_response method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def _make_agent(self):
        plan = Plan(steps=["A"])
        TaskManager.set_plan("test_plan", plan)
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            return ReplannerAgent(plan_id="test_plan", model=MagicMock())

    def test_parse_redesign_response(self):
        """Should parse redesign response with new format fields"""
        agent = self._make_agent()

        response = """
        Based on the failure, I'll redesign the steps.

        ```json
        {
            "action": "redesign",
            "failed_step_description": "New approach to data extraction",
            "failed_step_intent": "generate",
            "continuation_steps": {
                "0": "Build API framework",
                "1": "Implement endpoints",
                "2": "Write tests"
            },
            "continuation_dependencies": {"1": [0], "2": [1]},
            "continuation_intents": {"0": "generate", "1": "generate", "2": "review"}
        }
        ```
        """

        result = agent._parse_response(response)

        assert result.action == "redesign"
        assert result.failed_step_description == "New approach to data extraction"
        assert result.failed_step_intent == "generate"
        assert len(result.continuation_steps) == 3
        assert result.continuation_steps[0] == "Build API framework"
        assert result.continuation_dependencies == {1: [0], 2: [1]}
        assert result.continuation_intents == {0: "generate", 1: "generate", 2: "review"}

    def test_parse_give_up_response(self):
        """Should parse give_up response with empty defaults"""
        agent = self._make_agent()

        response = """
        The task cannot be completed with available tools.

        ```json
        {"action": "give_up"}
        ```
        """

        result = agent._parse_response(response)

        assert result.action == "give_up"
        assert result.failed_step_description == ""
        assert result.failed_step_intent == ""
        assert result.continuation_steps == {}
        assert result.continuation_dependencies == {}
        assert result.continuation_intents == {}

    def test_parse_response_handles_malformed_json(self):
        """Should return None on malformed response (caller retries)"""
        agent = self._make_agent()

        response = "This response has no valid JSON"

        result = agent._parse_response(response)

        assert result is None

    def test_parse_response_extracts_intents(self):
        """Should parse continuation_intents and failed_step_intent from JSON response"""
        agent = self._make_agent()

        response = """```json
    {
        "action": "redesign",
        "failed_step_description": "Revised step",
        "failed_step_intent": "generate",
        "continuation_steps": {"0": "Gen code", "1": "Review"},
        "continuation_dependencies": {"1": [0]},
        "continuation_intents": {"0": "generate", "1": "review"}
    }
    ```"""

        result = agent._parse_response(response)

        assert result.failed_step_intent == "generate"
        assert result.continuation_intents == {0: "generate", 1: "review"}

    def test_parse_response_no_continuation(self):
        """Should handle response with no continuation_steps"""
        agent = self._make_agent()

        response = """```json
    {
        "action": "redesign",
        "failed_step_description": "Retry with different approach",
        "failed_step_intent": "default"
    }
    ```"""

        result = agent._parse_response(response)

        assert result is not None
        assert result.action == "redesign"
        assert result.failed_step_description == "Retry with different approach"
        assert result.continuation_steps == {}
        assert result.continuation_dependencies == {}
        assert result.continuation_intents == {}

    def test_parse_response_raw_json_without_code_block(self):
        """Should parse JSON even without code block markers"""
        agent = self._make_agent()

        response = """Here is the plan:
    {"action": "redesign", "failed_step_description": "New approach",
     "failed_step_intent": "default", "continuation_steps": {"0": "New step"},
     "continuation_dependencies": {}, "continuation_intents": {"0": "generate"}}
    """

        result = agent._parse_response(response)

        assert result is not None
        assert result.action == "redesign"
        assert result.failed_step_description == "New approach"
        assert result.continuation_steps == {0: "New step"}
        assert result.continuation_intents == {0: "generate"}


class TestReplannerAgentReplan:
    """Tests for replan method"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def _default_replan_kwargs(self, **overrides):
        defaults = dict(
            original_query="Find ZIP codes",
            failed_step_id=1,
            failed_step_desc="Download data",
            failed_output="No results found",
            failed_reason="Executor failed",
            failed_tool_history=[],
            attempt=1,
            max_attempts=3,
            available_tools=["tool_a"],
        )
        defaults.update(overrides)
        return defaults

    @pytest.mark.asyncio
    async def test_replan_calls_execute(self):
        """Should call execute with built prompt"""
        plan = Plan(steps=["S0", "S1", "S2"])
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {"action": "redesign", "failed_step_description": "New S1",
         "failed_step_intent": "default", "continuation_steps": {"0": "Follow-up"},
         "continuation_dependencies": {}, "continuation_intents": {"0": "generate"}}
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan(**self._default_replan_kwargs())

        agent.execute.assert_called_once()
        assert result.action == "redesign"

    @pytest.mark.asyncio
    async def test_replan_returns_parsed_result(self):
        """Should return parsed ReplanResult with new fields"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {
            "action": "redesign",
            "failed_step_description": "Revised approach",
            "failed_step_intent": "generate",
            "continuation_steps": {"0": "Step X", "1": "Step Y"},
            "continuation_dependencies": {"1": [0]},
            "continuation_intents": {"0": "generate", "1": "review"}
        }
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan(**self._default_replan_kwargs())

        assert isinstance(result, ReplanResult)
        assert result.failed_step_description == "Revised approach"
        assert result.failed_step_intent == "generate"
        assert result.continuation_steps == {0: "Step X", 1: "Step Y"}
        assert result.continuation_dependencies == {1: [0]}
        assert result.continuation_intents == {0: "generate", 1: "review"}

    @pytest.mark.asyncio
    async def test_replan_retries_on_parse_failure(self):
        """Should retry up to 3 times on JSON parse failure then give up"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = "This has no valid JSON"

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan(**self._default_replan_kwargs())

        assert agent.execute.call_count == 3
        assert result.action == "give_up"

    @pytest.mark.asyncio
    async def test_replan_with_intents(self):
        """Should return result with continuation_intents and failed_step_intent"""
        plan = Plan(steps=["S0", "S1"])
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {
            "action": "redesign",
            "failed_step_description": "Generate code differently",
            "failed_step_intent": "generate",
            "continuation_steps": {"0": "Gen code", "1": "Review"},
            "continuation_dependencies": {"1": [0]},
            "continuation_intents": {"0": "generate", "1": "review"}
        }
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        result = await agent.replan(**self._default_replan_kwargs())

        assert result.failed_step_intent == "generate"
        assert result.continuation_intents == {0: "generate", 1: "review"}

    @pytest.mark.asyncio
    async def test_replan_passes_all_params_to_prompt(self):
        """Should pass all parameters through to _build_prompt"""
        plan = Plan(steps=["S0", "S1"])
        plan.mark_step(0, step_status="completed")
        TaskManager.set_plan("test_plan", plan)

        mock_result = MagicMock()
        mock_result.output = """
        ```json
        {"action": "redesign", "failed_step_description": "New",
         "failed_step_intent": "default", "continuation_steps": {},
         "continuation_dependencies": {}, "continuation_intents": {}}
        ```
        """

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            agent = ReplannerAgent(plan_id="test_plan", model=MagicMock())
            agent.execute = AsyncMock(return_value=mock_result)

        tool_hist = [{"tool": "web_search", "args": {"q": "test"}, "result": "ok"}]
        await agent.replan(
            original_query="Find ZIP codes for Florida",
            failed_step_id=1,
            failed_step_desc="Download data from USGS",
            failed_output="Downloaded 1 record...",
            failed_reason="No ZIP code field found",
            failed_tool_history=tool_hist,
            attempt=2,
            max_attempts=3,
            available_tools=["web_search", "python_executor"],
        )

        # Verify execute was called with a prompt containing the key info
        call_args = agent.execute.call_args
        prompt = call_args[0][0]
        assert "Find ZIP codes for Florida" in prompt
        assert "Download data from USGS" in prompt
        assert "No ZIP code field found" in prompt
        assert "web_search" in prompt


class TestReplannerIntents:
    """Tests for intent handling in ReplannerAgent"""

    def setup_method(self):
        TaskManager._plans.clear()

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

        prompt = replanner._build_prompt(
            original_query="Build a web app",
            failed_step_id=1,
            failed_step_desc="Generate code",
            failed_output="",
            failed_reason="Failed",
            failed_tool_history=[],
            attempt=1,
            max_attempts=3,
            available_tools=["tool_a"],
        )

        assert "generate" in prompt
        assert "review" in prompt
        assert "Available Intents" in prompt

    def test_available_intents_defaults_empty(self):
        """Should default available_intents to empty dict"""
        plan = Plan(steps=["A"], dependencies={})
        TaskManager.set_plan("p5", plan)

        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            replanner = ReplannerAgent(plan_id="p5", model=MagicMock())
        assert replanner.available_intents == {}


class TestFailureRecord:
    """Tests for FailureRecord dataclass"""

    def test_failure_record_creation(self):
        """Should create FailureRecord with all fields"""
        record = FailureRecord(
            step_description="Search for weather data",
            failure_notes="[FAIL]: API returned 403 | No data retrieved",
            tool_history=[
                {
                    "tool": "web_search",
                    "args": {"query": "weather"},
                    "status": "success",
                    "result": "403 Forbidden",
                }
            ],
        )
        assert record.step_description == "Search for weather data"
        assert record.failure_notes == "[FAIL]: API returned 403 | No data retrieved"
        assert len(record.tool_history) == 1
        assert record.tool_history[0]["tool"] == "web_search"

    def test_failure_record_empty_tool_history(self):
        """Should work with empty tool history"""
        record = FailureRecord(
            step_description="Some step",
            failure_notes="[FAIL]: something wrong",
            tool_history=[],
        )
        assert record.tool_history == []


class TestFailureHistoryInPrompt:
    """Tests for failure history appearing in replan prompt"""

    def setup_method(self):
        TaskManager.remove_plan("test_plan")

    def teardown_method(self):
        TaskManager.remove_plan("test_plan")

    def _make_agent(self, plan):
        TaskManager.set_plan("test_plan", plan)
        with patch("app.agents.replanner.replanner_agent._get_llm_agent"):
            return ReplannerAgent(plan_id="test_plan", model=MagicMock())

    def _default_prompt_kwargs(self, **overrides):
        defaults = dict(
            original_query="Get weather",
            failed_step_id=1,
            failed_step_desc="Search weather",
            failed_output="output",
            failed_reason="[FAIL]: timeout",
            failed_tool_history=[],
            attempt=2,
            max_attempts=3,
            available_tools=["tool_a"],
        )
        defaults.update(overrides)
        return defaults

    def test_prompt_includes_failure_history(self):
        """Should include past failure records in prompt"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        records = [
            FailureRecord(
                step_description="Search weather with web_search",
                failure_notes="[FAIL]: API returned 403",
                tool_history=[
                    {
                        "tool": "web_search",
                        "args": {"query": "weather"},
                        "status": "success",
                        "result": "403",
                    }
                ],
            ),
        ]

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(),
            failure_history=records,
        )

        assert "Search weather with web_search" in prompt
        assert "API returned 403" in prompt
        assert "web_search" in prompt

    def test_prompt_shows_multiple_failure_records(self):
        """Should show all past failure records in order"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        records = [
            FailureRecord(
                step_description="Attempt with API v1",
                failure_notes="[FAIL]: v1 deprecated",
                tool_history=[],
            ),
            FailureRecord(
                step_description="Attempt with API v2",
                failure_notes="[FAIL]: auth required",
                tool_history=[
                    {
                        "tool": "fetch_url",
                        "args": {"url": "http://api.v2"},
                        "status": "success",
                        "result": "401",
                    }
                ],
            ),
        ]

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(attempt=3, max_attempts=5),
            failure_history=records,
        )

        assert "v1 deprecated" in prompt
        assert "auth required" in prompt
        assert "Attempt with API v1" in prompt
        assert "Attempt with API v2" in prompt

    def test_prompt_no_failure_history_section_when_empty(self):
        """Should not include failure history section when no past failures"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(
            **self._default_prompt_kwargs(attempt=1),
            failure_history=[],
        )

        assert "Past Failed Attempts" not in prompt

    def test_prompt_no_failure_history_section_when_none(self):
        """Should not include failure history section when None (backward compat)"""
        plan = Plan(steps=["S0", "S1"], dependencies={1: [0]})
        plan.mark_step(0, step_status="completed")
        agent = self._make_agent(plan)

        prompt = agent._build_prompt(**self._default_prompt_kwargs())

        assert "Past Failed Attempts" not in prompt
