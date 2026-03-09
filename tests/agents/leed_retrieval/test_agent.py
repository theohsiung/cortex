"""Tests for create_leed_retrieval_agent factory."""

from __future__ import annotations

from unittest.mock import patch


class TestCreateLeedRetrievalAgent:
    def test_create_agent_returns_agent_instance(self) -> None:
        """create_leed_retrieval_agent 應回傳 LeedRetrievalAgent 實例"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from leed_retrieval._agent import LeedRetrievalAgent, create_leed_retrieval_agent

            agent = create_leed_retrieval_agent(
                model_name="openai/test-model",
                api_base="http://localhost:8000/v1",
                api_key="test-key",
            )
            assert isinstance(agent, LeedRetrievalAgent)

    def test_create_agent_no_args(self) -> None:
        """無參數呼叫也應可建立 agent"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from leed_retrieval._agent import create_leed_retrieval_agent

            agent = create_leed_retrieval_agent()
            assert agent is not None

    def test_agent_name(self) -> None:
        """Agent 名稱應為 LeedRetrievalAgent"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from leed_retrieval._agent import create_leed_retrieval_agent

            agent = create_leed_retrieval_agent()
            assert agent.name == "LeedRetrievalAgent"

    def test_no_alias_tools(self) -> None:
        """Agent 不應包含 alias 工具（避免 token 溢出）"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from leed_retrieval._agent import create_leed_retrieval_agent

            agent = create_leed_retrieval_agent()
            tool_names = [getattr(t, "name", "") for t in agent.tools]
            alias_tools = [
                n for n in tool_names if "json" in n or "commentary" in n or "<|channel|>" in n
            ]
            assert alias_tools == [], f"Found alias tools: {alias_tools}"

    def test_flow_excludes_heavy_processors(self) -> None:
        """Flow 不應包含 _nl_planning 和 _code_execution（節省 token）"""
        from google.adk.flows.llm_flows import _code_execution, _nl_planning

        from leed_retrieval._agent import LeedRetrievalFlow

        flow = LeedRetrievalFlow()
        req_processors = flow.request_processors
        resp_processors = flow.response_processors

        assert _nl_planning.request_processor not in req_processors
        assert _code_execution.request_processor not in req_processors
        assert _nl_planning.response_processor not in resp_processors
        assert _code_execution.response_processor not in resp_processors
