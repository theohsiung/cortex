"""Tests for create_general_task_agent factory."""

from __future__ import annotations

from unittest.mock import patch


class TestCreateGeneralTaskAgent:
    def test_create_agent_returns_general_task_agent(self) -> None:
        """create_general_task_agent 應回傳 GeneralTaskAgent 實例"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from general_task._agent import GeneralTaskAgent, create_general_task_agent

            agent = create_general_task_agent(
                model_name="openai/test-model",
                api_base="http://localhost:8000/v1",
                api_key="test-key",
            )
            assert isinstance(agent, GeneralTaskAgent)

    def test_create_agent_no_args(self) -> None:
        """無參數呼叫也應可建立 agent（供 ExecutorAgent 使用）"""
        with patch("google.adk.models.lite_llm.LiteLlm"):
            from general_task._agent import create_general_task_agent

            agent = create_general_task_agent()
            assert agent is not None

    def test_agent_as_executor_factory(self) -> None:
        """create_general_task_agent 應可作為 ExecutorAgent 的 agent_factory"""
        from app.agents.executor.executor_agent import ExecutorAgent
        from app.task.plan import Plan
        from app.task.task_manager import TaskManager

        plan = Plan(title="test", steps=["step 1"])
        TaskManager.set_plan("plan_test_general", plan)

        with patch("google.adk.models.lite_llm.LiteLlm"):
            from general_task._agent import create_general_task_agent

            executor = ExecutorAgent(
                plan_id="plan_test_general",
                agent_factory=create_general_task_agent,
            )
            assert executor is not None
