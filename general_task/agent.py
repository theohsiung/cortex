"""ADK CLI entry point for General Task Agent."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ._agent import create_general_task_agent

# 載入 .env
agent_dir = Path(__file__).parent
load_dotenv(agent_dir / ".env")

root_agent = create_general_task_agent()
