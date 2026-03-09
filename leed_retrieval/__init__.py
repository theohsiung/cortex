"""ADK CLI entry point for LEED Retrieval Agent."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ._agent import create_leed_retrieval_agent

# 載入 .env
agent_dir = Path(__file__).parent
load_dotenv(agent_dir / ".env")

root_agent = create_leed_retrieval_agent()
