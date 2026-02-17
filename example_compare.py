"""Compare single LLM call vs Cortex multi-step execution.

Usage:
    uv run python example_compare.py
"""

from __future__ import annotations

import asyncio
import time
import warnings

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from cortex import Cortex
from app.config import CortexConfig


async def single_llm_call(model, query: str) -> tuple[str, float]:
    """Direct single LLM call without planning/execution steps."""
    start = time.time()

    agent = LlmAgent(
        name="direct",
        model=model,
        instruction="You are a helpful assistant. Complete the user's request directly.",
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="direct", user_id="user")

    runner = Runner(agent=agent, session_service=session_service, app_name="direct")
    content = Content(parts=[Part(text=query)])

    result = ""
    async for event in runner.run_async(
        user_id="user", session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content:
            if hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        result = part.text

    elapsed = time.time() - start
    return result, elapsed


async def cortex_call(config: CortexConfig, query: str) -> tuple[str, float]:
    """Cortex multi-step execution with planning and parallel execution."""
    start = time.time()

    cortex = Cortex(config)
    result = await cortex.execute(query)

    elapsed = time.time() - start
    return result, elapsed


async def main() -> None:
    """Run comparison between single LLM call and Cortex multi-step execution."""
    config = CortexConfig()

    # For single_llm_call, create model from config
    model = LiteLlm(
        model=config.model.name,
        api_base=config.model.api_base,
        api_key=config.model.resolve_api_key(),
    )

    query = "寫一篇短篇中文兒童故事"
    print(f"Query: {query}\n")
    print("=" * 80)

    # Single LLM call
    print("\n[1] SINGLE LLM CALL")
    print("-" * 40)
    single_result, single_time = await single_llm_call(model, query)
    print(single_result)
    print(f"\nTime: {single_time:.2f}s")

    print("\n" + "=" * 80)

    # Cortex multi-step
    print("\n[2] CORTEX MULTI-STEP")
    print("-" * 40)
    cortex_result, cortex_time = await cortex_call(config, query)
    print(cortex_result)
    print(f"\nTime: {cortex_time:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("-" * 40)
    print(f"Single LLM:  {single_time:.2f}s")
    print(f"Cortex:      {cortex_time:.2f}s")
    print(f"Difference:  {cortex_time - single_time:+.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
