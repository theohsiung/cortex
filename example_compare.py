"""
Compare single LLM call vs Cortex multi-step execution.

Usage:
    uv run python example_compare.py
"""

import asyncio
import time
import warnings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from cortex import Cortex

# Model configuration
API_BASE_URL = "http://deltallm-proxy.10.143.156.8.sslip.io"
MODEL_NAME = "gpt-oss-20b"


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


async def cortex_call(model, query: str) -> tuple[str, float]:
    """Cortex multi-step execution with planning and parallel execution."""
    start = time.time()

    cortex = Cortex(model=model)
    result = await cortex.execute(query)

    elapsed = time.time() - start
    return result, elapsed


async def main():
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
        api_key="sk-LR0Tm1AzzJp75jfzzG1jzQ",
    )
    # model = LiteLlm(
    #     model=f"gemini/gemini-2.5-flash",
    #     api_key="AIzaSyD1_AEknDmF_gjdlLJoUc-1UDUv3kBpwRE",
    # )
    # model = LiteLlm(
    #     model=f"openai/Qwen/Qwen3-4B-Instruct-2507",
    #     api_base="http://0.0.0.0:8000/v1",
    #     api_key="dummy",
    # )

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
    cortex_result, cortex_time = await cortex_call(model, query)
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
