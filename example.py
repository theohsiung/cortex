"""
Example script to run Cortex.

Usage:
    uv run python example.py
"""

import asyncio
import warnings

# Filter Pydantic warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from cortex import Cortex

# Model configuration
API_BASE_URL = "http://10.136.3.209:8000/v1"
API_BASE_URL = "http://deltallm-proxy.10.143.156.8.sslip.io"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "gpt-oss-20b"


async def main():
    # Initialize model via LiteLLM
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
        api_key="sk-LR0Tm1AzzJp75jfzzG1jzQ",
    )

    # Create Cortex (default mode)
    cortex = Cortex(model=model)

    # --- Custom agent factory example ---
    # from google.adk.agents import LoopAgent
    #
    # def my_planner_factory(tools: list):
    #     return LoopAgent(
    #         name="planner",
    #         model=model,
    #         tools=tools,  # toolkit tools are injected here
    #     )
    #
    # cortex = Cortex(planner_factory=my_planner_factory)
    # ------------------------------------

    # Execute a task
    query = "寫一篇短篇兒童小說"
    # query = "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
