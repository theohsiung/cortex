"""
Example script to run Cortex.

Usage:
    uv run python example.py
"""

import asyncio
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# Configure logging to see Cortex execution status
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Filter Pydantic warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from google.adk.models import LiteLlm
from cortex import Cortex

# Model configuration
API_BASE_URL = "http://deltallm-proxy.10.143.156.8.sslip.io"
MODEL_NAME = "gpt-oss-20b"


async def main():
    # Initialize model via LiteLLM
    model = LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=API_BASE_URL,
        api_key=os.getenv("DELTALLM_API_KEY"),
    )

    # Create Cortex (default mode)
    # cortex = Cortex(model=model)

    # --- Dynamic executor routing example ---
    from app.agents.coding_agent.agent.mistral_vibe._agent import create_coding_agent

    api_key = os.getenv("DELTALLM_API_KEY")
    coding_agent_kwargs = {
        "model_name": f"openai/{MODEL_NAME}",
        "api_base": API_BASE_URL,
        "api_key": api_key,
    }

    cortex = Cortex(
        model=model,
        executors={
            "generate": {
                "factory": lambda: create_coding_agent(
                    "/workspace", **coding_agent_kwargs
                ),
                "description": "Generate new code",
            },
            "fix": {
                "factory": lambda: create_coding_agent(
                    "/workspace", **coding_agent_kwargs
                ),
                "description": "Fix code based on review feedback",
            },
            "review": {
                "factory": lambda: create_coding_agent(
                    "/workspace", **coding_agent_kwargs
                ),
                "description": "Review code quality and correctness",
            },
        },
    )
    # ------------------------------------

    # Execute a task
    query = "調用agent檢查以下python程式碼有沒有錯誤?:print(我在家)"
    # query = "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
