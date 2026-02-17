"""Example script to run Cortex.

Usage:
    uv run python example.py
"""

from __future__ import annotations

import asyncio
import logging
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

from cortex import Cortex
from app.config import CortexConfig


async def main() -> None:
    """Run Cortex with a sample query."""
    config = CortexConfig()
    cortex = Cortex(config)

    query = "調用agent檢查以下python程式碼有沒有錯誤?:print(我在家)"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
