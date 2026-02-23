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

from app.config import CortexConfig
from cortex import Cortex


async def main() -> None:
    """Run Cortex with a sample query."""
    config = CortexConfig()  # type: ignore[call-arg]
    cortex = Cortex(config)

    query = (
        "I’m researching species that became invasive after people who kept them as pets"
        " released them. There’s a certain species of fish that was popularized as a pet"
        " by being the main character of the movie Finding Nemo. According to the USGS,"
        " where was this fish found as a nonnative species, before the year 2020? I need"
        " the answer formatted as the five-digit zip codes of the places the species was"
        " found, separated by commas if there is more than one place."
    )
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
