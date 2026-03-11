"""Test script for Flowchart Agent."""

from __future__ import annotations

import asyncio
import logging
import warnings

from dotenv import load_dotenv

from app.config import CortexConfig
from cortex import Cortex

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


async def main() -> None:
    config = CortexConfig()  # type: ignore[call-arg]
    cortex = Cortex(config)

    query = "畫一個簡單的用戶登入流程圖：開始 -> 輸入帳號密碼 -> 驗證 -> 成功/失敗 -> 結束"
    print(f"Query: {query}\n")

    result = await cortex.execute(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
