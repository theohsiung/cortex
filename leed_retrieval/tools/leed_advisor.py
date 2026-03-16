"""LEED Advisor tool — calls LEED MCP server."""

from __future__ import annotations

import logging
import os
import time

import requests as requests
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


def leed_query(query: str) -> str:
    """查詢 LEED 綠建築相關資訊（標準、案例、產品）。

    Args:
        query: LEED 相關的自然語言查詢
    """
    url = os.getenv("LEED_MCP_URL", "")
    token = os.getenv("LEED_MCP_TOKEN", "")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "leed_orchestrator",
            "arguments": {"query": query},
        },
    }

    last_err: requests.exceptions.ConnectionError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("result", {}).get("content", [{}])[0].get("text", "")
            logger.info("MCP result:\n%s", text[:500])
            return str(text)
        except requests.exceptions.ConnectionError as e:
            last_err = e
            logger.warning("leed_query attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise last_err or RuntimeError("leed_query failed after all retries")


leed_advisor_tool = FunctionTool(leed_query)
