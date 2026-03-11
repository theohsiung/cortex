"""LEED Advisor tool — calls LEED MCP server and parses streaming response."""

from __future__ import annotations

import json
import logging
import os

import requests as requests
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)


def parse_streaming_chunks(raw: str) -> str:
    """Parse concatenated streaming JSON chunks into plain text."""
    if not raw:
        return ""
    content = ""
    for chunk_str in raw.replace("}{", "}\n{").split("\n"):
        try:
            obj = json.loads(chunk_str)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content += delta["content"]
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return content


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

    resp = requests.post(
        url,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "leed_orchestrator",
                "arguments": {"query": query},
            },
        },
        timeout=120,
    )
    resp.raise_for_status()

    data = resp.json()
    raw_text = data.get("result", {}).get("content", [{}])[0].get("text", "")
    logger.info("MCP raw response text:\n%s", raw_text)
    parsed = parse_streaming_chunks(raw_text)
    logger.info("MCP parsed result:\n%s", parsed)
    return parsed


leed_advisor_tool = FunctionTool(leed_query)
