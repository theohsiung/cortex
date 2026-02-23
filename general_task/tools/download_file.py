"""Download file tool."""

from __future__ import annotations

import os
from typing import Optional

from google.adk.tools import FunctionTool

from .._config import agent_config


def download_file(url: str, save_path: Optional[str] = None) -> str:
    """Download a file from a URL using requests.

    Useful for obtaining PDF/Excel files before reading them.

    Status: IMPLEMENTED
    Dependencies: requests
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return "[ERROR] requests not installed. Run: pip install requests"

    try:
        output_dir = agent_config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Determine save path
        if not save_path:
            # Try to guess from URL
            filename = url.split("/")[-1].split("?")[0]
            if not filename:
                filename = "downloaded_file"
            save_path = os.path.join(output_dir, filename)
        elif not os.path.isabs(save_path):
            save_path = os.path.join(output_dir, save_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, stream=True, timeout=30)  # type: ignore[import-untyped]
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size = os.path.getsize(save_path)
        return (
            f"[SUCCESS] Downloaded {url}\nSaved to: {save_path}\nSize: {size} bytes\n"
            "(You can now use file_reader, pdf_reader, or excel_reader on this path)"
        )

    except Exception as e:
        return f"[ERROR] Failed to download {url}: {str(e)}"


download_file_tool = FunctionTool(download_file)
