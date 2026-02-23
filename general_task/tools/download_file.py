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

        # Detect file format from Content-Type if save_path has no extension
        _, existing_ext = os.path.splitext(save_path)
        if not existing_ext:
            content_type = response.headers.get("Content-Type", "").lower().split(";")[0].strip()
            ext_map = {
                "text/csv": ".csv",
                "application/csv": ".csv",
                "application/json": ".json",
                "text/json": ".json",
                "application/vnd.ms-excel": ".xls",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "application/pdf": ".pdf",
                "text/plain": ".txt",
                "text/html": ".html",
            }
            detected_ext = ext_map.get(content_type, "")
            if detected_ext:
                save_path = save_path + detected_ext

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size = os.path.getsize(save_path)
        _, final_ext = os.path.splitext(save_path)
        if final_ext in (".csv", ".tsv"):
            read_hint = "Use python_executor with pandas.read_csv() to process this file."
        elif final_ext in (".xls", ".xlsx"):
            read_hint = (
                "Use excel_reader or python_executor with pandas.read_excel() to process this file."
            )
        elif final_ext == ".pdf":
            read_hint = "Use pdf_reader to process this file."
        elif final_ext == ".json":
            read_hint = (
                "Use python_executor with json.load() or pandas.read_json() to process this file."
            )
        else:
            read_hint = "Use file_reader, pdf_reader, or excel_reader on this path."

        return f"[SUCCESS] Downloaded {url}\nSaved to: {save_path}\nSize: {size} bytes\n{read_hint}"

    except Exception as e:
        return f"[ERROR] Failed to download {url}: {str(e)}"


download_file_tool = FunctionTool(download_file)
