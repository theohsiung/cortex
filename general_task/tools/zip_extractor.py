"""ZIP extractor tool."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

from google.adk.tools import FunctionTool

from .._config import agent_config


def zip_extractor(file_path: str, extract_to: Optional[str] = None) -> str:
    """Extract ZIP archive contents.

    Returns list of extracted file paths.
    """
    if not os.path.exists(file_path):
        return f"[ERROR] File not found: {file_path}"

    try:
        zip_name = Path(file_path).stem
        output_dir = agent_config.output_dir

        # Determine extraction directory
        if extract_to:
            extract_dir = extract_to
        else:
            extract_dir = os.path.join(output_dir, f"extracted_{zip_name}")

        os.makedirs(extract_dir, exist_ok=True)

        extracted_paths = []
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

            # Get file list and assemble absolute paths
            for f in zip_ref.namelist():
                full_path = os.path.join(extract_dir, f)
                # Only record files, ignore pure directory entries
                if os.path.isfile(full_path):
                    extracted_paths.append(full_path)

        # Build output
        output = [f"[SUCCESS] Extracted '{os.path.basename(file_path)}'"]
        output.append(f"Directory: {extract_dir}")

        # Explicitly tell the model these are the paths to use
        output.append("\n=== Extracted File Paths (Use these for file_reader) ===")
        if not extracted_paths:
            output.append("(No files found in zip)")

        for p in extracted_paths:
            size = os.path.getsize(p)
            output.append(f"- {p} ({size} bytes)")

        return "\n".join(output)

    except zipfile.BadZipFile:
        return f"[ERROR] Invalid or corrupted ZIP file: {file_path}"
    except Exception as e:
        return f"[ERROR] Failed to extract ZIP: {str(e)}"


zip_extractor_tool = FunctionTool(zip_extractor)
