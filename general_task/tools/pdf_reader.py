"""PDF reader tool using pypdf."""

from __future__ import annotations

import os
from typing import Optional

from google.adk.tools import FunctionTool


def pdf_reader(file_path: str, page: Optional[int] = None) -> str:
    """Read PDF files using pypdf.

    Extracts text page by page.
    """
    if not os.path.exists(file_path):
        return f"[ERROR] File not found: {file_path}"

    try:
        from pypdf import PdfReader
    except ImportError:
        return "[ERROR] pypdf not installed. Run: pip install pypdf"

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        output = [f"PDF: {os.path.basename(file_path)}"]
        output.append(f"Total pages: {total_pages}\n")

        for page_idx, pdf_page in enumerate(reader.pages, 1):
            # Skip if specific page requested and this isn't it
            if page is not None and page_idx != page:
                continue

            output.append(f"[Page {page_idx}]")
            text = pdf_page.extract_text() or "(No text content)"
            output.append(text.strip())
            output.append("")  # Blank line between pages

        return "\n".join(output)

    except Exception as e:
        return f"[ERROR] Failed to read PDF: {str(e)}"


pdf_reader_tool = FunctionTool(pdf_reader)
