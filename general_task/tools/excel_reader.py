"""Excel reader tool using pandas."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from google.adk.tools import FunctionTool


def excel_reader(file_path: str, sheet: Optional[str] = None, query: Optional[str] = None) -> str:
    """Read Excel/CSV files using pandas.

    Returns: columns and head(5) as markdown table.
    """
    try:
        import pandas as pd
    except ImportError:
        return "[ERROR] pandas not installed. Run: pip install pandas openpyxl"

    if not os.path.exists(file_path):
        return f"[ERROR] File not found: {file_path}"

    try:
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            if sheet:
                df = pd.read_excel(file_path, sheet_name=sheet)
            else:
                # Read all sheets if no specific sheet requested
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names

                if len(sheet_names) == 1:
                    df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                else:
                    # Multiple sheets - return info about all sheets
                    output = [f"Excel file contains {len(sheet_names)} sheets: {sheet_names}\n"]
                    for sn in sheet_names:
                        sheet_df = pd.read_excel(file_path, sheet_name=sn)
                        output.append(f"\n=== Sheet: {sn} ===")
                        output.append(
                            f"Columns ({len(sheet_df.columns)}): {list(sheet_df.columns)}"
                        )
                        output.append(f"Rows: {len(sheet_df)}")
                        output.append("\nFirst 5 rows:")
                        output.append(sheet_df.head(5).to_markdown(index=False))
                    return "\n".join(output)
        else:
            return f"[ERROR] Unsupported file format: {ext}"

        # Build output
        output = []
        output.append(f"File: {os.path.basename(file_path)}")
        output.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        output.append(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
        output.append("\nFirst 5 rows:")
        output.append(df.head(5).to_markdown(index=False))

        return "\n".join(output)

    except Exception as e:
        return f"[ERROR] Failed to read Excel file: {str(e)}"


excel_reader_tool = FunctionTool(excel_reader)
