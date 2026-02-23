"""Python executor tool for safe code execution."""

from __future__ import annotations

import io
import os
import sys

from google.adk.tools import FunctionTool

from .._config import agent_config


def python_executor(code: str) -> str:
    """Execute Python code with captured stdout.

    Injects pandas (pd), numpy (np), and Biopython (Bio) into the execution namespace.
    """
    exec_globals: dict[str, object] = {
        "__builtins__": __builtins__,
    }

    # Check Pandas
    libraries_status = []

    # Check Pandas
    try:
        import pandas as pd

        exec_globals["pd"] = pd
        exec_globals["pandas"] = pd
    except ImportError:
        libraries_status.append("pandas (MISSING)")

    # Check Numpy
    try:
        import numpy as np

        exec_globals["np"] = np
        exec_globals["numpy"] = np
    except ImportError:
        libraries_status.append("numpy (MISSING)")

    # Check Biopython
    try:
        import Bio

        exec_globals["Bio"] = Bio

        from Bio.PDB import PDBParser

        exec_globals["PDBParser"] = PDBParser
    except ImportError:
        libraries_status.append("biopython (MISSING)")

    # If core libraries are missing, return error
    if "pandas (MISSING)" in libraries_status or "numpy (MISSING)" in libraries_status:
        return (
            f"[SYSTEM ERROR] Essential libraries missing."
            f" Status: {', '.join(libraries_status)}."
            " Please ask admin to install them."
        )

    # Check Matplotlib
    plt: object = None
    try:
        import matplotlib

        matplotlib.use("Agg")  # Set to non-interactive mode
        import matplotlib.pyplot as _plt

        plt = _plt
        exec_globals["plt"] = plt
        exec_globals["matplotlib"] = matplotlib
    except ImportError:
        pass

    # Set output directory
    output_dir = agent_config.output_dir
    if output_dir:
        exec_globals["OUTPUT_DIR"] = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # --- Execution Phase ---

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    result_value = None
    error_msg = None
    saved_files: list[str] = []

    try:
        # Execute the code
        exec(code, exec_globals)  # noqa: S102

        # Check for common result variable names
        for var_name in ["result", "answer", "output", "value"]:
            if var_name in exec_globals:
                result_value = exec_globals[var_name]
                break

        # Auto-save matplotlib figures if any are open
        if plt is not None and output_dir:
            import matplotlib.pyplot as _mpl_plt

            fig_nums = _mpl_plt.get_fignums()
            for i, fig_num in enumerate(fig_nums):
                fig = _mpl_plt.figure(fig_num)
                fig_path = os.path.join(output_dir, f"plot_{i + 1}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                saved_files.append(fig_path)
            _mpl_plt.close("all")

    except SystemExit as e:
        # Model-generated code may call sys.exit() or raise SystemExit.
        # Since exec() runs in-process, we must catch this to prevent
        # killing the entire runner process.
        error_msg = (
            f"[ERROR] Code called SystemExit({e.code}). This is not allowed in sandboxed execution."
        )
    except Exception as e:
        # Capture specific error types, help model debug
        error_msg = f"[ERROR] Execution failed: {type(e).__name__}: {str(e)}"

    finally:
        sys.stdout = old_stdout

    # --- Build return message ---
    output_lines = []
    stdout_content = captured_output.getvalue()

    if stdout_content:
        output_lines.append("=== Output ===")
        output_lines.append(stdout_content.strip())

    if result_value is not None:
        output_lines.append("\n=== Result ===")
        output_lines.append(str(result_value))

    if saved_files:
        output_lines.append("\n=== Generated Files (Use these paths for next steps) ===")
        for fp in saved_files:
            output_lines.append(f"  {fp}")

    if error_msg:
        output_lines.append(error_msg)

    if not output_lines:
        output_lines.append("[INFO] Code executed successfully (no output)")

    return "\n".join(output_lines)


python_executor_tool = FunctionTool(python_executor)
