"""Calculator tool using python_executor."""

from __future__ import annotations

from google.adk.tools import FunctionTool

from .python_executor import python_executor


def calculator(expression: str) -> str:
    """Calculator that evaluates a Python math expression.

    Wraps expression in print() statement for evaluation.
    """
    expression = expression.strip()
    code = f"result = {expression}\nprint(f'Result: {expression} = {{result}}')"
    return python_executor(code)


calculator_tool = FunctionTool(calculator)
