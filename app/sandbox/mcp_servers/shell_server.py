"""
Shell MCP Server for sandboxed command execution.

This server runs inside the Docker container and provides
shell command execution within /workspace directory.

Usage:
    python -m app.sandbox.mcp_servers.shell_server
"""

import asyncio
import os
import shlex
from pathlib import Path

from fastmcp import FastMCP

# All operations are relative to /workspace
WORKSPACE_ROOT = Path("/workspace")

# Maximum output size (10MB)
MAX_OUTPUT_SIZE = 10 * 1024 * 1024

# Default timeout (5 minutes)
DEFAULT_TIMEOUT = 300

mcp = FastMCP("Shell Server")


async def _run_command_impl(
    command: str,
    working_dir: str = ".",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Execute a shell command.

    Args:
        command: The command to execute
        working_dir: Working directory (relative to workspace, default: workspace root)
        timeout: Command timeout in seconds (default: 300)

    Returns:
        Dictionary with exit_code, stdout, stderr
    """
    # Resolve working directory
    if working_dir.startswith("/"):
        cwd = WORKSPACE_ROOT / working_dir.lstrip("/")
    else:
        cwd = WORKSPACE_ROOT / working_dir

    cwd = cwd.resolve()

    # Security check: ensure cwd is within workspace
    try:
        cwd.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Working directory '{working_dir}' is outside workspace")

    if not cwd.exists():
        raise FileNotFoundError(f"Working directory not found: {working_dir}")

    if not cwd.is_dir():
        raise ValueError(f"Not a directory: {working_dir}")

    # Execute command
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env={**os.environ, "HOME": str(WORKSPACE_ROOT)},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "timed_out": True,
            }

        # Truncate output if too large
        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        if len(stdout_str) > MAX_OUTPUT_SIZE:
            stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
        if len(stderr_str) > MAX_OUTPUT_SIZE:
            stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"

        return {
            "exit_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
        }

    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
        }


@mcp.tool()
async def run_command(
    command: str,
    working_dir: str = ".",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Execute a shell command.

    Args:
        command: The command to execute
        working_dir: Working directory (relative to workspace, default: workspace root)
        timeout: Command timeout in seconds (default: 300)

    Returns:
        Dictionary with exit_code, stdout, stderr
    """
    return await _run_command_impl(command, working_dir, timeout)


@mcp.tool()
async def run_python(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Execute Python code.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 300)

    Returns:
        Dictionary with exit_code, stdout, stderr
    """
    # Write code to temp file and execute
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=str(WORKSPACE_ROOT),
        delete=False
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        result = await _run_command_impl(
            f"python {shlex.quote(script_path)}",
            timeout=timeout
        )
        return result
    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except OSError:
            pass


@mcp.tool()
def get_environment() -> dict:
    """
    Get current environment information.

    Returns:
        Dictionary with environment details
    """
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "workspace": str(WORKSPACE_ROOT),
        "cwd": str(WORKSPACE_ROOT),
    }


if __name__ == "__main__":
    mcp.run()
