"""
Filesystem MCP Server for sandboxed file operations.

This server runs inside the Docker container and provides
file operations within /workspace directory.

Usage:
    python -m app.sandbox.mcp_servers.filesystem_server
"""

import os
from pathlib import Path

from fastmcp import FastMCP

# All operations are relative to /workspace
WORKSPACE_ROOT = Path("/workspace")

mcp = FastMCP("Filesystem Server")


def _safe_path(path: str) -> Path:
    """Ensure path is within workspace and return absolute path."""
    # Resolve the path relative to workspace
    if path.startswith("/"):
        # Absolute path - make it relative to workspace
        full_path = WORKSPACE_ROOT / path.lstrip("/")
    else:
        full_path = WORKSPACE_ROOT / path

    # Resolve to get canonical path (handles ..)
    resolved = full_path.resolve()

    # Security check: ensure path is within workspace
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Path '{path}' is outside workspace")

    return resolved


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a file.

    Args:
        path: Path to the file (relative to workspace or absolute within workspace)

    Returns:
        The file contents as a string
    """
    file_path = _safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")

    return file_path.read_text()


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates the file if it doesn't exist.

    Args:
        path: Path to the file (relative to workspace or absolute within workspace)
        content: Content to write to the file

    Returns:
        Confirmation message
    """
    file_path = _safe_path(path)

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content)
    return f"Successfully wrote {len(content)} characters to {path}"


@mcp.tool()
def list_directory(path: str = ".") -> list[dict]:
    """
    List contents of a directory.

    Args:
        path: Path to the directory (default: workspace root)

    Returns:
        List of entries with name, type, and size
    """
    dir_path = _safe_path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    entries = []
    for entry in sorted(dir_path.iterdir()):
        entry_info = {
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            entry_info["size"] = entry.stat().st_size
        entries.append(entry_info)

    return entries


@mcp.tool()
def get_file_info(path: str) -> dict:
    """
    Get information about a file or directory.

    Args:
        path: Path to the file or directory

    Returns:
        Dictionary with file information
    """
    file_path = _safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    stat = file_path.stat()
    return {
        "path": str(file_path.relative_to(WORKSPACE_ROOT)),
        "type": "directory" if file_path.is_dir() else "file",
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
    }


@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a directory.

    Args:
        path: Path to the directory to create

    Returns:
        Confirmation message
    """
    dir_path = _safe_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return f"Successfully created directory: {path}"


@mcp.tool()
def delete_file(path: str) -> str:
    """
    Delete a file.

    Args:
        path: Path to the file to delete

    Returns:
        Confirmation message
    """
    file_path = _safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")

    file_path.unlink()
    return f"Successfully deleted: {path}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """
    Move or rename a file.

    Args:
        source: Source path
        destination: Destination path

    Returns:
        Confirmation message
    """
    src_path = _safe_path(source)
    dst_path = _safe_path(destination)

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Create parent directories if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src_path.rename(dst_path)
    return f"Successfully moved {source} to {destination}"


if __name__ == "__main__":
    mcp.run()
