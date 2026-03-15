"""MCP tool for executing shell commands in a debugging environment."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastmcp import FastMCP

_DEFAULT_TIMEOUT = 30


def _build_environment() -> dict[str, str]:
    """Return the environment for diagnostic shell execution."""
    return dict(os.environ)


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def exec_shell(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        """Execute a shell command and return its output (stdout + stderr)."""
        if not command or not command.strip():
            raise ValueError("command is required")
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_build_environment(),
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stderr:
            return f"{stdout}\n[stderr]\n{stderr}".strip()
        return stdout.strip()