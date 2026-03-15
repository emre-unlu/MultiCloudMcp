import logging
import sys
from pathlib import Path

from fastmcp import FastMCP  # pip install fastmcp

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from tools import register_all

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def _build_mcp() -> FastMCP:
    instance = FastMCP(
        name="kubernetes-mcp",
        instructions="Safe Kubernetes tools (deletions require human approval).",
    )
    register_all(instance)
    return instance


mcp = _build_mcp()

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8080,
        path="/",
        stateless_http=True,
    )
