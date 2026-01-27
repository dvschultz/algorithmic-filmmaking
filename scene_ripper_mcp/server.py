"""MCP Server entry point using FastMCP."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (critical for stdio transport)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default timeout for long-running operations (5 minutes)
DEFAULT_TOOL_TIMEOUT = 300


def get_tool_timeout() -> int:
    """Get tool timeout from environment variable.

    Returns:
        Timeout in seconds (default: 300)
    """
    try:
        return int(os.environ.get("MCP_TOOL_TIMEOUT", DEFAULT_TOOL_TIMEOUT))
    except ValueError:
        logger.warning("Invalid MCP_TOOL_TIMEOUT value, using default")
        return DEFAULT_TOOL_TIMEOUT


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on server startup."""
    logger.info("Scene Ripper MCP Server starting...")

    # Load settings at startup
    from core.settings import load_settings

    settings = load_settings()
    timeout = get_tool_timeout()

    logger.info(f"Settings loaded: download_dir={settings.download_dir}")
    logger.info(f"Tool timeout: {timeout}s (set MCP_TOOL_TIMEOUT to customize)")

    yield {"settings": settings, "tool_timeout": timeout}

    logger.info("Scene Ripper MCP Server shutting down...")


# Create the MCP server instance
mcp = FastMCP(
    name="scene-ripper",
    instructions="Scene Ripper MCP Server - Video scene detection, analysis, and editing tools",
    lifespan=lifespan,
)

# Import tool registrations (these register tools on the mcp instance)
# pylint: disable=wrong-import-position
from scene_ripper_mcp.tools import project  # noqa: F401, E402
from scene_ripper_mcp.tools import youtube  # noqa: F401, E402
from scene_ripper_mcp.tools import analyze  # noqa: F401, E402
from scene_ripper_mcp.tools import clips  # noqa: F401, E402
from scene_ripper_mcp.tools import sequence  # noqa: F401, E402
from scene_ripper_mcp.tools import export  # noqa: F401, E402


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Scene Ripper MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for HTTP transport (default: 8765)",
    )
    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport={args.transport}")

    if args.transport == "http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
