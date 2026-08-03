"""MCP server, the only external write path to the store. Unit U4."""

from autoevolve.mcp.server import build_server, serve_http, serve_stdio

__all__ = ["build_server", "serve_http", "serve_stdio"]
