#!/bin/bash
# Run the MCP server in dev/inspector mode from a local virtualenv.
# For normal use with Claude Code, point your MCP client at server.py directly
# (see README) — this script is only for interactive development.

cd "$(dirname "$0")"

if [ -d .venv ]; then
    source .venv/bin/activate
fi

mcp dev server.py
