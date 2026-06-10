#!/bin/bash
# Activate virtual environment and run MCP server
# Using this allows direct access of Claude code

cd "$(dirname "$0")"
source .venv/bin/activate
mcp dev server.py
