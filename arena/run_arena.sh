#!/bin/bash
# POKER AI HUMAN ARENA Lite Server Runner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PORT=8000
echo "=========================================================="
echo "Starting Poker AI Human Arena Lite on port $PORT"
echo "URL: http://localhost:$PORT"
echo "=========================================================="

python3 -m uvicorn arena.app:app --host 0.0.0.0 --port $PORT
