#!/bin/bash
# POKER AI HUMAN ARENA: Stop Server and Tunnel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Stopping POKER AI HUMAN ARENA services..."

if [ -f arena/server.pid ]; then
    SERVER_PID=$(cat arena/server.pid)
    kill $SERVER_PID 2>/dev/null || true
    rm -f arena/server.pid
fi

if [ -f arena/tunnel.pid ]; then
    TUNNEL_PID=$(cat arena/tunnel.pid)
    kill $TUNNEL_PID 2>/dev/null || true
    rm -f arena/tunnel.pid
fi

pkill -f "arena.app:app" 2>/dev/null || true
pkill -f "cloudflared tunnel --url" 2>/dev/null || true

echo "All services stopped."
