#!/bin/bash
# Cloudflare Tunnel Runner for Poker AI Human Arena

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Starting Cloudflare Quick Tunnel to http://localhost:8000..."
cloudflared tunnel --url http://localhost:8000 > arena/tunnel.log 2>&1
