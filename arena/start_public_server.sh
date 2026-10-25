#!/bin/bash
# POKER AI HUMAN ARENA: One-Click Public Server Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting POKER AI HUMAN ARENA Public Server..."
echo "=========================================================="

# 1. 기존 프로세스 정리
pkill -f "arena.app:app" 2>/dev/null || true
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
sleep 1

# 2. FastAPI 서버 기동 (Port 8000)
python3 -m uvicorn arena.app:app --host 0.0.0.0 --port 8000 > arena/server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > arena/server.pid
echo "[+] FastAPI Server running (PID: $SERVER_PID, Port: 8000)"

# 3. Cloudflare Tunnel 기동
cloudflared tunnel --url http://localhost:8000 > arena/tunnel.log 2>&1 &
TUNNEL_PID=$!
echo $TUNNEL_PID > arena/tunnel.pid
echo "[+] Cloudflare Tunnel running (PID: $TUNNEL_PID)"

# 4. 외부 HTTPS URL 추출 (최대 15초 대기)
echo -n "[*] Waiting for public HTTPS URL from Cloudflare"
PUBLIC_URL=""
for i in {1..15}; do
    echo -n "."
    sleep 1
    PUBLIC_URL=$(grep -o 'https://[-a-zA-Z0-9.]*\.trycloudflare\.com' arena/tunnel.log | head -n 1)
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
done
echo ""

if [ -n "$PUBLIC_URL" ]; then
    echo "=========================================================="
    echo "🎉 POKER AI ARENA PUBLIC ACCESS ONLINE!"
    echo "=========================================================="
    echo "👉 접속 주소: $PUBLIC_URL"
    echo "👉 원클릭 접속(PIN 자동): $PUBLIC_URL/?pin=7777"
    echo "👉 기본 입장 번호 (PIN): 7777"
    echo "=========================================================="
else
    echo "[!] Warning: Public URL could not be retrieved within 15s. Check arena/tunnel.log"
fi
