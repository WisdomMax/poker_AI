"""
POKER AI HUMAN ARENA Lite: FastAPI Application with Secure External Access (DIRECTIVE 029)
==========================================================================================
Enhanced with:
- Token-based Session Authentication (Zero plaintext PIN leakage in URLs)
- Brute-force Protection (Rate limiting on PIN verification attempts)
- Environment-controlled confidential PIN (No hardcoded credentials)
- Independent multi-session management (Supports 1~3 concurrent human players)
- Strict WebSocket & API security (Zero model leak, zero card leak)
- Health & Resource Monitoring
"""

import os
import sys
import json
import time
import secrets
import hmac
import asyncio
from collections import defaultdict
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 파일에서 보안 설정 로드
def _load_env():
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

_load_env()

# 비밀 PIN 로드 (하드코딩 제거)
ACCESS_PIN = os.environ.get("ARENA_ACCESS_PIN")
if not ACCESS_PIN:
    # 환경변수 부재 시 보안 난수 생성 후 .env에 안전 저장
    generated = ''.join(secrets.choice('0123456789') for _ in range(6))
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_file, "a", encoding="utf-8") as f:
        f.write(f"\nARENA_ACCESS_PIN={generated}\n")
    os.chmod(env_file, 0o600)
    ACCESS_PIN = generated
    os.environ["ARENA_ACCESS_PIN"] = ACCESS_PIN

from arena.game_engine import PokerArenaSession

app = FastAPI(title="Poker AI Human Arena Lite (Secure External)", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 딕셔너리: session_id -> PokerArenaSession
sessions = {}

# Rate Limiting & Auth State
FAILED_ATTEMPTS = defaultdict(list)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 600  # 10분 잠금

VALID_TOKENS = {}  # token -> {"session_id": str, "created_at": float}
TOKEN_TTL_SECONDS = 86400  # 24시간

def is_ip_locked(ip: str) -> bool:
    now = time.time()
    recent = [t for t in FAILED_ATTEMPTS[ip] if now - t < LOCKOUT_SECONDS]
    FAILED_ATTEMPTS[ip] = recent
    return len(recent) >= MAX_FAILED_ATTEMPTS

def record_failed_attempt(ip: str):
    FAILED_ATTEMPTS[ip].append(time.time())

def reset_failed_attempts(ip: str):
    FAILED_ATTEMPTS.pop(ip, None)

def is_token_valid(token: str) -> bool:
    if not token or token not in VALID_TOKENS:
        return False
    created = VALID_TOKENS[token]["created_at"]
    if time.time() - created > TOKEN_TTL_SECONDS:
        VALID_TOKENS.pop(token, None)
        return False
    return True

class FeedbackModel(BaseModel):
    session_id: str
    token: str
    hand_number: int
    category: str
    comment: str

@app.post("/api/verify-pin")
async def verify_pin(request: Request, data: dict):
    client_ip = request.client.host if request.client else "unknown"
    if is_ip_locked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="인증 시도 횟수를 초과했습니다. 10분 후 다시 시도해 주세요."
        )
        
    input_pin = str(data.get("pin", "")).strip()
    session_id = str(data.get("session_id", "")).strip()
    
    # hmac.compare_digest로 타이밍 공격 방지 (비밀값 로깅 절대 금지)
    if input_pin and hmac.compare_digest(input_pin, ACCESS_PIN):
        reset_failed_attempts(client_ip)
        token = secrets.token_urlsafe(32)
        VALID_TOKENS[token] = {"session_id": session_id, "created_at": time.time()}
        return {"status": "ok", "valid": True, "token": token}
    else:
        record_failed_attempt(client_ip)
        recent_count = len(FAILED_ATTEMPTS[client_ip])
        remaining = max(0, MAX_FAILED_ATTEMPTS - recent_count)
        print(f"[Arena Auth] Failed verification attempt from {client_ip} (Remaining: {remaining})", flush=True)
        return {
            "status": "error",
            "valid": False,
            "message": f"입장 번호가 올바르지 않습니다. (남은 시도: {remaining}회)"
        }

@app.post("/api/feedback")
async def save_feedback(item: FeedbackModel):
    if not is_token_valid(item.token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="인증되지 않은 접근입니다.")
        
    log_dir = "arena/data"
    os.makedirs(log_dir, exist_ok=True)
    fpath = os.path.join(log_dir, "feedback.jsonl")
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": item.session_id,
        "hand_number": item.hand_number,
        "category": item.category,
        "comment": item.comment
    }
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok", "message": "Feedback saved successfully"}

@app.get("/api/health")
async def health_check():
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        "status": "online",
        "active_sessions": len(sessions),
        "cpu_usage_pct": cpu_pct,
        "memory_used_mb": round(mem.used / 1024 / 1024, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str,
    token: str = Query(default="")
):
    # 토큰 기반 인증 검증 (URL 평문 PIN 노출 원천 차단)
    if not is_token_valid(token):
        print(f"[Arena WS] Rejected unauthorized connection from session {session_id}", flush=True)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    if session_id not in sessions:
        sessions[session_id] = PokerArenaSession(session_id)
    session = sessions[session_id]
    
    # 초기 상태 전송
    await websocket.send_json(session.get_client_state())
    
    # 세션별 게임 진행 속도 (1.0 = 보통/정석 템포, 2.0 = 빠름)
    game_speed = 1.0

    async def sleep_paced(base_sec: float):
        actual_delay = max(0.2, base_sec / game_speed)
        await asyncio.sleep(actual_delay)

    async def process_game_flow():
        """인간 플레이어가 행동할 때까지 챈스 노드(공용 카드)와 AI 턴을 인간 템포로 순차 전개"""
        while session.is_hand_active and not session.current_state.is_terminal():
            # 1. 공용 카드 챈스 노드 전개 (플랍 3장 -> 턴 1장 -> 리버 1장)
            if session.current_state.is_chance_node():
                prev_board_cnt = len(session.board_cards)
                advanced, state = session.advance_one_chance_street()
                if advanced:
                    new_board_cnt = len(session.board_cards)
                    await websocket.send_json(state)
                    # 플랍(3장 공개)은 2.0초, 턴/리버(4~5장 공개)는 1.5초 감상 딜레이
                    if new_board_cnt == 3 and prev_board_cnt < 3:
                        await sleep_paced(2.0)
                    elif new_board_cnt > 3:
                        await sleep_paced(1.5)
                    continue

            cp = session.current_state.current_player()
            if cp == 0:  # 인간 차례: 무한 대기
                break

            # 2. AI 플레이어의 차례: 약 1.2초 생각하는 템포
            await sleep_paced(1.2)
            ai_acted, state = session.step_ai_single()
            await websocket.send_json(state)
            if not ai_acted:
                break
            # Let chips and the action caption settle before a board transition.
            await sleep_paced(0.9)

        # 3. 쇼다운 및 핸드 종료 처리
        if session.current_state and session.current_state.is_terminal():
            term_state = session.get_client_state()
            await websocket.send_json(term_state)
            await sleep_paced(3.0)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "set_speed":
                game_speed = 2.0 if data.get("speed") == 2.0 else 1.0
                print(f"[Arena WS] Game speed updated: {game_speed}x for session {session_id}")
                
            elif msg_type == "set_ai_mode":
                mode = data.get("mode", "human_friendly")
                session.set_ai_mode(mode)
                print(f"[Arena WS] AI mode updated: {mode} for session {session_id}")
                await websocket.send_json(session.get_client_state())
                
            elif msg_type == "start_hand":
                state = session.start_new_hand()
                await websocket.send_json(state)
                await process_game_flow()
                
            elif msg_type == "human_action":
                action_id = int(data.get("action_id", 1))
                state = session.step_human(action_id)
                await websocket.send_json(state)
                await sleep_paced(0.9)
                await process_game_flow()
                
            elif msg_type == "reset_stacks":
                session.reset_stacks()
                await websocket.send_json(session.get_client_state())
                
    except WebSocketDisconnect:
        print(f"[Arena WS] Client disconnected: {session_id}", flush=True)
        if session_id in sessions and not sessions[session_id].is_hand_active:
            # 유휴 세션 메모리 정리
            del sessions[session_id]
    except Exception as e:
        print(f"[Arena WS] Error in session {session_id}: {e}", flush=True)

# 정적 파일 마운트 (오직 arena/static 내의 HTML/CSS/JS만 서빙)
app.mount("/", StaticFiles(directory="arena/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arena.app:app", host="0.0.0.0", port=8000, reload=False)
