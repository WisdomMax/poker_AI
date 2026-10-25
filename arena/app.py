"""
POKER AI HUMAN ARENA Lite: FastAPI Application with Secure External Access (DIRECTIVE 022)
==========================================================================================
Enhanced with:
- PIN Access Gatekeeper (Blocks unauthorized access, default PIN: 7777)
- Independent multi-session management (Supports 1~3 concurrent human players)
- Strict WebSocket & API security (Zero model leak, zero card leak)
- Health & Resource Monitoring
"""

import os
import sys
import json
import time
import asyncio
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.game_engine import PokerArenaSession

app = FastAPI(title="Poker AI Human Arena Lite (Secure External)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 보안 PIN 설정 (환경 변수 또는 기본값 7777)
ACCESS_PIN = os.environ.get("ARENA_ACCESS_PIN", "7777")

# 세션 딕셔너리: session_id -> PokerArenaSession
sessions = {}

class FeedbackModel(BaseModel):
    session_id: str
    pin: str
    hand_number: int
    category: str
    comment: str

@app.post("/api/verify-pin")
async def verify_pin(data: dict):
    input_pin = str(data.get("pin", "")).strip()
    if input_pin == ACCESS_PIN:
        return {"status": "ok", "valid": True}
    return {"status": "error", "valid": False, "message": "입장 번호가 일치하지 않습니다."}

@app.post("/api/feedback")
async def save_feedback(item: FeedbackModel):
    if item.pin != ACCESS_PIN:
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
    pin: str = Query(default="")
):
    # PIN 인증 검증
    if pin != ACCESS_PIN:
        print(f"[Arena WS] Rejected connection from {session_id} - Invalid PIN: '{pin}'", flush=True)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    if session_id not in sessions:
        sessions[session_id] = PokerArenaSession(session_id)
    session = sessions[session_id]
    
    # 초기 상태 전송
    await websocket.send_json(session.get_client_state())
    
    async def process_ai_turns():
        while session.is_hand_active and not session.current_state.is_terminal():
            cp = session.current_state.current_player()
            if cp == 0:  # Human turn
                break
            await asyncio.sleep(0.35)
            ai_acted, state = session.step_ai_single()
            await websocket.send_json(state)
            if not ai_acted:
                break
        if session.current_state and session.current_state.is_terminal():
            await websocket.send_json(session.get_client_state())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "start_hand":
                state = session.start_new_hand()
                await websocket.send_json(state)
                await process_ai_turns()
                
            elif msg_type == "human_action":
                action_id = int(data.get("action_id", 1))
                state = session.step_human(action_id)
                await websocket.send_json(state)
                await process_ai_turns()
                
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
