"""
POKER AI HUMAN ARENA Lite: FastAPI Application
==============================================
Ultra-lightweight web server for human vs 4-AI poker play:
- WebSocket endpoint /ws/{session_id} for real-time play
- Static file serving for web UI
- Research feedback & statistics APIs
"""

import os
import sys
import json
import time
import asyncio
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.game_engine import PokerArenaSession

app = FastAPI(title="Poker AI Human Arena Lite", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 딕셔너리: session_id -> PokerArenaSession
sessions = {}

class FeedbackModel(BaseModel):
    session_id: str
    hand_number: int
    category: str
    comment: str

@app.post("/api/feedback")
async def save_feedback(item: FeedbackModel):
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
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        sessions[session_id] = PokerArenaSession(session_id)
    session = sessions[session_id]
    
    # 초기 상태 전송
    await websocket.send_json(session.get_client_state())
    
    async def process_ai_turns():
        """
        AI 턴이 끝날 때까지 0.35초 간격으로 AI 의사결정을 자동 전진하고 브로드캐스트
        """
        while session.is_hand_active and not session.current_state.is_terminal():
            cp = session.current_state.current_player()
            if cp == 0:  # Human turn
                break
            await asyncio.sleep(0.35)  # 사람의 시각적 인지를 위한 자연스러운 인터벌
            ai_acted, state = session.step_ai_single()
            await websocket.send_json(state)
            if not ai_acted:
                break
        # 혹시 터미널 도달했으면 최종 상태 전송
        if session.current_state and session.current_state.is_terminal():
            await websocket.send_json(session.get_client_state())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "start_hand":
                state = session.start_new_hand()
                await websocket.send_json(state)
                # 만약 첫 턴이 AI라면 바로 AI 턴들 처리
                await process_ai_turns()
                
            elif msg_type == "human_action":
                action_id = int(data.get("action_id", 1))
                state = session.step_human(action_id)
                await websocket.send_json(state)
                # 인간 액션 이후 AI 턴들 자동 전진
                await process_ai_turns()
                
            elif msg_type == "reset_stacks":
                session.reset_stacks()
                await websocket.send_json(session.get_client_state())
                
    except WebSocketDisconnect:
        print(f"[Arena WS] Client disconnected: {session_id}", flush=True)
    except Exception as e:
        print(f"[Arena WS] Error in session {session_id}: {e}", flush=True)

# 정적 파일 마운트
app.mount("/", StaticFiles(directory="arena/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arena.app:app", host="0.0.0.0", port=8000, reload=False)
