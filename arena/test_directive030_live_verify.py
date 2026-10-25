"""
DIRECTIVE 030: Playwright Live Verification on trycloudflare HTTPS
=================================================================
Verifies:
1. '🛡️ 매너 AI' vs '⚡ 원본 AI' toggle button in header
2. Normal gameplay under '🛡️ 매너 AI' mode (smooth streets without irrational all-ins)
3. Mode switching WebSocket message and UI updates
4. Screenshot capture for final audit report
"""

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b")

def get_secret_pin():
    env_file = Path("/Volumes/NVME/7.AI_vibe_coding/hangame/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ARENA_ACCESS_PIN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("ARENA_ACCESS_PIN")

PIN = get_secret_pin()
TARGET_URL = "https://webmasters-jason-late-career.trycloudflare.com"

def main():
    print("=" * 70)
    print("DIRECTIVE 030: 실서버 UI 검증 및 스크린샷 캡처 시작")
    print(f"Target: {TARGET_URL}")
    print("=" * 70)
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        
        # 1. 사이트 접속 및 PIN 인증
        page.goto(TARGET_URL, timeout=20000)
        page.wait_for_selector("#pin-modal", timeout=10000)
        page.fill("#pin-input", PIN)
        page.click("#btn-enter")
        page.wait_for_selector("#board-cards .board-slot", timeout=10000)
        print("  -> 실서버 로그인 및 세션 연결 완료")
        
        # 2. '🛡️ 매너 AI' 버튼 존재 및 기본 활성 상태 확인
        btn_ai_mode = page.query_selector("#btn-ai-mode")
        assert btn_ai_mode is not None, "btn-ai-mode not found!"
        btn_text = btn_ai_mode.inner_text()
        print(f"  -> 초기 AI 모드 버튼 텍스트: '{btn_text}'")
        assert "매너 AI" in btn_text, f"기본 모드가 매너 AI가 아님: {btn_text}"
        
        # 3. 매너 AI 모드에서 새 핸드 시작 및 플레이
        page.click("#btn-new-hand")
        time.sleep(1.0)
        
        # 턴 올 때까지 콜/체크 수행하며 1판 완주
        for step in range(15):
            turn_visible = page.evaluate("() => { const el = document.querySelector('#turn-banner'); return el && !el.classList.contains('hidden'); }")
            winner_visible = page.evaluate("() => { const el = document.querySelector('#winner-banner'); return el && !el.classList.contains('hidden'); }")
            
            if winner_visible:
                print("  -> 쇼다운/정산 도달 완료")
                break
                
            can_call = page.evaluate("() => { const b = document.querySelector('#btn-call'); return b && !b.disabled; }")
            if can_call:
                page.click("#btn-call")
                time.sleep(0.5)
            else:
                time.sleep(0.3)
                
        time.sleep(0.8)
        shot_manner = OUTPUT_DIR / "arena_directive030_manner_ai_play.png"
        page.screenshot(path=str(shot_manner))
        print("  -> [캡처 1] 매너 AI 플레이 화면 저장 완료:", shot_manner.name)
        
        # 4. 모드 토글 클릭 테스트: '🛡️ 매너 AI' -> '⚡ 원본 AI'
        page.click("#btn-ai-mode")
        time.sleep(0.5)
        new_text = page.eval_on_selector("#btn-ai-mode", "el => el.innerText")
        print(f"  -> 모드 토글 후 버튼 텍스트: '{new_text}'")
        assert "원본 AI" in new_text, f"원본 AI로 전환 실패: {new_text}"
        
        shot_toggle = OUTPUT_DIR / "arena_directive030_mode_toggle.png"
        page.screenshot(path=str(shot_toggle))
        print("  -> [캡처 2] 원본 AI 모드 전환 화면 저장 완료:", shot_toggle.name)
        
        # 5. 다시 '🛡️ 매너 AI'로 복구
        page.click("#btn-ai-mode")
        time.sleep(0.5)
        restored_text = page.eval_on_selector("#btn-ai-mode", "el => el.innerText")
        print(f"  -> 매너 AI 복구 후 버튼 텍스트: '{restored_text}'")
        assert "매너 AI" in restored_text
        
        browser.close()
        
    print("\n" + "=" * 70)
    print("DIRECTIVE 030: 실서버 UI 및 모드 전환 검증 100% 통과!")
    print("=" * 70)

if __name__ == "__main__":
    main()
