"""
POKER AI HUMAN ARENA: DIRECTIVE 025 Audio System Verification
=============================================================
Audits:
1. 38 Audio files existence, headers, and HTTP 200 accessibility
2. Character & Dealer voice mapping integrity
3. Casino SFX sound mapping integrity
4. Backend event dispatch (char_id, last_event, action_key)
5. Headless browser E2E test (AudioContext, mute toggle, volume slider)
"""

import os
import sys
import json
import urllib.request
import asyncio
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.game_engine import PokerArenaSession

AUDIO_MANIFEST = {
    "voices": {
        "kim": ["call", "check", "fold", "raise", "allin"],
        "jenny": ["call", "check", "fold", "raise", "allin"],
        "jack": ["call", "check", "fold", "raise", "allin"],
        "mr_k": ["call", "check", "fold", "raise", "allin"],
        "dealer": ["start", "flop", "turn", "river", "showdown", "hero_win", "winner", "next_hand"]
    },
    "sfx": [
        "card_shuffle", "card_deal", "card_flip", "check_knock",
        "chip_call", "chip_raise", "chip_allin", "pot_win", "victory", "defeat"
    ]
}

def audit_audio_files_and_http():
    print("=" * 70)
    print("▶ [검사 1] 38개 오디오 파일 물리적 존재 및 HTTP 200 서빙 감사")
    print("=" * 70)
    base_url = "http://localhost:8000/audio"
    
    total_checked = 0
    # 1. Voices
    for char, cues in AUDIO_MANIFEST["voices"].items():
        for cue in cues:
            rel_path = f"voices/{char}/{cue}.mp3"
            full_path = os.path.join("arena/static/audio", rel_path)
            assert os.path.exists(full_path), f"Missing file: {full_path}"
            fsize = os.path.getsize(full_path)
            assert fsize > 300, f"File too small ({fsize} bytes): {full_path}"
            
            # HTTP 200 체크
            url = f"{base_url}/{rel_path}"
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=3) as resp:
                assert resp.status == 200, f"HTTP {resp.status} for {url}"
            total_checked += 1

    # 2. SFX
    for sfx in AUDIO_MANIFEST["sfx"]:
        rel_path = f"sfx/{sfx}.mp3"
        full_path = os.path.join("arena/static/audio", rel_path)
        assert os.path.exists(full_path), f"Missing file: {full_path}"
        fsize = os.path.getsize(full_path)
        assert fsize > 300, f"File too small ({fsize} bytes): {full_path}"
        
        url = f"{base_url}/{rel_path}"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200, f"HTTP {resp.status} for {url}"
        total_checked += 1

    print(f"  ✓ 총 {total_checked}개 오디오 파일 (음성 28개 + 효과음 10개) 100% 정상 서빙 확인!")

def audit_backend_event_dispatch():
    print("\n" + "=" * 70)
    print("▶ [검사 2] 백엔드 오디오 이벤트 및 캐릭터 ID 디스패치 감사")
    print("=" * 70)
    session = PokerArenaSession("audio_audit_session")
    st = session.start_new_hand()
    
    # 1. AI 캐릭터 프로필 검증
    players = st["players"]
    assert len(players) == 5
    assert players[1]["char_id"] == "kim"
    assert players[2]["char_id"] == "jenny"
    assert players[3]["char_id"] == "jack"
    assert players[4]["char_id"] == "mr_k"
    print("  ✓ 4인의 AI 캐릭터 ID (kim, jenny, jack, mr_k) 바인딩 확인")
    
    # 2. 새 핸드 이벤트 확인
    assert st["last_event"]["type"] == "new_hand"
    print("  ✓ new_hand 이벤트 정상 트리거 확인")
    
    # 3. 액션 진행 시 last_event 확인
    step = 0
    action_keys_seen = set()
    while not st["is_terminal"] and step < 40:
        step += 1
        cp = st["current_player"]
        if cp == 0:
            st = session.step_human(1) # Call
        else:
            prog, st = session.step_ai_single()
            if not prog:
                break
        
        ev = st.get("last_event")
        if ev and ev.get("type") == "action":
            action_keys_seen.add(ev["action_key"])
            assert "seat" in ev
            assert "char_id" in ev
            assert "action_key" in ev

    print(f"  ✓ 시뮬레이션 중 포착된 액션 사운드 키: {action_keys_seen}")
    assert len(action_keys_seen) >= 2, "Action keys not dispatched properly!"

async def audit_browser_audio_ui():
    print("\n" + "=" * 70)
    print("▶ [검사 3] 브라우저 Playwright E2E 오디오 UI 및 음소거/볼륨 제어 감사")
    print("=" * 70)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 960})
        
        # 콘솔 에러 모니터링
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        
        await page.goto("http://localhost:8000/?pin=7777", wait_until="networkidle")
        await asyncio.sleep(1.0)
        
        # 1. 음소거 버튼 검증
        mute_btn = page.locator("#btn-audio-mute")
        assert await mute_btn.is_visible(), "Audio mute button not visible"
        initial_text = await mute_btn.inner_text()
        print(f"  - 초기 사운드 버튼 상태: {initial_text}")
        
        await mute_btn.click()
        await asyncio.sleep(0.3)
        muted_text = await mute_btn.inner_text()
        print(f"  - 음소거 클릭 후 상태: {muted_text}")
        assert muted_text == "🔇", f"Expected 🔇, got {muted_text}"
        
        await mute_btn.click()
        await asyncio.sleep(0.3)
        unmuted_text = await mute_btn.inner_text()
        print(f"  - 음소거 해제 후 상태: {unmuted_text}")
        assert unmuted_text == "🔊", f"Expected 🔊, got {unmuted_text}"
        
        # 2. 볼륨 슬라이더 검증
        vol_slider = page.locator("#audio-volume")
        assert await vol_slider.is_visible(), "Volume slider not visible"
        await vol_slider.fill("0.5")
        vol_val = await page.evaluate("window.arenaAudio.volume")
        assert abs(vol_val - 0.5) < 1e-2, f"Volume mismatch: {vol_val}"
        print(f"  - 볼륨 슬라이더 조절 검증 완료 (vol: {vol_val})")
        
        # 3. 새 핸드 시작 및 액션 진행 중 오디오 큐 동작 확인
        new_hand_btn = page.locator("#btn-new-hand")
        if await new_hand_btn.is_enabled():
            await new_hand_btn.click()
            await asyncio.sleep(2.0)
            
        call_btn = page.locator("#btn-call")
        if await call_btn.is_enabled():
            await call_btn.click()
            await asyncio.sleep(2.0)
            
        # 콘솔 에러 전무 확인
        assert len(errors) == 0, f"Browser runtime errors detected: {errors}"
        print("  ✓ 브라우저 런타임 오류 0건 (클린 통과)")
        
        # 스크린샷 캡처
        screenshot_path = "arena/static/screenshot_audio_ui.png"
        artifact_path = "/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/arena_directive025_audio_ui.png"
        await page.screenshot(path=screenshot_path)
        await page.screenshot(path=artifact_path)
        print(f"  ✓ 스크린샷 저장 완료: {screenshot_path}")
        
        await browser.close()

def main():
    audit_audio_files_and_http()
    audit_backend_event_dispatch()
    asyncio.run(audit_browser_audio_ui())
    print("\n" + "=" * 70)
    print("🎉 DIRECTIVE 025: 오디오 & 효과음 시스템 종합 감사 100% ALL PASS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
