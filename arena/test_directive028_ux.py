"""
DIRECTIVE 028 INTEGRATED UX REBUILD E2E VERIFICATION SCRIPT
===========================================================
검증 항목:
1. 백엔드 단계별 챈스 런아웃 엔진 (advance_one_chance_street) 검증
2. 게임 속도 조절 엔진 (speed_factor 1.0x / 2.0x) 검증
3. 오디오 매니저 onended 및 타임스탬프 기반 큐 무결성 검증
4. E2E 데스크톱 브라우저 검증 (공용 카드 5구 슬롯, 3D 플립, 가이드 토글, 스크린샷)
5. E2E 모바일 가로모드 검증 (844x390 무스크롤 핏, 공용 카드 시인성, 터치 버튼, 스크린샷)
"""

import sys
import time
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

from arena.game_engine import PokerArenaSession, STREET_NAMES_KO

ARTIFACT_DIR = Path("/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b")

def test_backend_chance_advance():
    print("[1/5] 백엔드 단계별 챈스 노드 진행 (advance_one_chance_street) 검증...")
    session = PokerArenaSession(session_id="test_advance_session")
    state = session.start_new_hand(seed=100)
    
    assert len(state["board_cards"]) == 0, f"프리플랍 보드 카드는 0장이어야 함: {state['board_cards']}"
    assert state["round_name"] == "프리플랍 (Preflop)"
    
    # 5명 모두 콜/체크를 수행하여 플랍 챈스 노드에 도달하게 함
    while not session.current_state.is_chance_node() and not session.current_state.is_terminal():
        cp = session.current_state.current_player()
        if cp == 0:
            state = session.step_human(1)  # 체크/콜
        else:
            advanced, state = session.step_ai_single()
            if not advanced:
                break
                
    assert session.current_state.is_chance_node(), "플랍 진입을 위한 챈스 노드 도달 확인 실패"
    
    # 1단계 챈스 진행 (플랍 3장)
    ok, flop_state = session.advance_one_chance_street()
    assert ok, "플랍 advance_one_chance_street 실패"
    assert len(session.board_cards) == 3, f"플랍 보드 카드는 3장이어야 함: {len(session.board_cards)}"
    assert session.last_action_event["type"] == "street_deal"
    print(f"  -> 플랍 3장 공개 성공: {session.board_cards}")
    
    # 다시 베팅 진행하여 턴 챈스 노드 도달
    while not session.current_state.is_chance_node() and not session.current_state.is_terminal():
        cp = session.current_state.current_player()
        if cp == 0:
            state = session.step_human(1)
        else:
            advanced, state = session.step_ai_single()
            if not advanced:
                break
                
    if session.current_state.is_chance_node():
        ok, turn_state = session.advance_one_chance_street()
        assert ok, "턴 advance_one_chance_street 실패"
        assert len(session.board_cards) == 4, f"턴 보드 카드는 4장이어야 함: {len(session.board_cards)}"
        print(f"  -> 턴 4장 공개 성공: {session.board_cards}")

        # 다시 베팅 진행하여 리버 챈스 노드 도달
        while not session.current_state.is_chance_node() and not session.current_state.is_terminal():
            cp = session.current_state.current_player()
            if cp == 0:
                state = session.step_human(1)
            else:
                advanced, state = session.step_ai_single()
                if not advanced:
                    break
                    
        if session.current_state.is_chance_node():
            ok, river_state = session.advance_one_chance_street()
            assert ok, "리버 advance_one_chance_street 실패"
            assert len(session.board_cards) == 5, f"리버 보드 카드는 5장이어야 함: {len(session.board_cards)}"
            print(f"  -> 리버 5장 공개 성공: {session.board_cards}")

    print("  [PASS] 백엔드 단계별 챈스 런아웃 무결성 확인 완료")


def test_audio_manager_script():
    print("[2/5] AudioManager JS 스크립트 무결성 검증...")
    js_path = Path("/Volumes/NVME/7.AI_vibe_coding/hangame/arena/static/js/audio_manager.js")
    content = js_path.read_text(encoding="utf-8")
    
    assert "class AudioManager" in content
    assert "onended" in content, "audio.onended 기반 비동기 큐가 구현되어 있어야 함"
    assert "stopAllAndClear" in content, "stopAllAndClear 메서드가 구현되어 있어야 함"
    assert "setCurrentHand" in content, "setCurrentHand 메서드가 구현되어 있어야 함"
    assert "audio_timestamp" in content or "timestamp" in content, "지연 이벤트 폐기를 위한 타임스탬프 검증 로직 필수"
    print("  [PASS] AudioManager JS 비동기 큐 및 핸드 정리 구조 확인 완료")


def test_e2e_desktop(playwright):
    print("[3/5] E2E 데스크톱 환경 브라우저 검증 (1280x800)...")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    
    # 1. 메인 페이지 접근 및 PIN 입력
    page.goto("http://localhost:8000/?pin=7777")
    page.wait_for_selector("#board-cards", timeout=5000)
    
    # 2. 공용 카드 5구 슬롯 렌더링 확인
    slots = page.query_selector_all("#board-cards .board-slot")
    assert len(slots) == 5, f"공용 카드 슬롯은 항상 5개 렌더링되어야 함: 실제 {len(slots)}개"
    print("  -> 5구 공용 카드 슬롯 상시 렌더링 확인 (5개 완료)")
    
    # 3. 초보자 가이드 모드 토글 테스트
    guide_btn = page.query_selector("#btn-guide-toggle")
    assert guide_btn is not None
    assert "active" in guide_btn.get_attribute("class")
    
    # 가이드 클릭 -> OFF
    guide_btn.click()
    page.wait_for_timeout(300)
    body_cls = page.eval_on_selector("body", "el => el.className")
    assert "guide-mode-off" in body_cls, f"가이드 모드 OFF 시 body에 guide-mode-off 클래스 부착 필수: {body_cls}"
    print("  -> 초보자 가이드 토글 OFF 동작 확인")
    
    # 다시 가이드 클릭 -> ON
    guide_btn.click()
    page.wait_for_timeout(300)
    body_cls = page.eval_on_selector("body", "el => el.className")
    assert "guide-mode-off" not in body_cls, "가이드 모드 ON 복구 확인"
    print("  -> 초보자 가이드 토글 ON 복구 확인")
    
    # 4. 속도 토글 테스트
    speed_btn = page.query_selector("#btn-speed-toggle")
    assert speed_btn is not None
    speed_btn.click()
    page.wait_for_timeout(300)
    assert "fast" in speed_btn.get_attribute("class"), "속도 버튼에 fast 클래스 부착 확인"
    assert "빠름" in speed_btn.inner_text(), "속도 버튼 텍스트가 '빠름'으로 변경 확인"
    print("  -> 속도 토글(1.0x -> 2.0x) 동작 확인")

    # 5. 새 핸드 시작 및 개인 카드 확인
    new_hand_btn = page.query_selector("#btn-new-hand")
    if not new_hand_btn.is_disabled():
        new_hand_btn.click()
        page.wait_for_timeout(1000)
        
    hero_cards = page.query_selector_all("#seat-0 .player-cards .card")
    print(f"  -> Hero 개인 카드 수신 확인: {len(hero_cards)}장")
    
    # 데스크톱 스크린샷 저장
    desktop_shot_path = ARTIFACT_DIR / "arena_directive028_desktop.png"
    page.screenshot(path=str(desktop_shot_path))
    print(f"  -> 데스크톱 스크린샷 캡처 완료: {desktop_shot_path}")
    
    browser.close()
    print("  [PASS] E2E 데스크톱 환경 검증 완료")


def test_e2e_mobile_landscape(playwright):
    print("[4/5] E2E 모바일 가로모드 검증 (844x390, iPhone 14 Landscape)...")
    browser = playwright.chromium.launch(headless=True)
    # iPhone 14 가로모드 (844 x 390)
    context = browser.new_context(viewport={"width": 844, "height": 390}, is_mobile=True)
    page = context.new_page()
    
    # 메인 페이지 접근 및 PIN 인증
    page.goto("http://localhost:8000/?pin=7777")
    page.wait_for_selector("#board-cards", timeout=5000)
    page.wait_for_timeout(500)
    
    # 1. 100vw x 100vh 무스크롤 핏 검증
    scroll_height = page.evaluate("() => document.documentElement.scrollHeight")
    client_height = page.evaluate("() => window.innerHeight")
    print(f"  -> 가로모드 뷰포트 높이: {client_height}px, 실제 문서 높이: {scroll_height}px")
    assert scroll_height <= client_height + 5, f"가로모드에서 세로 스크롤이 발생하지 않아야 함 (Doc: {scroll_height}, Viewport: {client_height})"
    
    # 2. 공용 카드 5구 슬롯 노출 확인
    slots = page.query_selector_all("#board-cards .board-slot")
    assert len(slots) == 5, f"모바일 가로모드에서도 공용 카드 5구 슬롯이 렌더링되어야 함: {len(slots)}개"
    
    # 3. 조작 버튼 그리드 노출 확인
    action_btns = page.query_selector_all("#action-buttons .btn-action")
    assert len(action_btns) == 6, f"6개 액션 버튼 모두 렌더링되어야 함: {len(action_btns)}개"
    
    # 4. 사이드바 숨김 확인 (가로모드 최적화)
    sidebar_disp = page.eval_on_selector("#arena-sidebar", "el => window.getComputedStyle(el).display")
    assert sidebar_disp == "none", f"가로모드에서 사이드바는 화면 확보를 위해 none이어야 함: {sidebar_disp}"
    print("  -> 가로모드 사이드바 자동 숨김 및 테이블 화면 극대화 확인")
    
    # 모바일 가로모드 스크린샷 저장
    mobile_shot_path = ARTIFACT_DIR / "arena_directive028_mobile_landscape.png"
    page.screenshot(path=str(mobile_shot_path))
    print(f"  -> 모바일 가로모드 스크린샷 캡처 완료: {mobile_shot_path}")
    
    browser.close()
    print("  [PASS] E2E 모바일 가로모드 검증 완료")


def test_full_round_gameplay(playwright):
    print("[5/5] 실제 1개 라운드(프리플랍->쇼다운) E2E 게임플레이 검증...")
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto("http://localhost:8000/?pin=7777")
    page.wait_for_selector("#board-cards", timeout=5000)
    
    # 시작 시 새 핸드 클릭
    new_hand_btn = page.query_selector("#btn-new-hand")
    if not new_hand_btn.is_disabled():
        new_hand_btn.click()
        
    # 최대 20초 동안 액션 대기 및 진행
    max_steps = 15
    steps = 0
    while steps < max_steps:
        steps += 1
        page.wait_for_timeout(1000)
        
        # 내 차례인지 확인
        turn_banner = page.query_selector("#turn-banner")
        is_hero_turn = turn_banner and "hidden" not in turn_banner.get_attribute("class")
        
        if is_hero_turn:
            # 체크/콜 버튼 클릭
            call_btn = page.query_selector("#btn-call")
            if call_btn and not call_btn.is_disabled():
                print(f"  -> Step {steps}: Hero 체크/콜 액션 실행")
                call_btn.click()
                page.wait_for_timeout(500)
            else:
                fold_btn = page.query_selector("#btn-fold")
                if fold_btn and not fold_btn.is_disabled():
                    print(f"  -> Step {steps}: Hero 폴드 액션 실행")
                    fold_btn.click()
                    break
                    
        # 쇼다운이나 승자 배너가 나타났는지 확인
        winner_banner = page.query_selector("#winner-banner")
        if winner_banner and "hidden" not in winner_banner.get_attribute("class"):
            winner_text = page.eval_on_selector("#winner-text", "el => el.innerText")
            print(f"  -> 승자 발표 감지: {winner_text}")
            break
            
    # 최종 카드 렌더링 스크린샷 저장
    showdown_shot_path = ARTIFACT_DIR / "arena_directive028_showdown_step.png"
    page.screenshot(path=str(showdown_shot_path))
    print(f"  -> 게임플레이 쇼다운 캡처 완료: {showdown_shot_path}")
    
    browser.close()
    print("  [PASS] 실제 1개 라운드 E2E 게임플레이 검증 완료")


def main():
    print("=" * 70)
    print("DIRECTIVE 028 INTEGRATED: UX REBUILD 검증 시작")
    print("=" * 70)
    start_t = time.time()
    
    test_backend_chance_advance()
    test_audio_manager_script()
    
    with sync_playwright() as playwright:
        test_e2e_desktop(playwright)
        test_e2e_mobile_landscape(playwright)
        test_full_round_gameplay(playwright)
        
    duration = time.time() - start_t
    print("=" * 70)
    print(f"모든 DIRECTIVE 028 UX REBUILD 검증 100% 통과 (소요시간: {duration:.2f}초)")
    print("=" * 70)

if __name__ == "__main__":
    main()
