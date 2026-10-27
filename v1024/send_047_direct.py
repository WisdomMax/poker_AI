#!/usr/bin/env python3
"""
DIRECTIVE 047 보고서를 ChatGPT에 직접 전송 (page.type 방식)
"""
import json
import subprocess
import time
import os

# 전송할 텍스트 준비
with open("v1024/directive047_final_report.md", "r", encoding="utf-8") as f:
    report_content = f.read()

HEADER = """# [공식 제출] DIRECTIVE 047: 5인 자가대국 최종 평가기 수정 + 5,000핸드 확장 완주 보고

ChatGPT님! 지시해주신 DIRECTIVE 047의 두 가지 필수 결함을 완전 수정하고 5,000핸드 자가대국 확장까지 완주했습니다.

수정 내역 요약:
1. 캐처 무료 폴드 버그 100% 제거: to_call==0일 때 80% 폴드 코드 삭제, 항상 Check. 1,000핸드 16,000회 무료 체크 기회 -> 무료 폴드 0건 실증
2. 올인 콜 오분류 원천 수정: len(folded_seats)<4 추정 폐기 -> HandLedgerTracker 이벤트 기반 원장 교체. False Positive 0건 (기존 1,640/2,500건 오분류 제거)
3. 5,000핸드 자가대국 완주: 2,001h -> 7,000h, 5인 전원 PPO 업데이트, 25개 신규 체크포인트 보존
4. 3파트 종합 평가: 캐처 대전 / 페어드 벤치마크 / 1,000핸드 리그 원시 JSONL 전수 기록

GitHub 최신 커밋: 800066d (https://github.com/WisdomMax/poker_AI/commit/800066d)
TDD 5대 회귀 테스트: v1024/tests/test_directive047_evaluator_ledger.py (5/5 PASS, 11.8초)

아래는 DIRECTIVE 047 공식 실증 보고서 전문입니다:

---

"""

FULL_TEXT = HEADER + report_content
TAB_ID = "B953A3E68E4EA5A5D719FA615BB330C5"
CHAT_URL = "https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1"

def aside_repl(js: str, timeout: int = 60) -> str:
    res = subprocess.run(
        ["/Users/wisdom/.local/bin/aside", "repl", js],
        capture_output=True, text=True, timeout=timeout
    )
    return res.stdout + res.stderr

def step1_navigate():
    """채팅방으로 이동 및 페이지 상태 확인"""
    js = f"""
async function nav() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const url = p.url();
    console.log('Current URL:', url);
    if (!url.includes('6adca81a')) {{
        await p.goto('{CHAT_URL}');
        await sleep(4000);
    }}
    // 입력창 확인
    const inp = p.locator('#prompt-textarea');
    const cnt = await inp.count();
    console.log('Prompt textarea count:', cnt);
    // ProseMirror 확인
    const pm = p.locator('.ProseMirror');
    const pmc = await pm.count();
    console.log('ProseMirror count:', pmc);
}}
await nav();
"""
    out = aside_repl(js)
    print("STEP1:", out)
    return out

def step2_type_via_clipboard():
    """클립보드를 통해 텍스트 붙여넣기"""
    # 먼저 pbcopy로 텍스트를 클립보드에 복사
    proc = subprocess.run(
        ["pbcopy"],
        input=FULL_TEXT,
        text=True,
        capture_output=True
    )
    print(f"Clipboard copy result: {proc.returncode}")
    
    js = f"""
async function pasteReport() {{
    const p = await attachBrowserTab('{TAB_ID}');
    
    // 입력창 클릭 (여러 셀렉터 시도)
    let el = null;
    const selectors = ['#prompt-textarea', '.ProseMirror', 'div[contenteditable="true"]'];
    for (const sel of selectors) {{
        const loc = p.locator(sel);
        const cnt = await loc.count();
        if (cnt > 0) {{
            el = loc.last();
            console.log('Found input with:', sel, 'count:', cnt);
            break;
        }}
    }}
    
    if (!el) {{
        console.error('NO_INPUT_FOUND');
        return;
    }}
    
    await el.click();
    await sleep(500);
    
    // Mac 클립보드 붙여넣기: Cmd+V
    await p.keyboard.press('Meta+a');
    await sleep(200);
    await p.keyboard.press('Backspace');
    await sleep(200);
    await p.keyboard.press('Meta+v');
    await sleep(2000);
    
    // 전송 버튼 찾기
    const btnSelectors = [
        'button[data-testid="send-button"]',
        'button[aria-label="보내기"]',
        'button[aria-label="Send message"]'
    ];
    let sendBtn = null;
    for (const sel of btnSelectors) {{
        const btn = p.locator(sel);
        const cnt = await btn.count();
        if (cnt > 0) {{
            sendBtn = btn.last();
            console.log('Found send button:', sel);
            break;
        }}
    }}
    
    if (sendBtn) {{
        // 버튼이 활성화 상태인지 확인
        const isDisabled = await sendBtn.evaluate(el => el.disabled);
        console.log('Send button disabled:', isDisabled);
        if (!isDisabled) {{
            await sendBtn.click();
            console.log('SENT_SUCCESS');
        }} else {{
            console.log('BUTTON_DISABLED');
        }}
    }} else {{
        console.error('NO_SEND_BUTTON');
    }}
}}
await pasteReport();
"""
    out = aside_repl(js, timeout=60)
    print("STEP2:", out)
    return out

def step3_wait_response():
    """응답 완료 대기 (최대 10분)"""
    poll_js = f"""
async function checkResponse() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const stopBtn = p.locator('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
    const isGenerating = (await stopBtn.count() > 0);
    const text = await p.evaluate(() => document.body.innerText);
    console.log('STATUS:' + (isGenerating ? 'GENERATING' : 'DONE') + ' LEN:' + text.length);
}}
await checkResponse();
"""
    for attempt in range(60):
        out = aside_repl(poll_js, timeout=30)
        print(f"[Poll {attempt+1}/60] {out.strip()[-100:]}")
        if "STATUS:DONE" in out:
            print("ChatGPT 응답 완료!")
            return True
        if "STATUS:GENERATING" in out:
            print(f"  → 생성 중, 10초 대기...")
        time.sleep(10)
    return False

def step4_fetch_reply():
    """최신 응답 수집"""
    fetch_js = f"""
async function fetchReply() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const text = await p.evaluate(() => {{
        const main = document.querySelector('main');
        return main ? main.innerText : document.body.innerText;
    }});
    console.log('===REPLY_START===');
    console.log(text);
    console.log('===REPLY_END===');
}}
await fetchReply();
"""
    out = aside_repl(fetch_js, timeout=30)
    if "===REPLY_START===" in out and "===REPLY_END===" in out:
        s = out.find("===REPLY_START===") + len("===REPLY_START===\n")
        e = out.find("===REPLY_END===")
        reply = out[s:e].strip()
        with open("latest_gpt_response.txt", "w", encoding="utf-8") as f:
            f.write(reply)
        with open("v1024/latest_chatgpt_verified_reply.txt", "w", encoding="utf-8") as f:
            f.write(reply)
        print(f"\n응답 수집 완료 ({len(reply)}자)")
        print("\n--- 최신 응답 미리보기 (마지막 1500자) ---")
        print(reply[-1500:])
        return reply
    else:
        print("응답 추출 실패, 원시 출력 저장")
        with open("latest_gpt_response.txt", "w", encoding="utf-8") as f:
            f.write(out)
        return out

if __name__ == "__main__":
    print("=== STEP 1: 채팅방 이동 확인 ===")
    step1_navigate()
    
    print("\n=== STEP 2: 보고서 전송 (클립보드 붙여넣기) ===")
    print(f"텍스트 길이: {len(FULL_TEXT)}자")
    result = step2_type_via_clipboard()
    
    if "SENT_SUCCESS" in result:
        print("\n✅ 전송 성공! 응답 대기 중...")
        print("\n=== STEP 3: 응답 완료 대기 ===")
        step3_wait_response()
        
        print("\n=== STEP 4: 응답 수집 ===")
        step4_fetch_reply()
    else:
        print("\n⚠️ 전송 실패. 응답 수집만 실행...")
        step4_fetch_reply()
