#!/usr/bin/env python3
"""
Sends DIRECTIVE 043 Final Report to ChatGPT in Aside browser
and monitors for response completion.
"""
import json
import os
import subprocess
import time

REPORT_PATH = "v1024/directive043_final_report.md"
OUTPUT_RESP_PATH = "latest_gpt_response.txt"

def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    header = """# [공식 보고] DIRECTIVE 043: Universal Poker AlphaGo (5인 자가대국 파일럿) 완주 보고

ChatGPT님! DIRECTIVE 043에서 지시하신 **5인 동시 강화학습 자가대국(AlphaGo) 파일럿 및 PPO 정규화 무결성 회귀 검증**을 완료하여 GitHub에 모든 소스, 단위 테스트, 체크포인트, 원시 JSONL을 푸시했습니다.

- **GitHub 커밋 SHA**: `53c4507` (https://github.com/WisdomMax/poker_AI/commit/53c4507)
- **체크포인트 5종 포함**: `Agent_A_2000h.pt` ~ `Agent_E_2000h.pt` (SHA256 검증 완료)
- **원시 2,000핸드 JSONL**: `v1024/data/directive043_raw_selfplay_2000h.jsonl`

상세 실측 보고서는 다음과 같습니다:

---

"""
    return header + content

def send_and_wait():
    report_text = load_report()
    print(f"Report length: {len(report_text)} chars")

    js_code = f"""
async function run() {{
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!gptTab) {{
        console.log('Holdem tab not found, opening...');
        await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
        await sleep(5000);
        const tabs2 = await listBrowserTabs();
        gptTab = tabs2.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    }}
    if (!gptTab) {{
        console.error('FAILED_NO_TAB');
        return;
    }}
    console.log('Attaching to tab:', gptTab.targetId);
    const p = await attachBrowserTab(gptTab.targetId);
    await sleep(2000);

    let targetInput = null;
    for (let r = 0; r < 10; r++) {{
        const inputs = p.locator('div[contenteditable="true"]');
        if (await inputs.count() > 0) {{
            targetInput = inputs.last();
            break;
        }}
        await sleep(1000);
    }}
    if (!targetInput) {{
        console.error('FAILED_INPUT');
        return;
    }}

    await targetInput.focus();
    await targetInput.click();
    await sleep(800);

    const txt = {json.dumps(report_text)};
    console.log('Inserting text...');
    await p.keyboard.insertText(txt);
    await sleep(2000);

    const sendBtn = p.locator('button[data-testid="send-button"], button[aria-label="보내기"], button[aria-label="메시지 전송"]').last();
    if (await sendBtn.count() > 0 && !(await sendBtn.isDisabled())) {{
        console.log('Clicking send button...');
        await sendBtn.click();
    }} else {{
        console.log('Pressing Enter...');
        await p.keyboard.press('Enter');
    }}
    console.log('SENT_SUCCESS');
}}
await run();
"""
    cmd = ["/Users/wisdom/.local/bin/aside", "repl", js_code]
    print("Executing Aside REPL to send report...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if "SENT_SUCCESS" not in res.stdout:
        print("STDERR:", res.stderr)
        raise RuntimeError("Failed to send report to ChatGPT")

    print("\nReport sent successfully! Now waiting for ChatGPT to finish generating response...")
    
    # Poll for response completion using Aside repl
    poll_js = """
async function checkResponse() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!gptTab) {
        console.log('STATUS:NO_TAB');
        return;
    }
    const p = await attachBrowserTab(gptTab.targetId);
    
    // Check if generating: look for stop button or streaming indicator
    const stopBtn = p.locator('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
    const isGenerating = (await stopBtn.count() > 0);
    
    // Get full text
    const text = await p.evaluate(() => document.body.innerText);
    console.log('STATUS:' + (isGenerating ? 'GENERATING' : 'DONE') + ' LEN:' + text.length);
}
await checkResponse();
"""
    # Wait at least 15s before first check
    time.sleep(15)
    
    for attempt in range(60): # Up to 10 minutes (60 * 10s)
        res_poll = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", poll_js], capture_output=True, text=True)
        out = res_poll.stdout
        print(f"[{attempt+1}/60] Poll output: {out.strip()[-100:]}")
        if "STATUS:DONE" in out:
            print("ChatGPT response generation is complete!")
            break
        time.sleep(10)

    # Fetch final conversation and save
    fetch_js = """
async function fetchFull() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    const p = await attachBrowserTab(gptTab.targetId);
    const text = await p.evaluate(() => document.body.innerText);
    console.log('===FULL_PAGE_TEXT_START===');
    console.log(text);
    console.log('===FULL_PAGE_TEXT_END===');
}
await fetchFull();
"""
    res_fetch = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", fetch_js], capture_output=True, text=True)
    full_output = res_fetch.stdout
    if "===FULL_PAGE_TEXT_START===" in full_output and "===FULL_PAGE_TEXT_END===" in full_output:
        start_idx = full_output.find("===FULL_PAGE_TEXT_START===") + len("===FULL_PAGE_TEXT_START===\n")
        end_idx = full_output.find("===FULL_PAGE_TEXT_END===")
        page_text = full_output[start_idx:end_idx]
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(page_text)
        print(f"Saved full page text ({len(page_text)} chars) to {OUTPUT_RESP_PATH}")
    else:
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"Saved raw output to {OUTPUT_RESP_PATH}")

if __name__ == "__main__":
    send_and_wait()
