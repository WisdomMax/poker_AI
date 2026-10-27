#!/usr/bin/env python3
"""
Sends DIRECTIVE 044 Audit Report to ChatGPT in Aside browser
and monitors for response completion.
"""
import json
import os
import subprocess
import time

REPORT_PATH = "v1024/directive044_final_report.md"
OUTPUT_RESP_PATH = "latest_gpt_response.txt"

def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    header = """# [공식 제출] DIRECTIVE 044: 5인 자가대국 독립 감사 결과 보고

ChatGPT님! DIRECTIVE 044에서 명령하신 **"체크포인트 전수 제출, 올인 계측 복원, 초기 노이즈 vs 실제 학습의 분리, 5인 전원 및 단계별 진화 검증"** 감사를 모두 완료하여 GitHub에 푸시했습니다.

- **GitHub 커밋 SHA**: `2b974c1` (최신 감사 코드 및 원시 JSONL)
  * 마일스톤 체크포인트 20개 전수 푸시: `8e27b74` (25개 체크포인트 GitHub 카탈로그 완성)
- **원시 JSONL**: `v1024/data/directive044_raw_audit_hands.jsonl` (352 KB)
- **정밀 감사 JSON**: `v1024/directive044_audit_results.json`

아래는 ChatGPT 4대 기준(코드 구현 확인 / 원시 데이터로 검증 / 통계적으로 확인 / 아직 미입증)에 따라 가감 없이 작성된 실측 보고서 전문입니다:

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

    // If retry button is present, click it first
    const retryBtns = await p.locator('button:has-text("다시 시도")').all();
    if (retryBtns.length > 0) {{
        console.log('Clicking retry button before sending...');
        await retryBtns[0].click();
        await sleep(4000);
    }}

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
    console.log('Inserting report text...');
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
    
    poll_js = """
async function checkResponse() {
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!gptTab) {
        console.log('STATUS:NO_TAB');
        return;
    }
    const p = await attachBrowserTab(gptTab.targetId);
    const stopBtn = p.locator('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
    const isGenerating = (await stopBtn.count() > 0);
    const text = await p.evaluate(() => document.body.innerText);
    console.log('STATUS:' + (isGenerating ? 'GENERATING' : 'DONE') + ' LEN:' + text.length);
}
await checkResponse();
"""
    time.sleep(15)
    
    for attempt in range(60):
        res_poll = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", poll_js], capture_output=True, text=True)
        out = res_poll.stdout
        print(f"[{attempt+1}/60] Poll: {out.strip()[-80:]}")
        if "STATUS:DONE" in out:
            print("ChatGPT response generation is complete!")
            break
        time.sleep(10)

    # Fetch final text
    fetch_js = """
async function fetchFull() {
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
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
