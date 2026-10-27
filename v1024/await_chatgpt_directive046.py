#!/usr/bin/env python3
"""
Awaits ChatGPT's response to DIRECTIVE 046 report.
Monitors the chat tab until deep reasoning finishes and final response is generated.
"""
import json
import os
import subprocess
import time

OUTPUT_RESP_PATH = "latest_gpt_response.txt"
VERIFIED_RESP_PATH = "v1024/latest_chatgpt_verified_reply.txt"

def await_response():
    print("Monitoring ChatGPT in Aside for DIRECTIVE 046 audit response...")
    
    poll_js = """
async function poll() {
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!gptTab) {
        console.log(JSON.stringify({ status: 'NO_TAB' }));
        return;
    }
    const p = await attachBrowserTab(gptTab.targetId);
    const data = await p.evaluate(() => {
        const bodyText = document.body.innerText;
        const marker = '## 8. 다음 단계 학습 확대 심사 요청';
        const markerIdx = bodyText.lastIndexOf(marker);
        if (markerIdx === -1) {
            return { status: 'NO_MARKER', len: bodyText.length };
        }
        const afterMarker = bodyText.slice(markerIdx + marker.length);
        const isReasoning = afterMarker.includes('⋮⋮') || afterMarker.includes('생각 중') || afterMarker.includes('추론 수준');
        const stopBtn = !!document.querySelector('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
        return {
            status: (isReasoning || stopBtn) ? 'GENERATING' : 'IDLE',
            isReasoning,
            hasStopBtn: stopBtn,
            afterMarkerLen: afterMarker.trim().length,
            snippet: afterMarker.slice(-200).replace(/\\n/g, ' ')
        };
    });
    console.log(JSON.stringify(data));
}
await poll();
"""
    last_len = -1
    stable_count = 0

    for attempt in range(60):
        res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", poll_js], capture_output=True, text=True)
        out = res.stdout.strip()
        lines = [l for l in out.split("\n") if l.startswith("{") and l.endswith("}")]
        if lines:
            try:
                info = json.loads(lines[-1])
                status = info.get("status")
                is_reasoning = info.get("isReasoning")
                after_len = info.get("afterMarkerLen", 0)
                snippet = info.get("snippet", "")
                print(f"[{attempt+1}/60] Status: {status} | Reasoning: {is_reasoning} | AfterMarkerLen: {after_len} | Snippet: {snippet[-60:]}")

                if after_len > 400 and not is_reasoning and not info.get("hasStopBtn"):
                    if after_len == last_len:
                        stable_count += 1
                        if stable_count >= 2:
                            print("\nChatGPT response has stabilized and finished generating!")
                            break
                    else:
                        stable_count = 0
                last_len = after_len
            except Exception as e:
                print(f"[{attempt+1}/60] Parse error: {e}")
        else:
            print(f"[{attempt+1}/60] Output: {out[-100:]}")

        time.sleep(10)

    # Fetch full reply after marker
    fetch_js = """
async function fetchReply() {
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    const p = await attachBrowserTab(gptTab.targetId);
    const reply = await p.evaluate(() => {
        const bodyText = document.body.innerText;
        const marker = '## 8. 다음 단계 학습 확대 심사 요청';
        const markerIdx = bodyText.lastIndexOf(marker);
        let res = '';
        if (markerIdx !== -1) {
            res = bodyText.slice(markerIdx + marker.length);
            const ansMarker = 'ChatGPT 답변:';
            if (res.includes(ansMarker)) {
                res = res.slice(res.indexOf(ansMarker) + ansMarker.length);
            }
        } else {
            res = bodyText;
        }
        return res.trim();
    });
    console.log('===REPLY_START===');
    console.log(reply);
    console.log('===REPLY_END===');
}
await fetchReply();
"""
    res_fetch = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", fetch_js], capture_output=True, text=True)
    fetch_out = res_fetch.stdout
    if "===REPLY_START===" in fetch_out and "===REPLY_END===" in fetch_out:
        start_idx = fetch_out.find("===REPLY_START===") + len("===REPLY_START===\n")
        end_idx = fetch_out.find("===REPLY_END===")
        final_reply = fetch_out[start_idx:end_idx].strip()
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_reply)
        with open(VERIFIED_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_reply)
        print(f"\nSuccessfully saved full ChatGPT reply ({len(final_reply)} chars) to {OUTPUT_RESP_PATH}!")
        print("\n=================== CHATGPT FULL RESPONSE ===================")
        print(final_reply)
        print("=============================================================")
    else:
        print("Failed to parse cleanly, saving raw output...")
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(fetch_out)

if __name__ == "__main__":
    await_response()
