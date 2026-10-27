#!/usr/bin/env python3
"""
Monitors ChatGPT's live response to DIRECTIVE 046 report,
waits until generation finishes, and saves the verified response.
"""
import json
import os
import subprocess
import time

OUTPUT_RESP_PATH = "latest_gpt_response.txt"
VERIFIED_RESP_PATH = "v1024/latest_chatgpt_verified_reply.txt"

def collect_response():
    print("Collecting ChatGPT response to DIRECTIVE 046...")
    
    poll_js = """
async function poll() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || "").includes("6adca81a-6da4-83e8-baef-2b48268858b1"));
    if (!gptTab) {
        console.log(JSON.stringify({ status: "NO_TAB" }));
        return;
    }
    const p = await attachBrowserTab(gptTab.targetId);
    const data = await p.evaluate(() => {
        const bodyText = document.body.innerText;
        // Search for marker
        const marker = "차마트 보고서는 제외하고, 홀덤 DIRECTIVE 046만 심사하겠습니다.";
        let replyText = "";
        let found = false;
        if (bodyText.includes(marker)) {
            found = true;
            replyText = bodyText.slice(bodyText.lastIndexOf(marker));
        } else {
            const lastAnsMarker = "ChatGPT 답변:";
            if (bodyText.includes(lastAnsMarker)) {
                found = true;
                replyText = bodyText.slice(bodyText.lastIndexOf(lastAnsMarker) + lastAnsMarker.length);
            }
        }
        
        const stopBtn = !!document.querySelector('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
        const sendBtn = !!document.querySelector('button[aria-label="보내기"], button[data-testid="send-button"]');
        
        return {
            found,
            hasStopBtn: stopBtn,
            hasSendBtn: sendBtn,
            replyLen: replyText.trim().length,
            snippet: replyText.slice(-200).replace(/\\n/g, " ")
        };
    });
    console.log(JSON.stringify(data));
}
await poll();
"""
    last_len = -1
    stable_count = 0

    for attempt in range(60):
        cmd = ["/Users/wisdom/.local/bin/aside", "repl", poll_js]
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = res.stdout.strip()
        lines = [l for l in out.split("\n") if l.startswith("{") and l.endswith("}")]
        if lines:
            try:
                info = json.loads(lines[-1])
                reply_len = info.get("replyLen", 0)
                has_stop = info.get("hasStopBtn", False)
                has_send = info.get("hasSendBtn", False)
                snippet = info.get("snippet", "")
                print(f"[{attempt+1}/60] ReplyLen: {reply_len} | StopBtn: {has_stop} | SendBtn: {has_send} | Snippet: {snippet[-60:]}")
                
                # Check for completion
                if reply_len > 300 and not has_stop:
                    if reply_len == last_len:
                        stable_count += 1
                        if stable_count >= 2:
                            print("\nChatGPT response generation complete and verified stable!")
                            break
                    else:
                        stable_count = 0
                last_len = reply_len
            except Exception as e:
                print(f"[{attempt+1}/60] Parse error: {e}")
        else:
            print(f"[{attempt+1}/60] Raw output: {out[-100:]}")
            
        time.sleep(10)

    # Fetch final complete response
    fetch_js = """
async function fetchFinal() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || "").includes("6adca81a-6da4-83e8-baef-2b48268858b1"));
    const p = await attachBrowserTab(gptTab.targetId);
    const reply = await p.evaluate(() => {
        const bodyText = document.body.innerText;
        const marker = "차마트 보고서는 제외하고, 홀덤 DIRECTIVE 046만 심사하겠습니다.";
        if (bodyText.includes(marker)) {
            return bodyText.slice(bodyText.lastIndexOf(marker)).trim();
        }
        const lastAns = "ChatGPT 답변:";
        if (bodyText.includes(lastAns)) {
            return bodyText.slice(bodyText.lastIndexOf(lastAns) + lastAns.length).trim();
        }
        return bodyText.slice(-3000).trim();
    });
    console.log("===REPLY_START===");
    console.log(reply);
    console.log("===REPLY_END===");
}
await fetchFinal();
"""
    res_fetch = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", fetch_js], capture_output=True, text=True)
    out_fetch = res_fetch.stdout
    if "===REPLY_START===" in out_fetch and "===REPLY_END===" in out_fetch:
        start_idx = out_fetch.find("===REPLY_START===") + len("===REPLY_START===\n")
        end_idx = out_fetch.find("===REPLY_END===")
        final_text = out_fetch[start_idx:end_idx].strip()
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_text)
        with open(VERIFIED_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\nSaved final ChatGPT response ({len(final_text)} chars) to {OUTPUT_RESP_PATH}!")
        print("\n=================== CHATGPT AUDIT & DIRECTIVE 047 ===================")
        print(final_text)
        print("=====================================================================")
    else:
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(out_fetch)
        print("Saved raw output to", OUTPUT_RESP_PATH)

if __name__ == "__main__":
    collect_response()
