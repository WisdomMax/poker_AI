#!/usr/bin/env python3
"""
Waits for ChatGPT to complete the full audit and DIRECTIVE 047 generation.
Checks every 15s until output stops changing and has completed.
"""
import json
import os
import subprocess
import time

OUTPUT_RESP_PATH = "latest_gpt_response.txt"
VERIFIED_RESP_PATH = "v1024/latest_chatgpt_verified_reply.txt"

def wait_until_done():
    print("Waiting for ChatGPT to finish generating full review and directive...")
    
    poll_js = """
async function check() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || "").includes("6adca81a-6da4-83e8-baef-2b48268858b1"));
    if (!gptTab) {
        console.log(JSON.stringify({ status: "NO_TAB" }));
        return;
    }
    const p = await attachBrowserTab(gptTab.targetId);
    const data = await p.evaluate(() => {
        const bodyText = document.body.innerText;
        const marker = "DIRECTIVE 046 독립 심사 결과";
        let text = "";
        if (bodyText.includes(marker)) {
            text = bodyText.slice(bodyText.lastIndexOf(marker));
        } else {
            text = bodyText.slice(-3000);
        }
        
        const stopBtn = !!document.querySelector('button[aria-label="생성 중지"], button[data-testid="stop-button"]');
        const sendBtn = !!document.querySelector('button[aria-label="보내기"], button[data-testid="send-button"]');
        
        return {
            hasStopBtn: stopBtn,
            hasSendBtn: sendBtn,
            len: text.length,
            snippet: text.slice(-250).replace(/\\n/g, " ")
        };
    });
    console.log(JSON.stringify(data));
}
await check();
"""
    last_len = -1
    stable_count = 0
    
    for i in range(40):
        time.sleep(15)
        cmd = ["/Users/wisdom/.local/bin/aside", "repl", poll_js]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in res.stdout.strip().split("\n") if l.startswith("{") and l.endswith("}")]
        if lines:
            try:
                info = json.loads(lines[-1])
                curr_len = info.get("len", 0)
                has_stop = info.get("hasStopBtn", False)
                has_send = info.get("hasSendBtn", False)
                snippet = info.get("snippet", "")
                print(f"[{i+1}/40] Len: {curr_len} | Stop: {has_stop} | Send: {has_send} | End: {snippet[-60:]}")
                
                # Check if DIRECTIVE 047 or final directive has appeared
                if curr_len > 1500 and not has_stop:
                    if curr_len == last_len:
                        stable_count += 1
                        if stable_count >= 2:
                            print("\nChatGPT output finished and stabilized!")
                            break
                    else:
                        stable_count = 0
                last_len = curr_len
            except Exception as e:
                print(f"[{i+1}/40] Error: {e}")
        else:
            print(f"[{i+1}/40] Raw: {res.stdout.strip()[-80:]}")

    # Fetch final complete text
    fetch_js = """
async function fetch() {
    const tabs = await listBrowserTabs();
    const gptTab = tabs.find(t => (t.url || "").includes("6adca81a-6da4-83e8-baef-2b48268858b1"));
    const p = await attachBrowserTab(gptTab.targetId);
    const text = await p.evaluate(() => {
        const marker = "DIRECTIVE 046 독립 심사 결과";
        const bodyText = document.body.innerText;
        if (bodyText.includes(marker)) {
            return bodyText.slice(bodyText.lastIndexOf(marker));
        }
        return bodyText.slice(-4000);
    });
    console.log("===FINAL_REPLY_START===");
    console.log(text.trim());
    console.log("===FINAL_REPLY_END===");
}
await fetch();
"""
    res_fetch = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", fetch_js], capture_output=True, text=True)
    out = res_fetch.stdout
    if "===FINAL_REPLY_START===" in out and "===FINAL_REPLY_END===" in out:
        s = out.find("===FINAL_REPLY_START===") + len("===FINAL_REPLY_START===\n")
        e = out.find("===FINAL_REPLY_END===")
        final_text = out[s:e].strip()
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_text)
        with open(VERIFIED_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\nSaved full response ({len(final_text)} chars) to {OUTPUT_RESP_PATH}")
        print("\n================== FULL CHATGPT RESPONSE ==================")
        print(final_text)
        print("===========================================================")
    else:
        with open(OUTPUT_RESP_PATH, "w", encoding="utf-8") as f:
            f.write(out)
        print("Saved raw output to", OUTPUT_RESP_PATH)

if __name__ == "__main__":
    wait_until_done()
