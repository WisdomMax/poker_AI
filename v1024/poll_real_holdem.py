#!/usr/bin/env python3
"""
Polls ChatGPT in the REAL Hold'em Chat Room
"""
import subprocess
import time
import sys

TARGET_URL = "https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1"
TAB_ID = "B953A3E68E4EA5A5D719FA615BB330C5"
OUTPUT_RESP = "latest_gpt_response.txt"
VERIFIED_RESP = "v1024/latest_chatgpt_verified_reply.txt"

def aside_repl(js: str, timeout: int = 30) -> str:
    res = subprocess.run(
        ["/Users/wisdom/.local/bin/aside", "repl", js],
        capture_output=True, text=True, timeout=timeout
    )
    return res.stdout + res.stderr

poll_js = f"""
async function checkHoldemResponse() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const url = p.url();
    const title = await p.title();
    const stop = await p.locator('button[aria-label="중지"], button[aria-label="생성 중지"]').count();
    const text = await p.evaluate(() => document.body.innerText);
    console.log('ROOM:' + title + ' GEN:' + (stop > 0 ? 'YES' : 'NO') + ' LEN:' + text.length);
}}
await checkHoldemResponse();
"""

print("Waiting for ChatGPT response in REAL Hold'em chat room...")
prev_len = 0
for attempt in range(90): # up to 15 min
    out = aside_repl(poll_js)
    is_gen = "GEN:YES" in out
    length = 0
    if "LEN:" in out:
        try:
            length = int(out.split("LEN:")[1].split()[0])
        except:
            pass
    print(f"[{attempt+1}] {out.strip()} (+{length - prev_len})")
    prev_len = length

    if not is_gen and length > 0 and attempt >= 2:
        print("ChatGPT response generation finished in Hold'em room!")
        break
    time.sleep(10)

fetch_js = f"""
async function fetch() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const text = await p.evaluate(() => {{
        const main = document.querySelector('main');
        return main ? main.innerText : document.body.innerText;
    }});
    console.log('===REPLY_START===');
    console.log(text);
    console.log('===REPLY_END===');
}}
await fetch();
"""

print("Fetching latest ChatGPT response from Hold'em room...")
out = aside_repl(fetch_js, timeout=45)
if "===REPLY_START===" in out:
    s = out.find("===REPLY_START===") + len("===REPLY_START===\n")
    e = out.find("===REPLY_END===")
    full = out[s:e].strip()
    with open(OUTPUT_RESP, "w", encoding="utf-8") as f:
        f.write(full)
    with open(VERIFIED_RESP, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"Saved reply ({len(full)} chars) to {OUTPUT_RESP}")
    print("\n--- Tail of Hold'em room response ---")
    print(full[-3000:])
else:
    print("Failed to parse reply delimiter.")
    print(out[:1000])
