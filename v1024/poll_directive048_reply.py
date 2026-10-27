#!/usr/bin/env python3
"""
Polls ChatGPT response for DIRECTIVE 048 and captures the reply.
"""
import subprocess
import time
import sys

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
async function checkResponse() {{
    const p = await attachBrowserTab('{TAB_ID}');
    const stop = await p.locator('button[aria-label="중지"], button[aria-label="생성 중지"]').count();
    const text = await p.evaluate(() => document.body.innerText);
    console.log('GEN:' + (stop > 0 ? 'YES' : 'NO') + ' LEN:' + text.length);
}}
await checkResponse();
"""

print("Waiting for ChatGPT response for DIRECTIVE 048...")
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
    print(f"[{attempt+1}] GEN={is_gen} LEN={length} (+{length - prev_len})")
    prev_len = length

    if not is_gen and length > 0 and attempt >= 2:
        print("ChatGPT response generation finished!")
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

print("Fetching latest ChatGPT response...")
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
    # print tail
    print("\n--- Tail of response ---")
    print(full[-3000:])
else:
    print("Failed to parse reply delimiter.")
    print(out[:1000])
