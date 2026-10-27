#!/usr/bin/env python3
"""
v1024/chatgpt_holdem_bridge.py
Robust communication bridge with ChatGPT "홀덤 기계적 전략" chat room via Aside.
"""

import subprocess
import time
import json
import os

SESSION_ID = "6adca81a-6da4-83e8-baef-2b48268858b1"
TARGET_URL = f"https://chatgpt.com/c/{SESSION_ID}"

def aside_repl(js_code: str, timeout: int = 45) -> str:
    try:
        res = subprocess.run(
            ["/Users/wisdom/.local/bin/aside", "repl", js_code],
            capture_output=True, text=True, timeout=timeout
        )
        return res.stdout + res.stderr
    except Exception as e:
        return f"[ERROR: {e}]"

def send_message_to_holdem(text: str) -> bool:
    print(f"[Bridge] Sending {len(text)} chars to Holdem room...")
    subprocess.run(["pbcopy"], input=text, text=True)

    js = f"""
    async function sendMsg() {{
        let p;
        try {{
            // openTab brings the page directly to focus and avoids background CDP attach timeout
            p = await openTab('{TARGET_URL}');
            await sleep(3500);
        }} catch(e) {{
            console.log('Open tab error:', e.message);
            const tabs = await listBrowserTabs();
            const holdem = tabs.find(t => (t.url || '').includes('{SESSION_ID}'));
            if (holdem) p = await attachBrowserTab(holdem.targetId);
        }}
        
        if (!p) return false;
        
        console.log('Target page URL:', p.url());
        
        // Target input
        let inputLoc = null;
        for (let i = 0; i < 12; i++) {{
            const loc = p.locator('div.ProseMirror[contenteditable="true"], div#prompt-textarea, textarea[data-id="root"]');
            if (await loc.count() > 0) {{
                inputLoc = loc.last();
                break;
            }}
            await sleep(1000);
        }}
        
        if (!inputLoc) {{
            console.log('INPUT_NOT_FOUND');
            return false;
        }}
        
        await inputLoc.click();
        await sleep(600);
        
        // Paste from clipboard
        await p.keyboard.down('Meta');
        await p.keyboard.press('KeyV');
        await p.keyboard.up('Meta');
        console.log('Pasted text from clipboard');
        await sleep(2500);
        
        // Submit
        const sendBtn = p.locator('button[data-testid="send-button"], button[aria-label*="보내기"], button[aria-label*="전송"]').first();
        if (await sendBtn.count() > 0 && !(await sendBtn.isDisabled())) {{
            await sendBtn.click();
            console.log('Clicked send button directly');
        }} else {{
            console.log('Pressing Enter key to send');
            await p.keyboard.press('Enter');
        }}
        
        await sleep(3000);
        console.log('BRIDGE_SEND_SUCCESS');
        return true;
    }}
    await sendMsg();
    """
    out = aside_repl(js, timeout=45)
    print("[Bridge Send Output]:", out[:300].strip())
    return "BRIDGE_SEND_SUCCESS" in out

def wait_for_chatgpt_response(poll_interval: int = 10, max_wait: int = 300) -> str:
    print(f"[Bridge] Waiting for ChatGPT response (max {max_wait}s)...")
    time.sleep(15)
    t0 = time.time()
    
    poll_js = f"""
    async function poll() {{
        let p;
        try {{
            p = await openTab('{TARGET_URL}');
            await sleep(2000);
        }} catch(e) {{
            const tabs = await listBrowserTabs();
            const holdem = tabs.find(t => (t.url || '').includes('{SESSION_ID}'));
            if (holdem) p = await attachBrowserTab(holdem.targetId);
        }}
        if (!p) return;
        const stopBtn = await p.locator('button[aria-label="중지"], button[aria-label="생성 중지"]').count();
        const main = document.querySelector('main');
        const text = main ? main.innerText : document.body.innerText;
        console.log('STATUS:GEN=' + (stopBtn > 0 ? 'YES' : 'NO') + ':LEN=' + text.length);
        console.log('===CONTENT_START===');
        console.log(text.slice(-5000));
        console.log('===CONTENT_END===');
    }}
    await poll();
    """
    
    prev_len = 0
    stable_count = 0
    
    while time.time() - t0 < max_wait:
        out = aside_repl(poll_js, timeout=30)
        is_gen = "STATUS:GEN=YES" in out
        length = 0
        if ":LEN=" in out:
            try:
                length = int(out.split(":LEN=")[1].split()[0])
            except:
                pass
        
        print(f"[Bridge Poll] Generating={is_gen}, Length={length} (+{length - prev_len})")
        if not is_gen and length > 0 and length == prev_len:
            stable_count += 1
            if stable_count >= 2:
                print("[Bridge] Response generation complete and stable!")
                if "===CONTENT_START===" in out:
                    s = out.find("===CONTENT_START===") + len("===CONTENT_START===\n")
                    e = out.find("===CONTENT_END===")
                    return out[s:e].strip()
        else:
            stable_count = 0
            
        prev_len = length
        time.sleep(poll_interval)
        
    return ""

if __name__ == "__main__":
    print("Bridge module ready.")
