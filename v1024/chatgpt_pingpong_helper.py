import os
import sys
import json
import time
import subprocess

TARGET_SESSION_ID = "6adca81a-6da4-83e8-baef-2b48268858b1"
TARGET_URL = f"https://chatgpt.com/c/{TARGET_SESSION_ID}"

def send_to_chatgpt(message_text):
    js_script = f"""
async function run() {{
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('{TARGET_SESSION_ID}'));
    let p;
    if (gptTab) {{
        console.log('Attaching to existing tab:', gptTab.targetId);
        p = await attachBrowserTab(gptTab.targetId);
    }} else {{
        console.log('Opening new tab for target session...');
        p = await openTab('{TARGET_URL}');
        await sleep(5000);
    }}
    
    // 입력창 탐색
    let targetInput = null;
    for (let retry = 0; retry < 15; retry++) {{
        const inputs = p.locator('div[contenteditable="true"]');
        const count = await inputs.count();
        if (count > 0) {{
            targetInput = inputs.last();
            break;
        }}
        await sleep(1500);
    }}
    
    if (!targetInput) {{
        console.error('Failed to locate editable input');
        return false;
    }}
    
    await targetInput.focus();
    await targetInput.click();
    await sleep(800);
    
    const textToSend = {json.dumps(message_text)};
    await p.keyboard.insertText(textToSend);
    await sleep(2000);
    
    // 엔터 전송 시도
    await p.keyboard.press('Enter');
    await sleep(1500);
    
    // 전송 버튼 클릭 시도
    const sendBtn = p.locator('button[data-testid="send-button"], button[aria-label="보내기"], button[aria-label="메시지 전송"]').last();
    if (await sendBtn.count() > 0 && !(await sendBtn.isDisabled())) {{
        await sendBtn.click();
    }}
    console.log('CHATGPT_MESSAGE_SENT_SUCCESS');
    return true;
}}
await run();
"""
    try:
        res = subprocess.run(
            ['/Users/wisdom/.local/bin/aside', 'repl', js_script],
            capture_output=True, text=True, timeout=60
        )
        return "CHATGPT_MESSAGE_SENT_SUCCESS" in res.stdout
    except Exception as e:
        print(f"Error sending to ChatGPT: {e}")
        return False

def wait_for_gpt_reply(min_wait_sec=25, max_wait_sec=120):
    print(f"Waiting for ChatGPT reply (min {min_wait_sec}s)...", flush=True)
    time.sleep(min_wait_sec)
    
    start_t = time.time()
    while time.time() - start_t < (max_wait_sec - min_wait_sec):
        res = subprocess.run(["python3", "extract_latest_response.py"], capture_output=True, text=True)
        if "Successfully extracted" in res.stdout:
            try:
                with open("latest_gpt_response.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                if "ChatGPT 답변:" in content or "최신 응답" in content:
                    return content
            except:
                pass
        time.sleep(10)
    return ""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = sys.argv[1]
        ok = send_to_chatgpt(msg)
        print("Send result:", ok)
    else:
        print("Usage: python3 chatgpt_pingpong_helper.py <message>")
