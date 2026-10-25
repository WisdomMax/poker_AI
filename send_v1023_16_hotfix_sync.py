import json
import subprocess
import time

TARGET_URL = "https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9"
REPORT_PATH = "/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/v1023_16_hotfix_final_report.md"

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report_content = f.read()

aside_script = f"""
async function run() {{
    console.log('1. 브라우저 탭 확인 또는 열기...');
    const tabs = await listBrowserTabs();
    let targetTab = tabs.find(t => (t.url || '').includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
    let p;
    if (targetTab) {{
        console.log('기존 탭 연결:', targetTab.targetId);
        p = await attachBrowserTab(targetTab.targetId);
    }} else {{
        console.log('새 탭 열기:', '{TARGET_URL}');
        targetTab = await openTab('{TARGET_URL}');
        await sleep(5000);
        p = await attachBrowserTab(targetTab.targetId);
    }}
    
    console.log('2. 입력창 대기...');
    await p.waitForSelector('div.ProseMirror', {{ timeout: 20000 }});
    await p.evaluate(() => {{
        const el = document.querySelector('div.ProseMirror');
        if (el) {{
            el.focus();
            el.click();
        }}
    }});
    await sleep(1000);
    
    console.log('3. 텍스트 주입...');
    const textToSend = {json.dumps(report_content)};
    await p.keyboard.insertText(textToSend);
    await sleep(2000);
    
    console.log('4. 보내기 버튼 클릭...');
    let sent = await p.evaluate(() => {{
        const btn = document.querySelector('button[data-testid=\"send-button\"]') ||
                    document.querySelector('button[aria-label=\"보내기\"]') ||
                    document.querySelector('button[aria-label=\"메시지 전송\"]');
        if (btn && !btn.disabled) {{
            btn.click();
            return true;
        }}
        return false;
    }});
    
    if (!sent) {{
        console.log('버튼 클릭 불가, Enter 키 시도');
        await p.keyboard.press('Enter');
    }}
    
    console.log('5. 생성 대기 상태 확인...');
    let generating = false;
    for (let i = 0; i < 15; i++) {{
        await sleep(500);
        generating = await p.evaluate(() => {{
            return !!document.querySelector('button[aria-label=\"생성 중지\"]') ||
                   !!document.querySelector('button[data-testid=\"stop-button\"]');
        }});
        if (generating) break;
    }}
    console.log('결과 isGenerating:', generating);
}}
await run();
"""

res = subprocess.run(
    ["/Users/wisdom/.local/bin/aside", "repl", aside_script],
    capture_output=True,
    text=True,
    timeout=60
)
print("STDOUT:", res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
