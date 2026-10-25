import json
import subprocess

with open('/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/v1023_16_hotfix_final_report.md', 'r', encoding='utf-8') as f:
    text = f.read()

# aside eval 명령으로 탭 열기부터 전송까지 완결
js = f'''
const targetUrl = 'https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9';
const tabs = await listBrowserTabs();
let tab = tabs.find(t => (t.url || '').includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
let p;
if (tab) {{
    p = await attachBrowserTab(tab.targetId);
}} else {{
    tab = await openTab(targetUrl);
    await sleep(4000);
    p = await attachBrowserTab(tab.targetId);
}}

await p.waitForSelector('div.ProseMirror', {{ timeout: 15000 }});
await p.evaluate(() => {{
    const el = document.querySelector('div.ProseMirror');
    if (el) {{ el.focus(); el.click(); }}
}});
await sleep(1000);

const content = {json.dumps(text)};
await p.keyboard.insertText(content);
await sleep(2000);

await p.keyboard.press('Enter');
await sleep(1000);

const clicked = await p.evaluate(() => {{
    const btn = document.querySelector('button[data-testid=\"send-button\"]') ||
                document.querySelector('button[aria-label=\"보내기\"]') ||
                document.querySelector('button[aria-label=\"메시지 전송\"]');
    if (btn && !btn.disabled) {{
        btn.click();
        return true;
    }}
    return false;
}});

await sleep(3000);
const isGen = await p.evaluate(() => {{
    return !!document.querySelector('button[aria-label=\"생성 중지\"]') ||
           !!document.querySelector('button[data-testid=\"stop-button\"]');
}});

console.log('Result isGenerating:', isGen);
'''

res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'eval', js], capture_output=True, text=True, timeout=60)
print('STDOUT:', res.stdout)
if res.stderr:
    print('STDERR:', res.stderr)
