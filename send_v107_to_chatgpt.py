import json
import subprocess

def main():
    report_path = '/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/v107_reproducible_evaluation_harness.md'
    with open(report_path, 'r', encoding='utf-8') as f:
        full_report = f.read()

    print(f"[V10.7 발송]: 보고서 {len(full_report)}자 100% 전문 로드 완료")

    js_code = f"""
const textContent = {json.dumps(full_report)};

async function run() {{
  console.log('1. 열린 탭 중 ChatGPT 탭 탐색...');
  const openTabs = await listBrowserTabs();
  const gptTab = openTabs.find(t => t.url.includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
  if (gptTab) {{
    console.log('Attaching to existing tab:', gptTab.targetId);
    await attachBrowserTab(gptTab.targetId);
  }} else {{
    console.log('Opening new tab for fixed room...');
    await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  }}
  await sleep(2500);

  console.log('2. 하단 입력창(ProseMirror) 포커스...');
  await page.evaluate(() => {{
    const all = document.querySelectorAll('div.ProseMirror');
    const target = all[all.length - 1];
    if (target) {{
      target.focus();
      target.click();
    }}
  }});
  await sleep(800);

  console.log('3. 키보드로 100% 전문 주입 시작 (' + textContent.length + '자)...');
  await page.keyboard.insertText(textContent);
  await sleep(2000);

  console.log('4. 보내기 버튼 감지 및 클릭...');
  const clickRes = await page.evaluate(() => {{
    const sendBtn = document.querySelector('button[aria-label="보내기"]') || 
                    document.querySelector('button[data-testid="send-button"]') ||
                    document.querySelector('button[aria-label="Send prompt"]');
    if (sendBtn) {{
      sendBtn.click();
    }}
    return {{ ok: !!sendBtn, aria: sendBtn ? sendBtn.getAttribute('aria-label') : null }};
  }});
  console.log('Click result:', JSON.stringify(clickRes));
  await sleep(3000);

  console.log('5. 발송 검증 (생성 중지 버튼 확인)...');
  const check = await page.evaluate(() => {{
    const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                    document.querySelector('button[data-testid="stop-button"]') ||
                    document.querySelector('button[aria-label*="Stop"]');
    return {{ isGenerating: !!stopBtn, aria: stopBtn ? stopBtn.getAttribute('aria-label') : null }};
  }});

  console.log('RESULT_IS_GENERATING:', JSON.stringify(check));
}}

await run();
"""

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

if __name__ == '__main__':
    main()
