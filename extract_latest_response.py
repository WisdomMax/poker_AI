import json
import subprocess

def get_latest_response():
    js_code = """
async function run() {
  console.log('1. 열린 탭 탐색...');
  const openTabs = await listBrowserTabs();
  console.log('Open tabs:', JSON.stringify(openTabs));
  const gptTab = openTabs.find(t => t.url && t.url.includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
  if (gptTab) {
    console.log('Attaching to existing tab:', gptTab.targetId);
    await attachBrowserTab(gptTab.targetId);
  } else {
    console.log('Opening tab...');
    await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
  }
  await sleep(3000);

  // 생성 중인지 확인
  for (let i = 0; i < 30; i++) {
    const isGenerating = await page.evaluate(() => {
      const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                      document.querySelector('button[data-testid="stop-button"]') ||
                      document.querySelector('button[aria-label*="Stop"]');
      return !!stopBtn;
    });
    if (!isGenerating) {
      break;
    }
    console.log(`Generating in progress... wait 4s (${i+1}/30)`);
    await sleep(4000);
  }

  // 최신 응답 추출
  const res = await page.evaluate(() => {
    // 1. role="assistant"
    const assistantTurns = document.querySelectorAll('[data-message-author-role="assistant"]');
    if (assistantTurns.length > 0) {
      const last = assistantTurns[assistantTurns.length - 1];
      return { ok: true, source: 'data-message-author-role', text: last.innerText };
    }

    // 2. markdown / prose 클래스
    const proseEls = document.querySelectorAll('.markdown, .prose');
    if (proseEls.length > 0) {
      const last = proseEls[proseEls.length - 1];
      return { ok: true, source: 'markdown-class', text: last.innerText };
    }

    // 3. article 요소
    const articles = document.querySelectorAll('article');
    if (articles.length > 0) {
      const last = articles[articles.length - 1];
      return { ok: true, source: 'article', text: last.innerText };
    }

    const fullBody = document.body.innerText;
    return { 
      ok: true, 
      source: 'body-fallback', 
      text: fullBody.slice(-8000) 
    };
  });

  console.log('EXTRACT_RESULT:' + JSON.stringify(res));
}

await run();
"""

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    out = res.stdout
    for line in out.splitlines():
        if line.startswith('EXTRACT_RESULT:'):
            data = json.loads(line[len('EXTRACT_RESULT:'):])
            return data
    print("Full stdout:\n", out)
    if res.stderr:
        print("Full stderr:\n", res.stderr)
    return None

if __name__ == '__main__':
    data = get_latest_response()
    if data and data.get('ok'):
        text = data['text']
        print(f"Successfully extracted {len(text)} characters from {data.get('source')}.")
        with open('/Volumes/NVME/7.AI_vibe_coding/hangame/latest_gpt_response.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Saved to latest_gpt_response.txt")
        print("\n--- Snippet ---\n")
        print(text[:1500])
    else:
        print("Failed to extract:", data)
