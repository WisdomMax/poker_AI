import json
import subprocess

def get_latest_response():
    js_code = """
async function run() {
  const openTabs = await listBrowserTabs();
  const gptTab = openTabs.find(t => t.url.includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
  if (gptTab) {
    console.log('Attaching to existing tab:', gptTab.targetId);
    await attachBrowserTab(gptTab.targetId);
  } else {
    console.log('Opening new tab for fixed room...');
    await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  }
  await sleep(2000);

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
    const assistantTurns = document.querySelectorAll('[data-message-author-role="assistant"]');
    if (assistantTurns.length > 0) {
      const last = assistantTurns[assistantTurns.length - 1];
      return {
        ok: true,
        text: last.innerText
      };
    }
    const articles = document.querySelectorAll('article');
    if (articles.length > 0) {
      const last = articles[articles.length - 1];
      return {
        ok: true,
        text: last.innerText
      };
    }
    return { ok: false, reason: 'no_message_found' };
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
        print("STDERR:\n", res.stderr)
    return None

if __name__ == '__main__':
    data = get_latest_response()
    if data and data.get('ok'):
        text = data['text']
        print(f"Successfully extracted {len(text)} characters.")
        with open('/Volumes/NVME/7.AI_vibe_coding/hangame/latest_gpt_response.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Saved to latest_gpt_response.txt")
        print("\n--- Full / Snippet ---\n")
        print(text[:1500])
    else:
        print("Failed to extract:", data)
