import json
import subprocess

def main():
    js_code = """
async function run() {
  const openTabs = await listBrowserTabs();
  const gptTab = openTabs.find(t => t.url.includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
  if (gptTab) {
    await attachBrowserTab(gptTab.targetId);
  } else {
    await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  }
  await sleep(3000);

  const data = await page.evaluate(() => {
    // assistant turn 찾기
    const turns = document.querySelectorAll('article[data-testid^="conversation-turn-"]');
    let lastTurn = null;
    for (let i = turns.length - 1; i >= 0; i--) {
      const turn = turns[i];
      if (turn.querySelector('[data-message-author-role="assistant"]')) {
        lastTurn = turn;
        break;
      }
    }
    if (!lastTurn && turns.length > 0) {
      lastTurn = turns[turns.length - 1];
    }
    const text = lastTurn ? lastTurn.innerText : document.body.innerText;
    
    // 생성 중인지 확인
    const stopBtn = document.querySelector('button[aria-label="생성 중지"]') ||
                    document.querySelector('button[data-testid="stop-button"]') ||
                    document.querySelector('button[aria-label="중지"]');

    return {
      isGenerating: !!stopBtn,
      text: text
    };
  });

  console.log("=== LATEST CHAT TEXT START ===");
  console.log(data.text);
  console.log("=== LATEST CHAT TEXT END ===");
  console.log("IS_GENERATING:", data.isGenerating);
}

await run();
"""
    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    with open('latest_gpt_response.txt', 'w', encoding='utf-8') as f:
        f.write(res.stdout)
    print("STDOUT len:", len(res.stdout))
    if res.stderr:
        print("STDERR len:", len(res.stderr))

if __name__ == "__main__":
    main()
