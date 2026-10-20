import json
import subprocess
import time

def monitor():
    js_code = """
async function run() {
  await openTab('https://chatgpt.com');
  await sleep(2000);

  let lastLen = 0;
  for (let t = 1; t <= 40; t++) {
    await sleep(3000);
    const progress = await page.evaluate(() => {
      const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                      document.querySelector('button[data-testid="stop-button"]');
      const bodyText = document.body.innerText;
      return {
        isGenerating: !!stopBtn,
        len: bodyText.length,
        tail: bodyText.slice(-250)
      };
    });

    if (progress.len !== lastLen) {
      console.log('  -> [Aside 모니터링 ' + (t*3) + '초] 답변 작성 중... 길이: ' + progress.len + '자');
      lastLen = progress.len;
    }

    if (!progress.isGenerating && t >= 2) {
      console.log('[Aside 완료] 답변 생성이 완료되었습니다! (총 소요: ' + (t*3) + '초)');
      break;
    }
  }

  // 최신 응답 본문 및 지침 추출
  const result = await page.evaluate(() => {
    const articles = document.querySelectorAll('article');
    const lastArticle = articles.length > 0 ? articles[articles.length - 1] : null;
    const bodyText = document.body.innerText;
    
    // 복사 버튼이 있는 경우 복사 대상 텍스트 확인
    let copiedText = null;
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === '복사');
    if (btn) {
      let fiberKey = Object.keys(btn).find(k => k.startsWith('__reactFiber'));
      let cur = btn[fiberKey];
      while (cur) {
        if (cur.memoizedProps) {
          for (let k of Object.keys(cur.memoizedProps)) {
            let val = cur.memoizedProps[k];
            if (typeof val === 'string' && val.length > 300) {
              copiedText = val;
              break;
            }
          }
        }
        if (copiedText) break;
        cur = cur.return;
      }
    }

    return {
      snippet: lastArticle ? lastArticle.innerText.slice(0, 1000) : bodyText.slice(-1500),
      copiedPrompt: copiedText
    };
  });

  console.log('===RESPONSE_JSON_START===');
  console.log(JSON.stringify(result));
  console.log('===RESPONSE_JSON_END===');
}

await run();
"""

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    out = res.stdout
    print(out)
    
    if "===RESPONSE_JSON_START===" in out:
        start = out.index("===RESPONSE_JSON_START===") + len("===RESPONSE_JSON_START===")
        end = out.index("===RESPONSE_JSON_END===")
        data = json.loads(out[start:end].strip())
        
        with open('/Volumes/NVME/7.AI_vibe_coding/hangame/latest_gpt_response.txt', 'w', encoding='utf-8') as f:
            f.write(data.get('snippet', ''))
        
        prompt = data.get('copiedPrompt')
        if prompt:
            with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v55_instruction_raw.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"\n[성공] 다음 지침({len(prompt):,}자)이 v55_instruction_raw.txt 에 저장되었습니다!")

if __name__ == '__main__':
    monitor()
