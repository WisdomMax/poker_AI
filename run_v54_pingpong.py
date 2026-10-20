import json
import subprocess
import time
import sys

def run_aside_pingpong():
    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/v54_send_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[*] 전송할 텍스트 준비 완료 ({len(text):,}자)")

    js_template = """
const content = __CONTENT_PLACEHOLDER__;

async function run() {
  console.log('[1/4] Aside 브라우저에서 ChatGPT 대화방 열기...');
  await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  await sleep(3000);

  console.log('[2/4] 프롬프트 입력창에 결과 텍스트 주입...');
  const fillRes = await page.evaluate((msg) => {
    const el = document.querySelector('#prompt-textarea') || document.querySelector('div[contenteditable="true"]');
    if (!el) return { ok: false, reason: 'no_input_element' };
    el.focus();
    if (el.tagName.toLowerCase() === 'textarea') {
      el.value = msg;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      el.innerText = msg;
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: msg }));
    }
    return { ok: true, tag: el.tagName };
  }, content);

  console.log('입력 주입 결과:', JSON.stringify(fillRes));
  await sleep(1500);

  console.log('[3/4] 보내기 버튼 클릭 및 발송 검증...');
  let sent = false;
  for (let attempt = 1; attempt <= 5; attempt++) {
    const clickRes = await page.evaluate(() => {
      const btn = document.querySelector('button[aria-label="보내기"]') || 
                  document.querySelector('button[data-testid="send-button"]') || 
                  document.querySelector('button[aria-label="프롬프트 보내기"]');
      if (btn && !btn.disabled) {
        btn.click();
        return true;
      }
      return false;
    });

    await sleep(2000);
    const checkState = await page.evaluate(() => {
      const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                      document.querySelector('button[data-testid="stop-button"]');
      const sendBtn = document.querySelector('button[aria-label="보내기"]');
      return {
        isGenerating: !!stopBtn,
        hasSendBtn: !!sendBtn
      };
    });

    if (checkState.isGenerating) {
      console.log('[발송 확인 성공 (시도 ' + attempt + ')] ChatGPT가 답변 생성을 시작했습니다!');
      sent = true;
      break;
    } else {
      console.log('[재시도 ' + attempt + '/5] 아직 전송 버튼 미반영, 엔터 키 입력 시도...');
      await page.keyboard.press('Enter');
    }
  }

  if (!sent) {
    console.log('[경고] 5회 시도 후에도 발송 미확인');
    return;
  }

  console.log('[4/4] ChatGPT 답변 실시간 생성 모니터링 시작...');
  let lastLen = 0;
  for (let t = 1; t <= 60; t++) {
    await sleep(3000);
    const progress = await page.evaluate(() => {
      const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                      document.querySelector('button[data-testid="stop-button"]');
      const bodyText = document.body.innerText;
      return {
        isGenerating: !!stopBtn,
        len: bodyText.length,
        tail: bodyText.slice(-200)
      };
    });

    if (progress.len !== lastLen) {
      console.log('  -> 답변 작성 중... [' + (t*3) + '초] 현재 페이지 길이: ' + progress.len + '자');
      lastLen = progress.len;
    }

    if (!progress.isGenerating && t >= 2) {
      console.log('[완료] ChatGPT 답변 작성이 완료되었습니다! (총 소요: ' + (t*3) + '초)');
      break;
    }
  }

  // 최신 응답 본문 및 코드블록 추출
  const result = await page.evaluate(() => {
    const articles = document.querySelectorAll('article');
    const lastArticle = articles.length > 0 ? articles[articles.length - 1] : null;
    const codeBlocks = Array.from(document.querySelectorAll('pre code')).map(c => c.innerText);
    
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
      fullText: lastArticle ? lastArticle.innerText : document.body.innerText.slice(-3000),
      codeBlocks: codeBlocks,
      copiedPrompt: copiedText
    };
  });

  console.log('===FINAL_RESPONSE_DATA_START===');
  console.log(JSON.stringify(result));
  console.log('===FINAL_RESPONSE_DATA_END===');
}

await run();
"""

    js_code = js_template.replace("__CONTENT_PLACEHOLDER__", json.dumps(text))

    p = subprocess.Popen(
        ['/Users/wisdom/.local/bin/aside', 'repl', js_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    full_output = []
    for line in iter(p.stdout.readline, ''):
        print(line, end='', flush=True)
        full_output.append(line)
    p.stdout.close()
    p.wait()

    raw = "".join(full_output)
    if "===FINAL_RESPONSE_DATA_START===" in raw:
        start = raw.index("===FINAL_RESPONSE_DATA_START===") + len("===FINAL_RESPONSE_DATA_START===")
        end = raw.index("===FINAL_RESPONSE_DATA_END===")
        data = json.loads(raw[start:end].strip())
        
        with open('/Volumes/NVME/7.AI_vibe_coding/hangame/latest_gpt_response.txt', 'w', encoding='utf-8') as f:
            f.write(data.get('fullText', ''))
        
        prompt = data.get('copiedPrompt')
        if prompt:
            with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v55_instruction_raw.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"\n[성공] 다음 단계 지침({len(prompt):,}자)이 v55_instruction_raw.txt 에 저장되었습니다!")

if __name__ == '__main__':
    run_aside_pingpong()
