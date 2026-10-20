import json
import subprocess
import time

def main():
    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/v54_send_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    js_template = """
const content = __CONTENT__;

async function run() {
  console.log('1. 기존 [홀덤 기계적 전략] 대화방으로 접속...');
  await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  await sleep(4000);

  console.log('2. 기존 대화방 ProseMirror 입력창에 결과 주입...');
  const res = await page.evaluate(async (msg) => {
    const pm = document.querySelector('div.ProseMirror') || document.querySelector('#prompt-textarea');
    if (!pm) return { ok: false, reason: 'no_input' };

    pm.focus();
    pm.innerHTML = '<p>' + msg.split('\\n').join('<br>') + '</p>';
    pm.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: msg }));
    return { ok: true };
  }, content);

  console.log('3. 입력 주입 결과:', JSON.stringify(res));
  await sleep(1500);

  console.log('4. 보내기 버튼 클릭...');
  const clickRes = await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label="보내기"]') || 
                document.querySelector('button[data-testid="send-button"]') || 
                document.querySelector('button[aria-label="프롬프트 보내기"]');
    if (btn) {
      btn.click();
      return { clicked: true, disabled: btn.disabled };
    }
    return { clicked: false };
  });

  console.log('5. 클릭 결과:', JSON.stringify(clickRes));
  await sleep(3000);

  const check = await page.evaluate(() => {
    const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                    document.querySelector('button[data-testid="stop-button"]');
    return { isGenerating: !!stopBtn };
  });

  console.log('6. [홀덤 기계적 전략] 발송 검증 (답변 생성 여부):', JSON.stringify(check));
}

await run();
"""

    js_code = js_template.replace("__CONTENT__", json.dumps(text))

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

if __name__ == '__main__':
    main()
