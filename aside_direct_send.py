import json
import subprocess
import time

def main():
    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/v54_send_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    js_template = """
const msg = __TEXT_PLACEHOLDER__;

async function run() {
  console.log('1. Aside에서 ChatGPT 열기...');
  await openTab('https://chatgpt.com');
  await sleep(3500);

  console.log('2. ProseMirror 입력창에 텍스트 주입...');
  const res = await page.evaluate(async (content) => {
    const pm = document.querySelector('div.ProseMirror') || document.querySelector('#prompt-textarea');
    if (!pm) return { ok: false, reason: 'no_input' };

    pm.focus();
    pm.innerHTML = '<p>' + content.replace(/\\n/g, '<br>') + '</p>';
    pm.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: content }));
    return { ok: true };
  }, msg);

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

  console.log('6. 발송 검증 (답변 생성 여부):', JSON.stringify(check));
}

await run();
"""

    js_code = js_template.replace("__TEXT_PLACEHOLDER__", json.dumps(text))

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

if __name__ == '__main__':
    main()
