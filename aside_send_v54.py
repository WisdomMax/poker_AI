import json
import subprocess
import time

def main():
    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/v54_send_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    js_code = f"""
const content = {json.dumps(text)};

async function run() {{
  console.log('Opening tab...');
  await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  await sleep(2000);

  console.log('Filling prompt textarea...');
  const fillRes = await page.evaluate((msg) => {{
    const el = document.querySelector('#prompt-textarea') || document.querySelector('div[contenteditable="true"]');
    if (!el) return {{ ok: false, reason: 'no_input_element' }};
    el.focus();
    if (el.tagName.toLowerCase() === 'textarea') {{
      el.value = msg;
      el.dispatchEvent(new Event('input', {{ bubbles: true }}));
      el.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }} else {{
      el.innerText = msg;
      el.dispatchEvent(new InputEvent('input', {{ bubbles: true, inputType: 'insertText', data: msg }}));
    }}
    return {{ ok: true, tag: el.tagName }};
  }}, content);

  console.log('Fill result:', JSON.stringify(fillRes));
  await sleep(1500);

  console.log('Clicking send button...');
  const clickRes = await page.evaluate(() => {{
    const btn = document.querySelector('button[aria-label="보내기"]') || 
                document.querySelector('button[data-testid="send-button"]') || 
                document.querySelector('button[aria-label="프롬프트 보내기"]');
    if (!btn) return {{ ok: false, reason: 'no_send_button' }};
    if (btn.disabled) return {{ ok: false, reason: 'button_disabled' }};
    btn.click();
    return {{ ok: true }};
  }});

  console.log('Click result:', JSON.stringify(clickRes));

  // 사후 검증: 3초 후 생성 중지 버튼으로 바뀌었는지 확인
  await sleep(3000);
  const verifyRes = await page.evaluate(() => {{
    const stopBtn = document.querySelector('button[aria-label*="중지"]') || 
                    document.querySelector('button[data-testid="stop-button"]');
    const sendBtn = document.querySelector('button[aria-label="보내기"]');
    return {{
      isGenerating: !!stopBtn,
      stopAria: stopBtn ? stopBtn.getAttribute('aria-label') : null,
      hasSendBtn: !!sendBtn
    }};
  }});

  console.log('POST_VERIFICATION:', JSON.stringify(verifyRes));
}}

await run();
"""

    res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

if __name__ == '__main__':
    main()
