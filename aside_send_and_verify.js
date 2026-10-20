// aside_send_and_verify.js

async function run() {
  const text = await fs.readFile('/Volumes/NVME/7.AI_vibe_coding/hangame/v534/v534_send_text.txt', 'utf8');
  console.log('Read text length:', text.length);

  await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  await sleep(3000);

  // 입력창 찾기 및 텍스트 설정
  const fillSuccess = await page.evaluate((content) => {
    const el = document.querySelector('#prompt-textarea') || document.querySelector('div[contenteditable="true"]');
    if (!el) return false;
    
    el.focus();
    if (el.tagName.toLowerCase() === 'textarea') {
      el.value = content;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      // contenteditable div
      el.innerText = content;
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: content }));
    }
    return true;
  }, text);

  console.log('Fill success:', fillSuccess);
  await sleep(1000);

  // 전송 버튼 클릭
  const clickSuccess = await page.evaluate(() => {
    const btn = document.querySelector('button[data-testid="send-button"]') || 
                document.querySelector('button[aria-label="프롬프트 보내기"]') || 
                document.querySelector('button[aria-label="Send prompt"]');
    if (btn && !btn.disabled) {
      btn.click();
      return true;
    }
    return false;
  });

  console.log('Send button clicked:', clickSuccess);

  // 사후 검증: 3초 후 생성 중지 버튼으로 바뀌었는지 확인
  await sleep(3000);
  const verifyResult = await page.evaluate(() => {
    const stopBtn = document.querySelector('button[data-testid="stop-button"]') || 
                    document.querySelector('button[aria-label*="중지"]') || 
                    document.querySelector('button[aria-label*="Stop"]') ||
                    document.querySelector('button svg rect');
    const sendBtn = document.querySelector('button[data-testid="send-button"]') || 
                    document.querySelector('button[aria-label="프롬프트 보내기"]');
    return {
      isGenerating: !!stopBtn,
      sendButtonPresent: !!sendBtn,
      sendButtonDisabled: sendBtn ? sendBtn.disabled : null
    };
  });

  console.log('Verification result:', JSON.stringify(verifyResult));
}

await run();
