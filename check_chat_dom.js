const tabs = await listBrowserTabs();
const gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
if (!gptTab) {
    console.log('Tab not found');
} else {
    const p = await attachBrowserTab(gptTab.targetId);
    
    // 현재 입력창 내용 확인
    const inputs = p.locator('div[contenteditable="true"]');
    const lastInput = inputs.last();
    const text = await lastInput.innerText();
    console.log('Input length:', text.length);
    console.log('Input snippet:', text.substring(0, 100));
    
    // 전송 버튼 상태
    const sendBtn = p.locator('button[data-testid="send-button"], button[aria-label="보내기"], button[aria-label="메시지 전송"]').last();
    const btnExists = await sendBtn.count();
    console.log('Send button exists:', btnExists);
    if (btnExists > 0) {
        const disabled = await sendBtn.isDisabled();
        console.log('Send button disabled:', disabled);
        if (!disabled && text.length > 10) {
            console.log('Clicking send button...');
            await sendBtn.click();
        }
    }
}
