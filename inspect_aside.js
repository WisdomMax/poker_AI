const tabs = await listBrowserTabs();
const gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
let out = '';
if (!gptTab) {
    out = 'Tab not found';
} else {
    const p = await attachBrowserTab(gptTab.targetId);
    const articles = p.locator('article');
    const artCount = await articles.count();
    out += `Articles count: ${artCount}\n`;
    
    // 마지막 article의 텍스트
    if (artCount > 0) {
        const lastArt = articles.last();
        const t = await lastArt.innerText();
        out += `Last article snippet (${t.length}):\n${t.substring(0, 300)}...\n--- Tail ---\n${t.substring(t.length - 300)}\n`;
    }
    
    const inputs = p.locator('div[contenteditable="true"]');
    const inputCount = await inputs.count();
    out += `Editable inputs: ${inputCount}\n`;
    if (inputCount > 0) {
        const inpText = await inputs.last().innerText();
        out += `Input text length: ${inpText.length}\n`;
        if (inpText.length > 0) {
            out += `Input text preview: ${inpText.substring(0, 100)}\n`;
        }
    }
}

// 브라우저 내부 document에 임시 보관하거나 Aside evaluate로 리턴
return out;
