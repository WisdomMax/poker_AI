async function run() {
  await attachBrowserTab('DC747975B223E18C360A14F4EC148ABA');
  const text = await page.evaluate(() => {
    const msgs = document.querySelectorAll('[data-message-author-role]');
    if (msgs.length > 0) {
      return Array.from(msgs).map(m => m.getAttribute('data-message-author-role') + ':\n' + m.innerText).join('\n\n===MSG===\n\n');
    }
    const main = document.querySelector('main');
    return main ? main.innerText : document.body.innerText;
  });
  console.log('EXTRACTED LENGTH:', text.length);
  return text;
}
const output = await run();
console.log('OUTPUT_START\n' + output + '\nOUTPUT_END');
