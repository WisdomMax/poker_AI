async function run() {
  const openTabs = await listBrowserTabs();
  const gptTab = openTabs.find(t => t.url.includes('6ad12105-6404-83ec-b272-9a671bac23f9'));
  if (gptTab) {
    await attachBrowserTab(gptTab.targetId);
  } else {
    await openTab('https://chatgpt.com/c/6ad12105-6404-83ec-b272-9a671bac23f9');
  }
  await sleep(3000);

  const fullText = await page.evaluate(() => document.body.innerText);
  console.log('=== LATEST CHAT TEXT START ===');
  console.log(fullText.slice(-5000));
  console.log('=== LATEST CHAT TEXT END ===');
}
await run();
