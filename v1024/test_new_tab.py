import subprocess

js = """
async function testAttach() {
    const tabs = await listBrowserTabs();
    const holdem = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!holdem) {
        console.log("No holdem tab found");
        return;
    }
    console.log("Found holdem tab:", holdem.targetId);
    const p = await attachBrowserTab(holdem.targetId);
    console.log("Attached successfully! Page title:", await p.title());
    const textLen = await p.evaluate(() => document.body.innerText.length);
    console.log("Page text length:", textLen);
}
await testAttach();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True, timeout=30)
print(res.stdout + res.stderr)
