import subprocess

js = """
async function findHoldemTab() {
    const browser = await getBrowser();
    const contexts = browser.contexts();
    console.log("Contexts:", contexts.length);
    for (let c = 0; c < contexts.length; c++) {
        const pages = contexts[c].pages();
        console.log(`Context ${c} pages:`, pages.length);
        for (let p = 0; p < pages.length; p++) {
            const page = pages[p];
            const title = await page.title();
            const url = page.url();
            console.log(`Page [${c}:${p}] Title: ${title} | URL: ${url}`);
        }
    }
}
await findHoldemTab();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
