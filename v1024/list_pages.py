import subprocess

js = """
async function listTabs() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    const browser = p.context();
    const pages = browser.pages();
    console.log("Total open pages:", pages.length);
    for (let i = 0; i < pages.length; i++) {
        const pg = pages[i];
        console.log(`[${i}] Title: ${await pg.title()} | URL: ${pg.url()}`);
    }
}
await listTabs();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
