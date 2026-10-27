import subprocess

js = """
async function inspect() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    console.log("Current URL:", p.url());
    console.log("Current Title:", await p.title());
    
    // Check elements
    const elements = await p.evaluate(() => {
        const textareas = Array.from(document.querySelectorAll('textarea, div[contenteditable="true"], div#prompt-textarea, [data-placeholder]')).map(el => ({
            tag: el.tagName,
            id: el.id,
            className: el.className,
            contenteditable: el.getAttribute('contenteditable'),
            placeholder: el.getAttribute('placeholder') || el.getAttribute('data-placeholder'),
            rect: el.getBoundingClientRect()
        }));
        return textareas;
    });
    console.log("Found input elements:", JSON.stringify(elements, null, 2));
}
await inspect();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
