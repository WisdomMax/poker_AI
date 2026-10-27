import subprocess

js = """
async function testInput() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    const loc = p.locator('div.ProseMirror[contenteditable="true"]');
    const count = await loc.count();
    console.log("ProseMirror count:", count);
    if (count > 0) {
        const lastInput = loc.nth(count - 1);
        await lastInput.click();
        console.log("Clicked last ProseMirror input successfully!");
    }
}
await testInput();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
