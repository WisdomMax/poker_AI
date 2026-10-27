import subprocess

js = """
async function findButtons() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    const buttons = await p.evaluate(() => {
        return Array.from(document.querySelectorAll('button')).map(b => ({
            text: b.innerText,
            ariaLabel: b.getAttribute('aria-label'),
            dataTestId: b.getAttribute('data-testid'),
            disabled: b.disabled,
            className: b.className
        })).filter(b => (b.ariaLabel && (b.ariaLabel.includes('보내기') || b.ariaLabel.includes('전송') || b.ariaLabel.includes('Send'))) || (b.dataTestId && b.dataTestId.includes('send')));
    });
    console.log("Matching send buttons:", JSON.stringify(buttons, null, 2));
}
await findButtons();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
