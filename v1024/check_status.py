import subprocess

js = """
async function check() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    const title = await p.title();
    console.log("Tab title:", title);
    const articles = await p.$$("article");
    console.log("Article count:", articles.length);
    if (articles.length > 0) {
        const lastArt = articles[articles.length - 1];
        const text = await lastArt.innerText();
        console.log("Last article snippet:", text.slice(0, 200).replace(/\\n/g, " "));
    }
    const sendBtn = await p.$('button[data-testid="send-button"]');
    if (sendBtn) {
        const dis = await sendBtn.evaluate(el => el.disabled);
        console.log("Send button present, disabled:", dis);
    } else {
        console.log("Send button not found");
    }
}
await check();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
