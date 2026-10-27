import subprocess

js = """
async function checkCurrentState() {
    const p = await attachBrowserTab("B953A3E68E4EA5A5D719FA615BB330C5");
    const title = await p.title();
    const url = p.url();
    console.log("Tab Title:", title);
    console.log("Tab URL:", url);
    const articles = await p.evaluate(() => {
        const arts = Array.from(document.querySelectorAll('article'));
        return arts.map(a => a.innerText.slice(0, 150).split("\\n").join(" "));
    });
    console.log("Total articles:", articles.length);
    if (articles.length >= 2) {
        console.log("Second to last article (Our submission):", articles[articles.length - 2]);
        console.log("Last article (ChatGPT response):", articles[articles.length - 1]);
    } else if (articles.length === 1) {
        console.log("Only 1 article:", articles[0]);
    }
}
await checkCurrentState();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
