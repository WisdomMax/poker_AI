import subprocess

js = """
async function connectHoldem() {
    console.log("Listing tabs...");
    const tabList = typeof listBrowserTabs === 'function' ? await listBrowserTabs() : (typeof tabs === 'function' ? await tabs() : []);
    console.log("Open tabs:", JSON.stringify(tabList, null, 2));
    
    let target = null;
    if (Array.isArray(tabList)) {
        target = tabList.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    }
    
    let p;
    if (target) {
        console.log("Found existing tab:", target.targetId);
        p = await attachBrowserTab(target.targetId);
    } else {
        console.log("Opening new tab to target URL...");
        p = await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
        await sleep(5000);
    }
    
    console.log("Active page URL:", p.url());
    console.log("Active page Title:", await p.title());
    
    // Check articles count
    const articles = await p.evaluate(() => {
        const arts = Array.from(document.querySelectorAll('article'));
        return arts.map(a => a.innerText.slice(0, 100).split('\\n').join(' '));
    });
    console.log("Articles count:", articles.length);
    if (articles.length > 0) {
        console.log("Last article:", articles[articles.length - 1]);
    }
}
await connectHoldem();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
