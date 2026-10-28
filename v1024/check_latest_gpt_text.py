import subprocess

js = """
async function check() {
    const p = await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
    await sleep(3000);
    console.log('Page Title:', await p.title());
    const text = await p.evaluate(() => {
        const m = document.querySelector('main');
        return m ? m.innerText : document.body.innerText;
    });
    console.log('Text length:', text.length);
    console.log('===TAIL_START===');
    console.log(text.slice(-4000));
    console.log('===TAIL_END===');
}
await check();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True, timeout=30)
print(res.stdout + res.stderr)
