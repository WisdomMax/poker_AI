import subprocess

js = """
async function listAllTabs() {
    const allTabs = typeof tabs === 'function' ? await tabs() : tabs;
    console.log("Tabs:", JSON.stringify(allTabs, null, 2));
}
await listAllTabs();
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
