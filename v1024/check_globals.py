import subprocess

js = """
console.log("Global keys:", Object.keys(globalThis).filter(k => !k.startsWith('_')).slice(0, 30));
"""

res = subprocess.run(["/Users/wisdom/.local/bin/aside", "repl", js], capture_output=True, text=True)
print(res.stdout + res.stderr)
