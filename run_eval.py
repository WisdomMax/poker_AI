import subprocess
with open("inspect_aside.js", "r") as f:
    js = f.read()
# wrap in async IIFE
wrapped_js = f"(async () => {{ {js} }})()"
escaped = wrapped_js.replace('\\', '\\\\').replace('"', '\\"')
as_cmd = f'tell application "Aside" to execute javascript "{escaped}"'
res = subprocess.run(["osascript", "-e", as_cmd], capture_output=True, text=True)
print(res.stdout)
