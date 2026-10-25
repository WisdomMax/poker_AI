import subprocess
import sys

with open("v1024/send_directive015_report.js", "r", encoding="utf-8") as f:
    js_code = f.read()

# AppleScript string escape
escaped_js = js_code.replace('\\', '\\\\').replace('"', '\\"')

applescript = f'''tell application "Aside"
    execute javascript "{escaped_js}"
end tell'''

res = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("ReturnCode:", res.returncode)
