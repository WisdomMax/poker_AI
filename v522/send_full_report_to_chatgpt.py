import subprocess
import time
import os

report_path = "/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/v522_nfsp_integrity_report.md"

def send_full_report():
    # 1. 보고서 전체 내용 읽기
    with open(report_path, 'r', encoding='utf-8') as f:
        full_content = f.read()

    print(f"[1] 보고서 전체 내용 로드 완료 ({len(full_content)} 글자)")

    # 2. 클립보드에 복사 (pbcopy)
    p = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    p.communicate(full_content.encode('utf-8'))
    print("[2] 클립보드에 전체 보고서 복사 완료.")

    # 3. ChatGPT 앱 활성화
    subprocess.run(['osascript', '-e', 'tell application "ChatGPT" to activate'])
    time.sleep(0.8)

    # 4. cliclick으로 ChatGPT 입력창 위치(X=1280, Y=1260) 클릭하여 포커스 맞추기
    cliclick_path = "/opt/homebrew/bin/cliclick"
    subprocess.run([cliclick_path, 'c:1280,1260'])
    time.sleep(0.4)

    # 5. 붙여넣기 (Cmd + V)
    subprocess.run([cliclick_path, 'kd:cmd', 't:v', 'ku:cmd'])
    time.sleep(0.8)

    # 6. 전송 (Return / Enter)
    subprocess.run([cliclick_path, 'kp:return'])
    print("[3] ChatGPT '홀덤 기계적 전략' 창에 보고서 전체 붙여넣기 및 전송 완료!")

if __name__ == '__main__':
    send_full_report()
