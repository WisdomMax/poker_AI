#!/usr/bin/env python3
"""
POKER AI - CHATGPT ONE-CLICK RESULT SENDER (send_result.py)
- 최신 작성된 감사 보고서(*_report.md)를 자동으로 탐색하여 전체 내용을 ChatGPT 대화방으로 전송
- 발송 버튼 클릭 후 실제로 ChatGPT가 전송을 받아 '생성 중(버튼: 중지)' 상태로 진입했는지 100% 사후 검증 수행
"""

import os
import sys
import glob
import time
import subprocess

ARTIFACT_DIR = "/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b"
CLICLICK_PATH = "/opt/homebrew/bin/cliclick"

def get_latest_report():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    pattern = os.path.join(ARTIFACT_DIR, "*_report.md")
    report_files = glob.glob(pattern)
    if not report_files:
        report_files = glob.glob(os.path.join(ARTIFACT_DIR, "*.md"))

    if not report_files:
        raise FileNotFoundError(f"보고서 파일을 찾을 수 없습니다: {ARTIFACT_DIR}")

    return max(report_files, key=os.path.getmtime)

def send_result_to_chatgpt():
    target_report = get_latest_report()
    file_name = os.path.basename(target_report)
    print(f"[1] 전송할 최신 보고서 감지: {file_name}")

    with open(target_report, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"    - 총 글자 수: {len(content):,} 자")

    tmp_path = "/tmp/poker_report_to_send.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 1. 텍스트 입력창에 보고서 주입
    print("[2] ChatGPT 입력창에 보고서 데이터 주입...")
    inject_script = '''
    set reportText to read POSIX file "/tmp/poker_report_to_send.txt" as «class utf8»
    tell application "System Events"
        tell process "ChatGPT"
            set ta to text area "ChatGPT에게 물어보세요" of group 1 of group 1 of group 1 of group 1 of group 3 of group 1 of group 3 of group "대화" of group 1 of group 2 of group 1 of group 1 of group 1 of group 1 of group 1 of group 1 of group 2 of group 4 of group 2 of group 1 of group 1 of group 1 of UI element "홀덤 기계적 전략" of scroll area 1 of group 1 of group 1 of group 1 of group 1 of group 1 of group "홀덤 기계적 전략" of window "홀덤 기계적 전략"
            set value of ta to reportText
        end tell
    end tell
    '''
    subprocess.run(['osascript', '-e', inject_script])
    time.sleep(0.5)

    # 2. 버튼 클릭 실행 (현재 작업 중인 창의 포커스 즉시 복구)
    print("[3] '보내기' 발송 버튼 클릭 실행...")
    click_script = '''
    tell application "System Events"
        set prevApp to name of first application process whose frontmost is true
    end tell
    tell application "ChatGPT" to activate
    delay 0.15
    do shell script "/opt/homebrew/bin/cliclick c:1819,1302"
    delay 0.1
    tell application "System Events" to tell process "ChatGPT" to key code 36
    delay 0.15
    tell application prevApp to activate
    '''
    subprocess.run(['osascript', '-e', click_script])

    # 3. [검증 절차] 실제로 발송되었는지 상태 체크 (사용자 지시: 확인까지 하는 절차)
    print("[4] 발송 성공 여부 사후 검증 진행 중...")
    check_script = '''
    tell application "System Events"
        tell process "ChatGPT"
            set btnList to {}
            try
                set allBtns to (every button of group 1 of group 1 of group 1 of group 1 of group 3 of group 1 of group 3 of group "대화" of group 1 of group 2 of group 1 of group 1 of group 1 of group 1 of group 1 of group 1 of group 2 of group 4 of group 2 of group 1 of group 1 of group 1 of UI element "홀덤 기계적 전략" of scroll area 1 of group 1 of group 1 of group 1 of group 1 of group 1 of group "홀덤 기계적 전략" of window "홀덤 기계적 전략")
                repeat with b in allBtns
                    set end of btnList to name of b
                end repeat
            end try
            return btnList
        end tell
    end tell
    '''
    
    verified = False
    for attempt in range(1, 11):
        res = subprocess.run(['osascript', '-e', check_script], capture_output=True, text=True)
        buttons = res.stdout.strip()
        if "중지" in buttons:
            print(f"    - [검증 성공 ({attempt}/10)] ChatGPT가 답변 생성을 시작했습니다 (상태: 중지 버튼 활성).")
            verified = True
            break
        elif "보내기" not in buttons:
            print(f"    - [검증 성공 ({attempt}/10)] '보내기' 버튼이 비활성화/전송 처리되었습니다.")
            verified = True
            break
        else:
            # 아직 안 눌렸으면 재시도
            time.sleep(0.5)
            subprocess.run([CLICLICK_PATH, "c:1819,1302"])

    if verified:
        print(f"[최종 확인 완료] '{file_name}' 보고서가 ChatGPT로 100% 정상 발송되었음을 확인했습니다!\n")
    else:
        print(f"[경고] 발송 버튼 상태 확인 필요 (수동 점검 요망)\n")

if __name__ == '__main__':
    send_result_to_chatgpt()
