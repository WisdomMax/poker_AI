import subprocess
import time

message = """[POKER AI V5.2.2 핵심 알고리즘 정합성 검증 및 EV 감사 결과 보고]

1. 99 핸드 확률 오류 공식 정정
- 남은 22장 중 TT~AA(30) + AKs(4) = 34콤보 기준, 상대 4명의 합집합 상한(Union Bound)은 58.87%입니다.
- 1,000,000회 몬테카를로 실측 참 확률은 47.10% ± 0.10%로, 이전의 '80% 초과' 주장은 수학적 계산 오류였음을 증명하고 공식 정정했습니다.
- 24장 덱에서 99의 Equity는 4 Random 상대 20.19%, Facing All-in 상대 19.06%로 평균 수준을 유지합니다.

2. 99 Facing All-in EV 모순 규명
- 상대 프리미엄 레인지(Realistic Shove: TT~AA, AKs) 직면 시:
  * Fold Net Return: -3.50 BB (확정 손실)
  * Call Net Return: -4.92 ± 0.78 BB (95% CI)
  * Call - Fold EV: -1.42 BB (Fold가 참 최적 행동)
- 상대 셔브 직면 시 99 폴드는 결함이 아니라 -1.42 BB의 추가 손실을 막는 합리적 리스크 방어 전략임이 확인되었습니다.

3. NFSP-DQN Reference 원 논문 규격 확립
- Heinrich & Silver (2016) 원 논문 규격(에피소드 레벨 eta 샘플링, Target Q-Network, 1-step Bellman 타겟)을 완벽 구현하여 기존 Monte Carlo 회귀 변형(NFSP-MC)과 분리 대조했습니다.

4. 공통 난수(CRN) 1,000판 대응 교차 대국
- NFSP-DQN은 CheckCall 봇을 상대로 +54.57 bb/100(기존 MC는 -22.28 bb/100)을 달성하며 뚜렷한 성능 개선을 보였습니다.
- PPO 상대 대국에서는 상호 공방을 형성하여 일방적 압도 가설은 성립하지 않음을 재확인했습니다.

5. 100BB 진입 판단 및 제언
- 100BB 전면 학습은 행동 추상화 부재로 보류하며, 다음 단계 관문으로 '5인 20BB 숏스택 학습'을 권고합니다.
"""

def send_to_chatgpt():
    # 1. 클립보드에 복사
    process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(message.encode('utf-8'))
    print("[1] 클립보드에 메시지 복사 완료.")

    # 2. AppleScript로 ChatGPT 앱 활성화 및 붙여넣기 + 전송
    applescript = '''
    tell application "ChatGPT"
        activate
    end tell
    delay 0.8
    tell application "System Events"
        tell process "ChatGPT"
            -- 클립보드 붙여넣기 (Cmd + V)
            keystroke "v" using command down
            delay 0.5
            -- 전송 (Enter)
            key code 36
        end tell
    end tell
    '''
    res = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)
    if res.returncode == 0:
        print("[2] ChatGPT '홀덤 기계적 전략' 창에 메시지 붙여넣기 및 전송(Enter) 실행 완료!")
    else:
        print("[오류 발생]:", res.stderr)

if __name__ == '__main__':
    send_to_chatgpt()
