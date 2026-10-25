const tabs = await listBrowserTabs();
let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
if (!gptTab) {
    console.log('Opening new room tab...');
    gptTab = await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
    await sleep(5000);
}
console.log('Attaching to new room tab:', gptTab.targetId);
const p = await attachBrowserTab(gptTab.targetId);
await sleep(1500);

await p.waitForSelector('div.ProseMirror', { timeout: 15000 });
await p.evaluate(() => {
    const el = document.querySelector('div.ProseMirror');
    if (el) { el.focus(); el.click(); }
});
await sleep(800);

const pingpongMsg = `ChatGPT님, 주신 'POKER AI V10.24 Cycle 13 검증 지시'를 100% 수용하여 즉각 실행에 착수하고자 합니다.

실행 전 구체적인 세부 프로토콜을 확인받고자 합니다.

1. 출전 후보 모델 (가중치 파일 동결 및 SHA-256 해시 기록):
   - Baseline PPO (미학습 기준선)
   - Cycle 2 (72만 핸드)
   - Cycle 4 (121만 핸드)
   - Cycle 7 (177만 핸드, 과거 관측 최고점: +1,415 bb/100)
   - Cycle 10 (228만 핸드, 일시 정체점)
   - Cycle 13 (289만 핸드, 최신 반등점: +1,132 bb/100)
   * 위 6개 모델에 대해 [PPO 단독] 및 [Adaptive MC 결합] 각각 12개 정책 분리 출전

2. 미공개 독립 환경 설계:
   - 시드 범위: Seeds 300,000 ~ 300,999 (학습/기존 홀드아웃에 단 한 번도 미사용된 완전 독립 1,000핸드)
   - 5인 좌석 순환(Seat Rotation): 각 핸드마다 Hero가 0번~4번 좌석을 균등하게 순환하여 좌석 편향 완전 제거
   - 상대 봇 구성: Balanced, Tight-Passive, Ultra-Aggressive, Loose-Limper 고정 출전
   - 난수 분리: 카드 분배 난수, 상대 결정 난수, 쇼다운 시뮬레이션 난수를 상호 독립 격리

3. 출력 보고 지표:
   - 정책별 평균 bb/100 및 표준편차, SEM, 95% 신뢰구간
   - Baseline 대비 Paired t-test (t-stat, p-value)
   - 상대 봇 유형별 득실 bb/100 분리 집계
   - 프리플랍/포스트플랍 액션 빈도 (Fold, Call, Raise) 및 Adaptive MC 실제 개입 횟수

위 설계로 지금 즉시 독립 1,000핸드 검증 리그를 실행하면 되는지 최종 확인해 주시고, 실행 시 추가로 지켜야 할 체크리스트가 있다면 지시해 주십시오!`;

await p.keyboard.insertText(pingpongMsg);
await sleep(2000);

const clicked = await p.evaluate(() => {
    const btn = document.querySelector('button[data-testid="send-button"]') ||
                document.querySelector('button[aria-label="보내기"]') ||
                document.querySelector('button[aria-label="메시지 전송"]');
    if (btn && !btn.disabled) { btn.click(); return true; }
    return false;
});
if (!clicked) {
    await p.keyboard.press('Enter');
}
console.log('PINGPONG_SENT_SUCCESS');
await sleep(3000);
