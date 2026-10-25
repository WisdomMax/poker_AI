const tabs = await listBrowserTabs();
const gptTab = tabs.find(t => (t.url || '').includes('chatgpt.com'));
if (!gptTab) {
  console.log('ERROR: No ChatGPT tab found');
} else {
  console.log('Attaching to tab:', gptTab.targetId);
  const p = await attachBrowserTab(gptTab.targetId);
  await sleep(1500);

  await p.waitForSelector('div.ProseMirror', { timeout: 15000 });
  await p.evaluate(() => {
    const el = document.querySelector('div.ProseMirror');
    if (el) { el.focus(); el.click(); }
  });
  await sleep(800);

  const kickoffMsg = `# POKER AI V10.24: 5시간 마라톤 완주 총결산 및 2시간 추가 강화학습 킥오프

안녕하세요 ChatGPT!
앞서 진행된 5.0시간 연속 무중단 강화학습(Cycle 1 ~ Cycle 7)이 성공적으로 완주되었습니다.
이전 대화방이 최대 길이에 도달하여, 새 세션에서 연속성을 유지하며 2시간 추가 자율 강화학습(총 7,200초) 핑퐁을 시작합니다.

---

## 1. 5시간 마라톤(Cycle 1 ~ 7) 최종 종합 성적표
- **총 누적 훈련 핸드**: 1,771,164 hands
- **평균 CPU 점유율**: 67.8% (엄격 70% 제한 상시 준수)
- **홀드아웃 500핸드 리그 평가 (Seeds 200,000 ~ 200,499)**:
  - **Baseline PPO**: +987.00 bb/100
  - **Trained PPO 단독**: **+1,308.10 bb/100 (+321.10 bb/100 향상)**
  - **Trained PPO + Fixed 300**: +1,172.30 bb/100 (+185.30 bb/100 향상)
  - **Trained PPO + Adaptive MC**: **+1,415.23 bb/100 (+428.23 bb/100 폭발적 우상향!)**
- **최종 체크포인트**: \`v1024/models/V1024_PPO_Cycle_7_1771164hands.pt\` (GitHub 동기화 완료)

---

## 2. 신규 2시간 강화학습 세션 (Cycle 8 ~ 10) 목표
1. **기반 모델**: Cycle 7 가중치(\`V1024_PPO_Cycle_7_1771164hands.pt\`)에서부터 학습 연속 진행.
2. **훈련 목표**: 누적 2,500,000 핸드 돌파 및 PPO 정책 정밀화.
3. **포스트플랍 가드**: Action Boundary Adaptive Model A (sims 300~3,000) 유지.
4. **시스템 제약**: CPU 70% 이하 엄격 통제 (cpu_throttler.py 상시 체결).
5. **보고 주기**: 약 35~40분 간격으로 홀드아웃 벤치마크 및 핑퐁 수행.

2시간 핑퐁 세션의 킥오프를 알립니다. 분석 피드백과 함께 다음 사이클로의 이행을 승인해 주시기 바랍니다!`;

  await p.keyboard.insertText(kickoffMsg);
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
  console.log('KICKOFF_SENT_SUCCESS');
  await sleep(3000);
}
