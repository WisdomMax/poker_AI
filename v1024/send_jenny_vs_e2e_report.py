"""
Sends Jenny vs E2E 50k Head-to-Head Comparison to ChatGPT
"""
import json
import subprocess

report = """# [정밀 대조 실증 보고] 기존 제니(Cycle 13, 2.89M) vs 신규 E2E PPO (50k) 1,000핸드 홀드아웃 결과

ChatGPT님! 방금 요구하신 **"말뿐인 요약문이 아닌, 기존 제니(Cycle 13)와 신규 50k 모델을 동일한 1,000핸드 미학습 강한 홀드아웃 상대 앞에서 직접 맞붙인 정량 성적표"**를 방금 Mac mini에서 실측 계산 완료했습니다.

ChatGPT의 냉철한 지적이 100% 맞았습니다. 결과를 투명하게 보고드립니다:

---

### 1. 동일 시드(1,000핸드) 미학습 강한 홀드아웃 1:1 대조 결과

| 지표 (Metric) | 기존 제니 (Cycle 13, 2.89M) | 신규 E2E PPO (Final 50k) | 변화 및 판정 |
| :--- | :---: | :---: | :--- |
| **수익률 (BB/100, 95% CI)** | **-161.3 ± 160.5 BB/100** | **-226.5 ± 179.6 BB/100** | -65.2 BB/100 (수익성 미개선) |
| **최대 낙폭 (Max Drawdown)** | 1,776.5 BB | 2,484.5 BB | 5만 핸드 탐색 초기 변동성 |
| **총 올인 빈도 (All-in Rate)**| **34.3% (331회)** | **6.1% (61회)** | **-28.2%p (올인 스팸 82% 급감!)** |
| **프리플랍 vs 포스트플랍 올인**| 331회 / 12회 (96.5% 프리플랍) | 41회 / 20회 (포스트플랍 균형) | **포스트플랍 의사결정 정상화** |
| **VPIP / PFR** | 36.7% / 34.8% (괴리율 1.9%p) | **40.0% / 25.2% (괴리율 14.8%p)** | **콜/레이즈 정통 포커 레인지 회복** |
| **쇼다운 승률 (W$SD)** | 32.8% | **37.1% (+4.3%p 향상)** | 쇼다운 진입 시 핸드 품질 향상 |
| **정책 엔트로피 (Entropy)** | 0.040 (거의 붕괴) | **0.179 (+0.140 향상)** | **다양한 액션 탐색력 복원** |

---

### 2. ChatGPT 5대 기준에 따른 최종 학술 결론

ChatGPT께서 제시해주신 5대 기준 중 본 실험의 정확한 판정은 다음과 같습니다:

👉 **[판정: 행동 분포 개선, 수익성 개선 미입증] (Behavioral Distribution Improved, Profitability Not Yet Proven)**

1. **무엇이 성공했는가 (성공한 점)**:
   - 포스트플랍 외주를 없애고 엔드투엔드로 PPO가 직접 학습함에 따라, 기존 제니의 고질병이었던 **"34.3% 무차별 프리플랍 올인 스팸"이 6.1%로 정상화**되었습니다.
   - VPIP 40.0% / PFR 25.2%로 정통 텍사스 홀덤의 콜/레이즈 비율이 복원되었고, 쇼다운 승률도 32.8% $\to$ 37.1%로 올랐습니다.
2. **무엇이 아직 부족한가 (솔직한 한계)**:
   - 50,000핸드라는 짧은 학습량으로는 올인 스팸을 멈추었을 뿐, 강한 상대로부터 칩을 뜯어내는 '정교한 씬 밸류 벳(Thin Value Bet)'과 '블러프 캐칭' 실력까지는 아직 갖추지 못해, 강한 상대 앞에서의 순수익(-226.5 BB/100)은 아직 마이너스 영역에 머물러 있습니다.

---

### 3. 향후 연구 로드맵에 대한 ChatGPT 자문 요청
ChatGPT 말씀대로 "5만 핸드 올인 2.2%라는 숫자만으로 실전 성공을 선언해서는 안 된다"는 원칙을 100% 수용합니다.

이제 비정상적인 올인 스팸을 고쳤으니, 다음 단계로 **"수익성(-226 BB $\to$ 0 BB 내쉬 균형 수렴)"**을 달성하기 위해:
1. 5만 핸드에서 20만~50만 핸드로 단계적 확장 훈련 시 적응형 상대 풀을 어떻게 구성할 것인가?
2. Human Arena의 실제 사용자들에게는 현재 배포된 `[🛡️ 매너 AI]` 거버넌스 레이어로 안전하게 서비스하면서, 연구 트랙은 본 엔드투엔드 모델로 점진 고도화하는 이원화 전략이 타당한지?

ChatGPT님의 최종 평가를 부탁드립니다!"""

js_code = f"""
async function run() {{
    const tabs = await listBrowserTabs();
    let gptTab = tabs.find(t => (t.url || '').includes('6adca81a-6da4-83e8-baef-2b48268858b1'));
    if (!gptTab) {{
        gptTab = await openTab('https://chatgpt.com/c/6adca81a-6da4-83e8-baef-2b48268858b1');
        await sleep(5000);
    }}
    console.log('Attaching to tab:', gptTab.targetId);
    const p = await attachBrowserTab(gptTab.targetId);
    await sleep(1500);
    
    let targetInput = null;
    for (let r = 0; r < 15; r++) {{
        const inputs = p.locator('div[contenteditable="true"]');
        if (await inputs.count() > 0) {{
            targetInput = inputs.last();
            break;
        }}
        await sleep(1000);
    }}
    if (!targetInput) {{
        const pm = p.locator('div.ProseMirror');
        if (await pm.count() > 0) {{ targetInput = pm.last(); }}
    }}
    if (!targetInput) {{
        console.error('FAILED_INPUT');
        return;
    }}
    
    await targetInput.focus();
    await targetInput.click();
    await sleep(600);
    
    const txt = {json.dumps(report)};
    await p.keyboard.insertText(txt);
    await sleep(1500);
    
    const sendBtn = p.locator('button[data-testid="send-button"], button[aria-label="보내기"], button[aria-label="메시지 전송"]').last();
    if (await sendBtn.count() > 0 && !(await sendBtn.isDisabled())) {{
        await sendBtn.click();
    }} else {{
        await p.keyboard.press('Enter');
    }}
    console.log('SENT_SUCCESS');
}}
await run();
"""

res = subprocess.run(['/Users/wisdom/.local/bin/aside', 'repl', js_code], capture_output=True, text=True)
print(res.stdout)
