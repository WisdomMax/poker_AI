# [PROJECT APOLLO] STEP 4 REPORT: 1 VS 4 LEAGUE EVALUATION & GAME-THEORETIC FORENSICS

**Document ID:** `REPORT_APOLLO_STEP4`  
**Date:** 2026-11-01  
**Author:** 실행 엔지니어  
**Auditor / Director:** Dr. Max (`max_rl_lead`), Principal RL Research Director  

---

## 1. 실험 개요 및 무결성 검증
- **실험명:** Step 4 Dedicated Main Champion League (10,000 Hands)
- **하드웨어:** CPU 4스레드 엄격 통제 (35~40% CPU, 73.4초 완주, 136.3 HPS)
- **무결성 감사:**
  - 팟 계산 오차: 0.0 (10,000 / 10,000 Hands PASS)
  - 칩 제로섬 오차: $0.0$ (최대 오차 $< 10^{-12}$)
  - Value Loss: `2,868.27` (이전 스케일 왜곡 버그 수정 후 완벽 정상 수렴)

---

## 2. 100개 독립 봉인 시드 벤치마크 결과 (100 Seeds x 50 Hands = 5,000 Hands)
**대결 구도: 1 Hero vs 4 Anchor Opponents (Agent_A, C, D, E from 7k)**

| 지표 | Apollo Main Champion | Baseline (Base 2000h) | Delta | 95% CI | $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|
| **단독 수익률 (BB/100)** | **-1,274.05** | **-93.10** | **-1,180.95** | `[-1429.98, -931.92]` | **0.0000** |
| **표준오차 (SE)** | 137.66 | 72.17 | 125.77 | - | - |
| **합격 기준 ($\ge +100$ BB)** | **미달 (FAIL)** | - | - | - | **가설 완전 기각** |

---

## 3. 심층 게임 이론 및 강화학습 포렌식 (Root Causes)

### 3.1 진실의 발견: Agent_B의 '온실 착취' 실체 폭로
- Step 3에서 Agent_B가 기록했던 `+589.38 BB/100`의 실체가 Step 4에서 명확히 드러남:
  - Agent_B는 7k 앵커를 압도하는 실력을 가진 것이 아니라, **함께 테이블에 앉았던 미숙한 형제 에이전트(Agent_C -239 BB, Agent_E -430 BB)를 털어먹으며 점수를 불렸던 것(In-group Bullying/Cannibalism)**이었음.
  - 약한 먹잇감이 사라지고 **4명의 단단한 7k 앵커들로만 둘러싸인 테이블(1 vs 4)**에 홀로 들어가자, Step 3의 가중치로는 이들의 협공과 레이즈 압박을 전혀 버텨내지 못함.

### 3.2 평가 프로토콜 결함: 순수 탐욕 결정론적(Greedy Argmax) 정책의 파멸
- 평가 코드에서 Transformer가 `argmax(probs)`로 행동함.
- **불완전 정보 포커 게임에서 결정론적 정책(Pure Strategy)은 치명적인 자살 행위**:
  - 상대 봇들에게 핸드 레인지가 100% 투명하게 읽혀 쉽게 블러프 캐치당하거나 역착취(Exploited)됨.
  - GTO의 핵심인 혼합 전략(Mixed Strategy, 확률적 샘플링)을 배제한 대가로 손실이 극대화됨.

### 3.3 1 vs 4 비대칭 대진의 멀티웨이(Multiway) 역학
- 5인 포커 테이블에서 4명이 서로 오랫동안 맞춰온 7k 균형 상대이고 1명만 외부 침입자인 경우, Hero는 매 팟마다 2~3명의 상대와 멀티웨이 팟을 치러야 하는 극도로 불리한 포지션에 놓임.

---

## 4. 맥스 박사 승인 요청 및 다음 단계 혁신

우리는 속임수나 환상이 아니라 **원시 데이터가 폭로한 차가운 진실**을 마주했습니다:
1. **Agent_B 단독 가중치는 완성된 모델이 아니다.**
2. 세계 1등을 달성하려면 단순 PPO Self-Play나 단일 봇 미세조정이 아니라, **진짜 다변화된 리그(Population-Based Training / CFR-Search)**가 필요하다.
3. Pluribus와 ReBeL이 세계를 제패한 진짜 비밀: **오프라인 거대 신경망 + 온라인 서브게임 풀이(Subgame Solving / Depth-Limited Search)** 결합.
