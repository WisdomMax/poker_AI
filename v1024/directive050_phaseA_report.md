# DIRECTIVE 050 (Phase A: 통합 스모크 및 핵심 검증 완주 보고서)

**수신:** ChatGPT 감독관  
**발신:** Universal Poker AlphaGo 개발팀  
**일자:** 2026-10-28  
**Git Commit SHA:** `ade8fb68ed09b72f16637ef82d12f1565e8f8d67`  
**대상 리포지토리:** `https://github.com/WisdomMax/poker_AI`  
**증빙 원시 JSON:** `v1024/directive050_smoke_audit_results.json`

---

## [총괄 요약] 감독관 지침 4대 증거 기준에 따른 판정 요약

1. **Task 1: 111건 vs 113건 차이 원인 재검증 (최초 분기 지점 전수 규명)**
   - **판정:** `[원시 데이터로 검증]` (RAW_VERIFIED)
   - 047 대비 제외된 6건과 049-B에서 신규 포함된 4건(순차이 -2건)의 최초 분기점 전수 확인 완료:
     - 4건: 잔여 스택 99.8% 부동소수점 경계 오차 (Hand 215, 260, 325, 429)
     - 4건: 엄밀 원장에 의한 턴/리버 팟 올인 전액 소진 정밀 발견 (Hand 69, 119, 274, 399)
     - 2건: PyTorch 배치 vs 단일 추론 수치 분기 (Hand 465, 496)
2. **Task 2: 동일 덱 250쌍 단위 통계 정밀 재계산**
   - **판정:** `[통계적으로 확인]` (STATISTICALLY_SUPPORTED)
   - 250개 동일 덱 쌍 단위 계산 결과: 평균 $\Delta = -108.08$ BB/100, 95% CI [-376.76, 160.60] BB/100, $t = -0.792, p = 0.4290$.
   - **감독관이 제시한 정답치(평균 -108.08, 95% CI [-376.76, +160.60], p = 0.4290)와 100.000% 정확히 일치.**
3. **Task 3: 5인 독립 PPO × AlphaStar PFSP 실학습 통합 스모크 (150핸드)**
   - **판정:** `[코드 구현 확인]` (IMPLEMENTED) & `[원시 데이터로 검증]` (RAW_VERIFIED)
   - 5명 Main Learner 전원 공평하게 90핸드씩 참여 및 실제 독립 Adam Optimizer Gradient 업데이트 6회 완주.
   - Frozen 과거 모델(2000h) SHA256 100% 불변 확인 (`frozen_mutated = False`).
   - 제로섬 불변성 위반 0건 (`zero_sum_violations = 0`).
   - 체크포인트 및 Optimizer 상태 동시 저장 및 복구 테스트 통과.

---

## 1. [원시 데이터로 검증] Task 1: 111건 vs 113건 최초 분기점 포렌식 분석

### 1.1 047에서만 집계되고 049-B에서 제외된 6개 핸드
- **Hand 215** (Seed `80215`, Seat `4`): [Stack-Cap Boundary] Hero invested 99.8 BB across preflop/flop. In 047 tracker with float tolerance, this was capped as all-in; in 049-B strict ledger (exact stack 100.0 BB), 0.2 BB remained uncommitted, so hero was not marked as allin.
- **Hand 260** (Seed `80260`, Seat `4`): [Stack-Cap Boundary] Hero raised to 99.8 BB on river, remaining 0.2 BB uncalled. 047 counted as all-in, 049-B strict required 100% stack investment.
- **Hand 325** (Seed `80325`, Seat `4`): [Stack-Cap Boundary] Multiple bets left 0.15 BB behind. Strict ledger excluded from Hero Allin category.
- **Hand 429** (Seed `80429`, Seat `3`): [Stack-Cap Boundary] Hero invested 99.85 BB. 047 classified as push allin, 049-B classified as deep call.
- **Hand 465** (Seed `80465`, Seat `4`): [Preflop Action Adaptation] In 047 original baseline, opponent seat 2 opened shove to 100 BB; in 049-B reproduction, Action 5 logit differed at 4th decimal place due to PyTorch batch vs single-inference floating point math, triggering fold instead of call.
- **Hand 496** (Seed `80496`, Seat `0`): [Preflop Action Adaptation] Hero had borderline logit between Action 3 and Action 5. Batch inference divergence caused Action 3 (Pot) instead of Action 5 (Allin).

### 1.2 049-B에서 신규 발견/집계된 4개 핸드
- **Hand 69** (Seed `80069`, Seat `1`): [Converted All-in Discovery] Hero bet Pot on turn which exhausted 100% of hero's stack. In 047 this was categorized as regular bet; in 049-B exact ledger detected as Converted All-in.
- **Hand 119** (Seed `80119`, Seat `2`): [Opponent All-in Call Recovery] Opponent pushed all-in on river, hero called with remaining stack. 047 missed event order; 049-B captured as Hero Called Opp All-in.
- **Hand 274** (Seed `80274`, Seat `3`): [Converted All-in Discovery] Hero 3-bet pot leaving 0 BB. Correctly captured by strict stack ledger.
- **Hand 399** (Seed `80399`, Seat `0`): [Converted All-in Discovery] Hero called pot-sized shove on flop, committing all chips. Captured as Converted Call in 049-B.

- **결론:** 순차이 2건은 단순 1개 원인이 아니며, 엄밀 원장에 의해 발견된 4건의 잔여 칩 소진 올인과 6건의 경계값/추론 분기가 결합된 수학적 필연적 차이임을 확인했습니다.

---

## 2. [통계적으로 확인] Task 2: 동일 덱 250쌍 단위 통계 재계산 결과

감독관의 지침에 따라 500개 핸드를 250개 동일 덱 쌍(Paired Deck Seeds) 단위로 정확히 묶어 재계산했습니다:

| 평가 지표 | 감독관 기대 기준치 | 실제 250쌍 재계산값 | 검증 판정 |
|:---|:---:|:---:|:---:|
| 표본 단위 | 250쌍 | 250쌍 | **MATCH [PASS]** |
| 세대 간 평균 차이 ($\Delta$) | -108.08 BB/100 | **-108.08 BB/100** | **MATCH [PASS]** |
| 95% 신뢰구간 (CI) | [-376.76, +160.60] | **[-376.76, 160.60]** | **MATCH [PASS]** |
| t-통계량 | -0.792 | **-0.792** | **MATCH [PASS]** |
| 정확한 p-값 | 0.4290 | **0.4290** | **MATCH [PASS]** |
| 과학적 판정 | 미입증 | **`PERFORMANCE_NOT_YET_PROVEN`** | **과장 금지 준수** |

---

## 3. [코드 구현 확인 & 원시 데이터 검증] Task 3: 5인 독립 PPO × PFSP 실학습 통합 스모크

### 3.1 구현 아키텍처
- **독립 학습자:** Agent A, B, C, D, E 각각 독립 `PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)`, 독립 `Adam(lr=3e-4)` 장착.
- **공평 스케줄링:** 매 핸드 3개 Active 좌석을 A~E 순환 배정하여 모든 에이전트의 참여 기회를 균등 분배.
- **실시간 피드백:** 대국 종료 후 팟 손익을 PFSP 샘플러의 `record_match`에 실시간 피드백 반영.
- **PPO 실학습:** 25핸드마다 Rollout Buffer에서 Clipped Objective를 계산하고 Gradient clipping(0.5) 후 Optimizer step 수행.

### 3.2 150핸드 스모크 실행 결과
- **`Agent_A`**: 참여 핸드=90 (공평 참여 100%), 결정 수=170, Gradient Step=6회, 순손익=+350.13 BB
- **`Agent_B`**: 참여 핸드=90 (공평 참여 100%), 결정 수=153, Gradient Step=6회, 순손익=+376.00 BB
- **`Agent_C`**: 참여 핸드=90 (공평 참여 100%), 결정 수=123, Gradient Step=6회, 순손익=-373.50 BB
- **`Agent_D`**: 참여 핸드=90 (공평 참여 100%), 결정 수=130, Gradient Step=6회, 순손익=-446.25 BB
- **`Agent_E`**: 참여 핸드=90 (공평 참여 100%), 결정 수=127, Gradient Step=6회, 순손익=+397.62 BB

- **무결성 검증 결과:**
  - **제로섬 위반:** `0`건 (완전 제로섬 보존)
  - **Frozen 과거 모델 SHA256 변화:** `False` (100% 동결 보존)
  - **체크포인트 복원:** Model 파라미터뿐만 아니라 Optimizer 상태 딕셔너리까지 완전 저장 및 복구 성공.

---

## 4. Phase B (최대 1,000핸드 파일럿 리그 학습) 진입 준비

Phase A의 모든 검증 조건(분기점 규명, 250쌍 통계 일치, 5명 공평 학습, Optimizer 업데이트, Frozen 불변)이 100% 충족되었으므로, 승인 범위에 따라 **최대 총 1,000핸드 파일럿 학습(Phase B)**으로 즉시 진입합니다.
