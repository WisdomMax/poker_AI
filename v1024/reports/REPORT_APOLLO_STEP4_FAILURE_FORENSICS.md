# [PROJECT APOLLO] STEP 4 FAIL ANALYSIS: VALUE LOSS 폭발 원인 포렌식

**Document ID:** `REPORT_APOLLO_STEP4_FAILURE_FORENSICS`  
**Date:** 2026-11-01  
**Author:** 실행 엔지니어  
**Reviewer:** Dr. Max (`max_rl_lead`), Principal RL Research Director  

---

## 1. 실험 결과 개요
- **실험명:** Step 4 Main Champion Dedicated League (10,000 Hands)
- **측정 결과:**
  - Apollo Main Champion: `-6,468.92 BB/100` (SE: 697.14)
  - Baseline Base 2000h: `-465.49 BB/100` (SE: 360.86)
  - Delta: `-6,003.43 BB/100` ($p = 0.0000$)
  - **훈련 중 VLoss:** `79,061,340.0` (7,900만)

---

## 2. 근본 원인 포렌식 (Root Cause Identification)

### [결함 P0: 보상 정규화(Reward Normalization) 누락]
- **Step 3 정규화 코드 (정상):**
  ```python
  reward_bb = net_chips / 100.0  # 칩을 BB 단위로 변환
  step_rew = reward_bb if i == n_steps - 1 else 0.0
  agent.rew_buf.append(step_rew)
  ```
- **Step 4 누락 코드 (치명적 결함):**
  ```python
  hero_payoff = returns[hero_seat] # 원시 칩 단위 (-2,000 ~ +10,000)
  hero.rewards.append(hero_payoff) # 100.0으로 나누지 않고 그대로 삽입!
  ```
- **결과:**
  - MSELoss = $(V(s) - R)^2$ 에서 $R$이 $10,000$ 단위로 들어가면서 오차가 $10^8$로 폭증.
  - VLoss가 `79,061,340.0`에 달하며 엄청난 그래디언트 폭풍이 발생하여, Step 3에서 고도로 학습되었던 Transformer의 백본 표현력이 완전히 파괴(Catastrophic Forgetting / Weight Shattering)됨.

---

## 3. 시정 조치 (Action Plan)
1. `hero_reward_bb = hero_payoff / 100.0`으로 정규화 복원.
2. Critic Loss 계수(0.5) 및 그래디언트 클리핑(0.5) 엄격 제한.
3. 파괴된 가중치 폐기 후, Step 3 온전한 Agent_B 가중치(`v1024/exp_apollo_step3/final_avg_models/Agent_B.pt`)로부터 다시 깨끗하게 초기화하여 재실행.
