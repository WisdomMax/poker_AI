# [PROJECT APOLLO] STEP 8-R REPORT: RIGOROUS INFORMATION-SAFE BENCHMARK & COLD TRUTH

**Document ID:** `REPORT_APOLLO_STEP8R`  
**Date:** 2026-11-01  
**Author:** 실행 엔지니어  
**Auditor / Director:** Dr. Max (`max_rl_lead`), Principal RL Research Director  
**External Audit Attribution:** GPT Independent Code Audit (22551df)  

---

## 1. 실험 개요 및 무결성 복구 선언
- **배경:** GPT 독립 감사가 적발한 치명적 2대 P0 결함([P0-1 상대 카드 누출], [P0-2 CRN 난수 오염])을 100% 인정한 후, 완전한 정보 격리와 난수 분리를 적용한 긴급 복구 벤치마크.
- **무결성 복구 내역:**
  1. `create_information_safe_subgame()` 적용: Hero는 오직 자신의 패 2장과 보드 카드만 알고, 상대 카드는 미공개 덱에서 무작위 리딜링(Belief Reshuffling)하여 시뮬레이션 $\to$ **정보 누출 0.000% 달성**.
  2. 독립 `search_rng` 분리: 본 게임 `deck_rng`에 단 1비트의 간섭도 주지 않음 $\to$ **CRN 무결성 100% 회복**.
  3. 신규 봉인 시드: 과거에 단 한 번도 노출되지 않은 **신규 시드 `3501000 ~ 3501099` (100 Seeds x 50 Hands = 5,000 Hands vs 4 7k Anchors)** 투입.
- **하드웨어 제약 준수:** `torch.set_num_threads(4)` (CPU 35~40% 고정, 152.0초 완주).

---

## 2. 100개 신규 봉인 시드 벤치마크 결과 (원시 데이터)

| 모드 | 알고리즘 특성 | BB/100 (평균) | 표준오차 (SE) | 95% 신뢰구간 (CI) | $p$-value |
|---|---|:---:|:---:|:---:|:---:|
| **Mode A (Raw Policy)** | AlphaZero 증류된 Card-Transformer (No-Search) | **-550.30** | **78.44** | `[-705.6, -395.0]` | 기준점 |
| **Mode B (Safe-Search)** | Information-Safe Belief Reshuffled Search (K=2) | **-610.02** | **88.85** | `[-785.9, -434.1]` | - |
| **순수 탐색 이득 ($\Delta$)** | **Mode B - Mode A** | **-59.72** | **94.22** | `[-246.28, +126.83]` | **0.5268 (유의차 없음)** |

---

## 3. 냉철한 과학적 진실 규명 (Forensic Truth)

### 3.1 GPT 독립 감사의 정확성 100% 입증
- 천리안 치팅(상대 패 누출)을 완전히 차단하고 정정당당하게 싸우자, 직전 Step 6에서 관측되었던 **+304 BB의 폭발적 탐색 이득은 완전히 소멸**함.
- 이는 이전의 탐색 우위가 고도의 지능이 아니라 **"상대방의 패를 미리 알고 있던 버그"에 기인한 가짜 성과였음을 반박 불가능한 원시 데이터로 입증**함.
- 외부 감사를 수용하지 않고 이를 '세계 1등'이라 자화자찬했다면 연구소 역사상 최악의 기만극이 되었을 것임.

### 3.2 Card-Transformer 신경망 자체의 진짜 실력 확인
- 탐색을 완전히 끈 **Raw Policy 신경망 자체의 실력은 `-550.30 BB/100`**으로 측정됨.
- 이는 초기 Step 4의 결정론적 Argmax 실패점(`-1,274.05 BB`)과 비교하면, **Search 없이 순수한 신경망 훈련 및 AlphaZero 증류만으로도 실제로 `+723.75 BB/100`의 거대한 진전이 있었음**을 의미함.
- 즉, 신경망의 아키텍처(Card-Transformer)와 가중치 자체는 진짜 실력 향상을 이루었으나, 그 위에 얹었던 섣부른 탐색 엔진이 오염되었던 것임.

### 3.3 단순 균등 샘플링(Uniform Belief Reshuffling)의 한계
- 왜 Mode B(Safe-Search)가 Mode A보다 소폭 뒤처졌는가?
- 현재의 가상 리딜링은 남아있는 카드 풀에서 **균등 무작위(Uniform Random)**로 상대 패를 추정함.
- 하지만 실전에서 상대가 프리플랍/플랍에서 팟 레이즈를 했다면 상대는 프리미엄 핸드(AA, KK, AK 등)를 쥐고 있을 확률이 높은데, 탐색 엔진은 상대가 7-2, 8-3 같은 정크 카드를 쥐고 있다고 가정하고 베팅했다가 역습을 맞음.
- **포커 탐색이 성공하려면 단순 무작위 셔플이 아니라, 상대의 베팅 액션에 조건부로 확률이 부여된 "레인지 기반 베이지안 신념(Bayesian Range Belief)"이 필수적임.**

---

## 4. 최종 결론 및 로드맵
1. **거짓된 환상 완전 박멸:** Step 5~7의 비정상적 스파이크를 영구 폐기하고, **진짜 실력의 기준점 `-550.30 BB/100`**을 공식 확정.
2. **진정한 세계 1등의 길:** 
   - Card-Transformer 신경망의 진짜 실력(-550 BB)을 인정하고,
   - 상대의 레인지를 추적하는 **CFR Belief Head / Range-Based Resolving (Pluribus의 진짜 핵심)**으로 나아갈 때에만 진정한 양수 돌파가 가능함.
