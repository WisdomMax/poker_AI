# REPORT 059-O: 팟 검사 복구, 음성 테스트 및 연구 기준 버전 고정 완료 보고서

- **작성 일시**: 2026-10-31
- **작성자**: Antigravity v1024
- **로컬 연구 기준 Git 커밋 SHA**: `8a901e2df1ce1b41929efb4ffa3868252f88d7d4` (브랜치: `baseline-059o-fixed`)
- **이전 작업 HEAD**: `6df5132aebbc807d069905db717790e03ea44d6f`
- **관련 지침**: `v1024/directives/DIRECTIVE_059O_FREEZE_RESEARCH_BASELINE.md`

---

## 1. 수정 전후 판정 및 정정 요약

1. **059-N 팟 검사 결함 정정**:
   - 기존 `directive059n_comparative_eval.py`의 `calculate_pots_exact(spents, folded)` 및 `eval_7card(holes[p], board)` 호출은 실제 함수 규격(3인자 및 1인자)과 맞지 않아 런타임 `TypeError`가 발생했으나, 넓은 `except Exception: pass`에 의해 삼켜져 팟 검사가 0회 실행된 상태였습니다.
   - 따라서 이전 보고서의 "전수 팟 검사 통과" 주장은 **공식 취소 및 정정**합니다.
   - 059-O에서 `calculate_pots_exact(spents, folded, scores)` 규격 수리 및 단독 생존자(Foldout) 손익 직접 계산 로직을 복구하고, 매 핸드 `audit_executed=True` 플래그 및 원시 상태 덤프를 강제했습니다.
2. **동등성(Parity) 주장 정정**:
   - 059-N/O의 관측 차이가 작거나 신뢰구간에 0이 포함되는 것은 동등성(Parity)의 증명이 아니며, 단순한 "우위 미입증(Inconclusive / Failure to Reject H0)"으로 정정합니다.

---

## 2. 지침 과제별 수행 결과 (TASK 1, 2, 3)

| 과제 | 세부 내용 | 판정 | 증거 파일 |
| :--- | :--- | :---: | :--- |
| **TASK 1** | 실제 059-N 팟 검사 규격 오류 수정, 제로섬 오배당 및 예외 주입 음성 테스트 통과, 25h 카나리 통과, 9개 군 4,500 테이블 핸드 재평가 완주 | **PASS** | `v1024/test_directive059o_integrity.py`<br>`v1024/reports/directive059o_comparative_evidence.json` |
| **TASK 2** | 소스·설정·모델·결과 SHA256 매니페스트 고정, 058~059 계보 연결표 작성, 로컬 `baseline-059o-fixed` 기준 커밋 생성 | **PASS** | `v1024/directive059o_manifest.json`<br>Git SHA: `8a901e2df1ce...` |
| **TASK 3** | 5개 AI 공동 학습 및 성장 검증 실험 설계서 1장 작성 (모델 구조, 4인+1상대+1휴식 Balanced Assignment, 고정 평가, 중단 조건) | **PASS** | `v1024/directives/PLAN_059P_COOPERATIVE_LEARNING.md` |

---

## 3. 제1부: TASK 1 팟 검사 복구 및 음성 테스트 실증

### 1) 무결성 및 음성 테스트 (`v1024/test_directive059o_integrity.py`)
```bash
# 실행 명령 및 종료 코드
PYTHONDONTWRITEBYTECODE=1 python3 -B v1024/test_directive059o_integrity.py
# Exit Code: 0 (ALL DIRECTIVE 059-O INTEGRITY & CANARY TESTS PASSED)
```

- **(Test 1) 제로섬 보존 오배당 주입 음성 테스트 -> PASS**:
  - 실제 평가 루프 경로의 `engine_rets`에 P0 +10칩, P1 -10칩의 제로섬 보존 오배당을 주입.
  - 독립 팟 감사 로직(`calculate_pots_exact`)에서 편차 10.0칩을 정확히 포착하고 즉각 `AssertionError: [AUDIT ABORT] Hand 0: Sidepot mismatch! diff=10.0`을 발생시키며 비영 종료됨을 입증.
- **(Test 2) 계산기 예외 주입 음성 테스트 -> PASS**:
  - 독립 계산기 내부 예외 발생 시 더 이상 넓은 except로 삼키지 않고 `ValueError: [INJECTED_CALCULATOR_EXCEPTION]`을 상위로 전파하여 즉시 중단됨을 입증.
- **(Test 3) CRN 브랜칭 및 카드 격리 불변성 -> PASS**:
  - Hand 1에서 Fold 조기 종료 vs Call 쇼다운 분기에도 불구하고 딜링된 5인의 홀 카드 10장이 100% 일치.
  - Hand 2에서 Hand 1의 분기와 무관하게 완전히 동일한 딜링 상태가 재현됨을 입증.
- **(Test 4) 25핸드 카나리 전수 검증 -> PASS**:
  - 9개 암 × 25핸드 = 총 225개 레코드 전원에서 `audit_executed=True` 확인.
  - 쇼다운 147회, 폴드아웃 78회, 최대 편차 0.0, 제로섬 오차 0.0 달성.

### 2) 공통 개발 시드 20개 (3006100~3006119) 4,500 테이블 핸드 재평가 결과

- **전체 핸드 수**: 4,500 테이블 핸드 (군당 500 테이블 핸드, 2,000 Hero 핸드)
- **전수 팟 검사 통과 현황**:
  - 총 쇼다운 팟 감사: 2,728 핸드
  - 총 폴드아웃 팟 감사: 1,772 핸드
  - `audit_executed`: **4,500 / 4,500 핸드 (100.0% 실행)**
  - `max_pot_discrepancy`: **`6.11e-13`** (부동소수점 오차 범위 내 완전 일치)
  - `max_zero_sum_error`: **`1.42e-14`** (완전 보존)

#### [표 1] 군별 관측 성능 및 표본 표준오차 (Within-Eval SE)
| 모델군 | Hero BB/100 | Opponent BB/100 | 표본 SE (BB/100) | 표본 SD (BB/100) | 팟 검사 편차 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base 2000h Initial** | -23.06 | +92.23 | 56.15 | 251.12 | 0.00 |
| **J_Static_s4008000** | -23.93 | +95.70 | 50.16 | 224.31 | 0.00 |
| **J_Dynamic_s4008000** | -60.76 | +243.05 | 52.65 | 235.48 | 0.00 |
| **J_Static_s4009000** | -46.45 | +185.78 | 53.43 | 238.94 | 0.00 |
| **J_Dynamic_s4009000** | -27.07 | +108.28 | 45.51 | 203.54 | 0.00 |
| **M_Static_s4010000** | -29.56 | +118.25 | 46.05 | 205.94 | 0.00 |
| **M_Dynamic_s4010000** | -24.70 | +98.80 | 55.57 | 248.54 | 0.00 |
| **M_Static_s4011000** | -57.92 | +231.68 | 57.14 | 255.55 | 0.00 |
| **M_Dynamic_s4011000** | -22.15 | +88.60 | 57.93 | 259.08 | 0.00 |

#### [표 2] 블록 대응 비교 (Paired Differences)
| 비교 항목 | $\Delta$ (Dynamic - Static) | 대응 SE | Base 대비 진척 (Static) | 대응 SE | Base 대비 진척 (Dynamic) | 대응 SE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **J-Cohort (s4008000)** | -36.84 | 24.15 | -0.87 | 30.22 | -37.71 | 22.83 |
| **J-Cohort (s4009000)** | +19.38 | 26.58 | -23.39 | 21.84 | -4.01 | 27.18 |
| **M-Cohort (s4010000)** | +4.86 | 42.98 | -6.51 | 29.23 | -1.64 | 47.40 |
| **M-Cohort (s4011000)** | +35.77 | 26.84 | -34.86 | 37.35 | +0.91 | 25.53 |

- **통계적 해석**:
  - 대응 SE가 24~43 BB/100 수준으로 표본 변동성이 크며, 모든 비교에서 95% 신뢰구간([$\Delta \pm 1.96 \times \text{SE}$])이 0을 포함합니다.
  - 시드 4008000에서는 Static이 Dynamic을 앞서고(-36.84), 시드 4011000에서는 Dynamic이 Static을 앞서는(+35.77) 등 시드 간 반전이 뚜렷하여 어떠한 리그의 우위나 동등성도 통계적으로 성립하지 않습니다. (NO-GO 유지 판정의 엄밀성 확인)

---

## 4. 제2부: TASK 2 로컬 연구 기준 버전 고정 및 계보 연결표

### 1) 연구 계보 및 엔진 변경점 연결표
| 지침/단계 | 적용 엔진 및 특징 | 주요 변경점 | 성능 증거 유효성 |
| :--- | :--- | :--- | :--- |
| **058** | `directive058_production_engine` | `+7.000` 인위적 보상 가산 결함 | **공식 무효 (증거 제외)** |
| **057** | 2,000h 이전 프로토타입 | 058 결함 환경의 영향을 받음 | **[미입증] 유지** |
| **059-J** | `directive059j_production_engine` | 매 10핸드 PPO 고정 업데이트, 시드 4008k/4009k | 탐색적 신호 (동등성 미증명) |
| **059-M** | `directive059m_production_engine` | 매 4핸드/버퍼 조건 PPO, 시드 4010k/4011k, deck_rng 사용 | 탐색적 신호 (동등성 미증명) |
| **059-N** | `directive059n_comparative_eval` | J vs M 공통 개발 시드 20개 비교 | 팟 검사 미실행 결함 발견 |
| **059-O** | `directive059o_comparative_eval` | **팟 검사 규격 수리 (`calculate_pots_exact`), 전수 팟 검사 강제, 제로섬 오배당 음성테스트 PASS** | **공식 연구 기준 버전 고정** |

### 2) 선별된 파일 SHA256 매니페스트 (`v1024/directive059o_manifest.json`)
```json
{
  "v1024/test_directive059o_integrity.py": {
    "sha256": "2ed8c0be238ac040e844578866742a94754caef011677f31a087b751cb14de8a",
    "size": 6295
  },
  "v1024/directive059o_comparative_eval.py": {
    "sha256": "5feac524d093f2303c17fcad62013f750b63a543570ed5d8bae8006b4ea22465",
    "size": 18059
  },
  "v1024/reports/directive059o_comparative_evidence.json": {
    "sha256": "993e17a256cf2791b22c75031d605aeb75b7ac34af270b4aa5e5cd9a690258b0",
    "size": 4004
  },
  "v1024/data/directive059o_comparative_raw.jsonl": {
    "sha256": "fb7fd80a94f9db50a86e160d6e3bcd3dd162d636adff789b7e121cc17b1f34d0",
    "size": 2060156
  },
  "v1024/directives/DIRECTIVE_059O_FREEZE_RESEARCH_BASELINE.md": {
    "sha256": "0be1cd0a9e390a02e6b1823240bb1dffac79047918100e7efd8273ae937612a5",
    "size": 5001
  },
  "v1024/directive059n_models_manifest.json": {
    "sha256": "13d66221649e44f62cd723e30a57a0daba1f69c0f41a5068f659837f39caccd0",
    "size": 11492
  }
}
```

- **Git 로컬 브랜치 커밋**:
  - 브랜치: `baseline-059o-fixed`
  - 커밋 SHA: `8a901e2df1ce1b41929efb4ffa3868252f88d7d4`
  - 원격 push 미실시 준수.

---

## 5. 제3부: TASK 3 다음 공동 학습 실험 설계서 요약 (`PLAN_059P_COOPERATIVE_LEARNING.md`)

- **문서 위치**: `v1024/directives/PLAN_059P_COOPERATIVE_LEARNING.md`
- **핵심 요약**:
  1. **5인 모델 구조 및 착석 규칙**: 5개 AI가 동시에 착석하지 않고, 매 핸드마다 4인 학습 영웅(`active_heroes`), 1인 휴식 영웅(`rest_hero`), 1인 동결 상대(`opp_seat`)가 직교 균등 분배(Balanced Assignment)로 교대 착석.
  2. **학습 엔진 및 초기 모델 고정**: 본 059-O에서 검증된 수리된 팟 감사 엔진 단일화 및 Base 2,000h 고정 모델 사용.
  3. **독립 성장 지표**: 에이전트별 BB/100, 엔트로피, PPO 클립 비율 측정.
  4. **공통 개발 평가**: 학습 후 신규 시드 20개에서 Base 2k, Opponent 7k, 자가 과거 스냅샷(250h, 500h)을 상대로 실질적 실력 향상 검증.
  5. **중단 조건**: 제로섬 오차 > 1e-4, 팟 불일치 > 1e-4, 카드 중복 시 즉시 중단.
