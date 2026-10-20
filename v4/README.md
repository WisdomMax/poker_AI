# POKER AI V4.0 - RESEARCH ENVIRONMENT & DEPENDENCIES

## 1. 공식 프레임워크 및 의존성
- **Google DeepMind OpenSpiel**: `open_spiel == 1.6.3`
  - GitHub: https://github.com/google-deepmind/open_spiel
  - Core C++ Engine with Python 3.9 bindings (`pyspiel`)
- **수치 연산 및 과학 라이브러리**:
  - `numpy == 2.0.2`
  - `scipy == 1.13.1`
  - `torch == 2.8.0`
- **실행 환경**:
  - OS: macOS (Darwin 24.3.0, arm64 Apple Silicon)
  - Python: 3.9.6

## 2. 모듈 구성
- `openspiel_adapter.py`: 자체 V3.6 정책과 OpenSpiel Policy 객체 간의 변환기
- `phase1_engine_comparison.py`: 자체 Kuhn 엔진 vs 공식 OpenSpiel 게임 트리/보상 1:1 비교
- `phase2_cross_validation.py`: NashConv, Exploitability 3방향 교차 검증
- `phase2b_exact_oracle.py`: 해석적 Exact Best Response Oracle 및 Regret >= 0 수학적 증명
- `phase3_safe_bayesian.py`: 불확실성 반영 Safe Bayesian Exploiter
- `phase4_self_play.py`: OpenSpiel 기반 Self-Play 정책 학습 및 모델 체크포인트
