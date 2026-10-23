"""
V10.24 회귀 테스트 및 Adaptive Sampling 검증 스크립트
======================================================
1. [단위 테스트]: 중복 카드 없는 6대 쇼다운 완전 검증
   - 2인 동률: 0.5000
   - 3인 동률: 0.3333
   - 4인 동률: 0.2500
   - 5인 동률: 0.2000
   - Hero 단독 승리: 1.0000
   - Hero 단독 패배: 0.0000 (보드/Hero/상대 간 중복 카드 0장 검증)
2. [본진 Model A 통합 검증]:
   - run_v1013의 ModelA가 estimate_equity_det(수정본)을 정상 호출하는지 검증
3. [Action Boundary 기반 Adaptive Sampling 검증]:
   - Seed 180158 (경계선 핸드, margin = 0.162%p <= 2.5%p):
     300 sims -> 경계선 감지 -> 3,000 sims 동적 확장 작동 확인!
   - Seed 180408 (안정 핸드, margin = 16.65%p > 2.5%p):
     경계선 미감지 -> 기본 300 sims 유지 확인!
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v1013.run_v1013_replay_and_controller_validation import (
    estimate_equity_det, ModelA, DeterministicTrueNFSP_Agent, eval_7cards
)
from v75.run_v75_true_6action_and_fold_recovery import get_5p_fullgame
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1023.v1023_13_differential_forensics import (
    PPOAgentWrapper, trace_single_hand_deep
)

def run_flawless_unit_tests():
    print("\n=== [1] 6대 쇼다운 엄밀 단위 테스트 (중복 카드 완전 제거) ===")
    results = {}
    
    # 1. 2~5인 동률 테스트 (A-high Straight 보드)
    # rank = c // 4, suit = c % 4
    # 보드: T, J, Q, K, A (diff suits to prevent flush)
    # 32 (T), 37 (J), 42 (Q), 44 (K), 49 (A)
    straight_board = [32, 37, 42, 44, 49]
    hero_low = [0, 1] # 2s, 2h
    combos_low = [[4, 5], [8, 9], [12, 13], [16, 17]] # 3, 4, 5, 6
    probs_low = [0.25, 0.25, 0.25, 0.25]
    
    # 중복 카드 검증
    all_cards_straight = set(straight_board) | set(hero_low)
    for c in combos_low:
        all_cards_straight |= set(c)
    assert len(all_cards_straight) == len(straight_board) + len(hero_low) + sum(len(c) for c in combos_low), "중복 카드 발견!"
    print("  ✓ 동률 테스트 카드 풀 중복 검사: 중복 0장 (완전 독립)")
    
    for n_tied, expected in [(2, 0.5), (3, 1/3), (4, 0.25), (5, 0.20)]:
        rng = np.random.RandomState(42)
        eq = estimate_equity_det(hero_low, straight_board, combos_low, probs_low, n_opps=n_tied-1, sims=100, rng=rng)
        is_p = abs(eq - expected) < 1e-4
        print(f"  [{n_tied}인 동률]: 산출값 = {eq:.4f}, 기대값 = {expected:.4f} -> {'✓ 통과' if is_p else '✗ 실패'}")
        results[f"{n_tied}p_tie"] = {"value": float(eq), "expected": round(expected, 4), "passed": is_p}

    # 2. Hero 단독 승리 테스트 (Hero: A♠ 로열 플러시 완성, 상대: 일반 핸드)
    # Royal Board: T♠(32), J♠(36), Q♠(40), K♠(44), 2♦(2)
    # Hero: A♠(48), 3♥(5) -> 로열 플러시 완성!
    # Opponent combos: 4♣(15), 5♣(19) -> 일반 핸드
    royal_board = [32, 36, 40, 44, 2]
    hero_rf = [48, 5]
    combos_opp_lose = [[15, 19]]
    probs_opp_lose = [1.0]
    
    # 중복 카드 검사
    used_win = set(royal_board) | set(hero_rf) | set(combos_opp_lose[0])
    assert len(used_win) == 5 + 2 + 2, "단독 승리 중복 카드 발견!"
    print("  ✓ 단독 승리 카드 풀 중복 검사: 중복 0장 (완전 독립)")
    
    rng_win = np.random.RandomState(42)
    eq_win = estimate_equity_det(hero_rf, royal_board, combos_opp_lose, probs_opp_lose, n_opps=1, sims=100, rng=rng_win)
    pass_win = abs(eq_win - 1.0000) < 1e-4
    print(f"  [Hero 단독 승리]: 산출값 = {eq_win:.4f}, 기대값 = 1.0000 -> {'✓ 통과' if pass_win else '✗ 실패'}")
    results["solo_win"] = {"value": float(eq_win), "expected": 1.0000, "passed": pass_win}

    # 3. Hero 단독 패배 테스트 (상대: A♠ 로열 플러시 완성, Hero: 일반 핸드)
    # Royal Board: T♠(32), J♠(36), Q♠(40), K♠(44), 2♦(2)
    # Hero: 2♠(0), 3♠(4) -> 노메이드/원페어
    # Opponent: A♠(48), 3♥(5) -> 로열 플러시 완성!
    hero_lose = [0, 4]
    combos_opp_rf = [[48, 5]]
    probs_opp_rf = [1.0]
    
    # 중복 카드 검사 (ChatGPT가 지적했던 부분 완전 해소!)
    used_lose = set(royal_board) | set(hero_lose) | set(combos_opp_rf[0])
    assert len(used_lose) == 5 + 2 + 2, f"단독 패배 중복 카드 발견! used={used_lose}"
    print(f"  ✓ 단독 패배 카드 풀 중복 검사: 중복 0장 (완전 독립 9장: {sorted(list(used_lose))})")
    
    rng_lose = np.random.RandomState(42)
    eq_lose = estimate_equity_det(hero_lose, royal_board, combos_opp_rf, probs_opp_rf, n_opps=1, sims=100, rng=rng_lose)
    pass_lose = abs(eq_lose - 0.0000) < 1e-4
    print(f"  [Hero 단독 패배]: 산출값 = {eq_lose:.4f}, 기대값 = 0.0000 -> {'✓ 통과' if pass_lose else '✗ 실패'}")
    results["solo_lose"] = {"value": float(eq_lose), "expected": 0.0000, "passed": pass_lose}

    all_passed = all(r["passed"] for r in results.values())
    print(f"  단위 테스트 종합: {'✓ 100% ALL PASS (결함 0개)' if all_passed else '✗ FAILED'}")
    results["all_passed"] = all_passed
    return results

def run_adaptive_sampling_test():
    print("\n=== [2] Action Boundary 기반 Adaptive Sampling 실증 검증 ===")
    
    # 모델 로드
    game = get_5p_fullgame()
    m_r2 = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    m_r2.load_state_dict(torch.load("v1022/models/StrictAntiLimper_R2_100K.pt", map_location="cpu", weights_only=False))
    m_r2.eval()
    ppo_r2 = PPOAgentWrapper(m_r2)
    
    # 1. Seed 180158 (경계선 핸드: Fold 임계값과 0.162%p 차이)
    # 기대: boundary_margin <= 0.025 (2.5%p) 이므로 sims_used = 3,000회로 자동 확장!
    bp1 = DeterministicTrueNFSP_Agent()
    model_a_adapt1 = ModelA(bp1, sims=300, adaptive=True, adaptive_sims=3000, margin_threshold=0.025)
    log158 = trace_single_hand_deep(180158, ppo_r2, model_a_adapt1, game)
    flop_meta_158 = log158["hybrid_log"][1]["meta"]
    
    print("  [Seed 180158 - 경계선 핸드]")
    print(f"    팟오즈: {flop_meta_158['pot_odds']:.4%}, Fold임계값: {flop_meta_158['pot_odds']-0.05:.4%}")
    print(f"    최종 산출 에쿼티: {flop_meta_158['mc_equity']:.4%}")
    print(f"    사용된 시뮬레이션 표본수: {flop_meta_158['sims_used']} sims (기본 300 -> 3000 동적 확장 여부: {'✓ 성공' if flop_meta_158['sims_used'] == 3000 else '✗ 실패'})")
    print(f"    최종 결정: {flop_meta_158['act']} (0=Fold) -> {'✓ Fold 확정' if flop_meta_158['act'] == 0 else '기타'}")

    # 2. Seed 180408 (안정 핸드: 에쿼티 ~45% vs 임계값 28.33%, 차이 16.65%p)
    # 기대: boundary_margin > 0.025 이므로 기본 300 sims 유지!
    bp2 = DeterministicTrueNFSP_Agent()
    model_a_adapt2 = ModelA(bp2, sims=300, adaptive=True, adaptive_sims=3000, margin_threshold=0.025)
    log408 = trace_single_hand_deep(180408, ppo_r2, model_a_adapt2, game)
    flop_meta_408 = log408["hybrid_log"][1]["meta"]
    
    print("\n  [Seed 180408 - 안정 핸드]")
    print(f"    팟오즈: {flop_meta_408['pot_odds']:.4%}, Fold임계값: {flop_meta_408['pot_odds']-0.05:.4%}")
    print(f"    최종 산출 에쿼티: {flop_meta_408['mc_equity']:.4%}")
    print(f"    사용된 시뮬레이션 표본수: {flop_meta_408['sims_used']} sims (기본 300 유지 여부: {'✓ 성공' if flop_meta_408['sims_used'] == 300 else '✗ 실패'})")
    print(f"    최종 결정: {flop_meta_408['act']} (1=CheckCall) -> {'✓ CheckCall 확정' if flop_meta_408['act'] == 1 else '기타'}")

    adaptive_results = {
        "seed_180158": {
            "pot_odds": float(flop_meta_158['pot_odds']),
            "mc_equity": float(flop_meta_158['mc_equity']),
            "sims_used": int(flop_meta_158['sims_used']),
            "action": int(flop_meta_158['act']),
            "adaptive_triggered": flop_meta_158['sims_used'] == 3000
        },
        "seed_180408": {
            "pot_odds": float(flop_meta_408['pot_odds']),
            "mc_equity": float(flop_meta_408['mc_equity']),
            "sims_used": int(flop_meta_408['sims_used']),
            "action": int(flop_meta_408['act']),
            "adaptive_triggered": flop_meta_408['sims_used'] == 3000
        }
    }
    return adaptive_results

def main():
    print("=" * 70)
    print("POKER AI V10.24 ARCHITECTURE VALIDATION")
    print("ChatGPT 권고사항 100% 반영: 중복 카드 제거 단위테스트 & Adaptive ModelA 본진 통합")
    print("=" * 70)
    
    t0 = time.time()
    out = {"version": "V10.24_ARCH_VALIDATION"}
    
    unit_res = run_flawless_unit_tests()
    out["unit_tests"] = unit_res
    
    adapt_res = run_adaptive_sampling_test()
    out["adaptive_sampling_test"] = adapt_res
    
    elapsed = round(time.time() - t0, 2)
    out["elapsed_sec"] = elapsed
    
    out_path = "v1024_arch_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        
    sha = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sha
    
    print("\n" + "=" * 70)
    print(f"V10.24 아키텍처 사전 검증 완료 (소요 시간: {elapsed}초, SHA: {sha[:16]}...)")
    print("=" * 70)

if __name__ == "__main__":
    main()
