#!/usr/bin/env python3
"""
DIRECTIVE 043 Pre-requisite Test Suite:
PPO Input Normalization Integrity & Multi-Agent Zero-Sum Regression Test
Verifies:
1. Normalization transforms betting features (981:1417) from [0..10000] to [0..1].
2. Identical weights produce exact old_log_prob == new_log_prob and ratio == 1.000000.
3. Raw state vs Normalized state mismatch detection (proves why the bug occurred).
4. Action masking strictly zeroes illegal action probabilities.
5. 5-player multi-agent zero-sum invariant: sum(returns) == 0.0 in every hand.
"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79, BET_START_INDEX, BET_END_INDEX
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

def test_normalization_betting_features():
    """정규화 함수가 981:1417 슬라이스를 정확히 stack_size로 스케일링하는지 검증"""
    dummy_inf = np.zeros(1417, dtype=np.float32)
    dummy_inf[BET_START_INDEX] = 5000.0   # $5,000 bet
    dummy_inf[BET_END_INDEX - 1] = 10000.0 # $10,000 all-in
    
    norm = normalize_inf_state_v79(dummy_inf, stack_size=10000.0)
    assert norm[BET_START_INDEX] == 0.5, f"Expected 0.5, got {norm[BET_START_INDEX]}"
    assert norm[BET_END_INDEX - 1] == 1.0, f"Expected 1.0, got {norm[BET_END_INDEX - 1]}"
    assert dummy_inf[BET_START_INDEX] == 5000.0, "Original raw state must remain unmodified!"
    print("[PASS] test_normalization_betting_features: 베팅 피처(981~1417) 0..1 스케일링 무결성 확인.")

def test_raw_vs_normalized_mismatch_detection():
    """과거 버그 증명: raw state로 뽑은 log_prob과 normalized state로 계산한 log_prob은 완전히 왜곡됨"""
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    model.eval()
    
    dummy_inf = np.zeros(1417, dtype=np.float32)
    dummy_inf[BET_START_INDEX:BET_START_INDEX + 10] = 5000.0
    norm_inf = normalize_inf_state_v79(dummy_inf)
    
    t_raw = torch.from_numpy(dummy_inf).unsqueeze(0)
    t_norm = torch.from_numpy(norm_inf).unsqueeze(0)
    
    with torch.no_grad():
        logits_raw, _ = model(t_raw)
        logits_norm, _ = model(t_norm)
        
    diff = torch.norm(logits_raw - logits_norm).item()
    assert diff > 1e-4, f"Raw and Normalized inputs MUST produce different logits! diff={diff}"
    print(f"[PASS] test_raw_vs_normalized_mismatch_detection: 입력 불일치 시 로짓 왜곡차={diff:.4f} 검출 확인 (과거 결함 규명).")

def test_identical_weights_ratio_is_exact_one():
    """핵심 무결성: 행동 선택 시와 업데이트 시 모두 norm_inf를 사용하면 ratio가 정확히 1.000000이어야 함"""
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    
    dummy_inf = np.zeros(1417, dtype=np.float32)
    dummy_inf[0] = 1.0
    dummy_inf[BET_START_INDEX] = 2000.0
    norm_inf = normalize_inf_state_v79(dummy_inf)
    
    legal_actions = [0, 1, 2, 3]
    
    # 1. 롤아웃 시: 반드시 norm_inf로 행동 및 log_prob 산출
    model.eval()
    t_state_1 = torch.from_numpy(norm_inf).unsqueeze(0)
    with torch.no_grad():
        logits_1, _ = model(t_state_1)
        mask = torch.full_like(logits_1, -1e9)
        for a in legal_actions: mask[0, a] = 0.0
        masked_1 = logits_1 + mask
        probs_1 = F.softmax(masked_1, dim=-1)
        dist_1 = torch.distributions.Categorical(probs_1)
        action = dist_1.sample()
        old_log_prob = dist_1.log_prob(action).item()
        
    # 2. 업데이트 시: 동일 가중치, 동일 norm_inf로 new_log_prob 산출
    model.train()
    t_state_2 = torch.from_numpy(norm_inf).unsqueeze(0)
    logits_2, _ = model(t_state_2)
    masked_2 = logits_2 + mask
    probs_2 = F.softmax(masked_2, dim=-1)
    dist_2 = torch.distributions.Categorical(probs_2)
    new_log_prob = dist_2.log_prob(action).item()
    
    ratio = np.exp(new_log_prob - old_log_prob)
    abs_err = abs(ratio - 1.0)
    
    assert abs_err < 1e-5, f"Importance ratio must be 1.0 under identical weights! Got ratio={ratio}, err={abs_err}"
    print(f"[PASS] test_identical_weights_ratio_is_exact_one: 동일 가중치 ratio={ratio:.6f} (오차 {abs_err:.2e}) 완벽 무결성 입증!")

def test_action_masking_integrity():
    """Action Masking 검증: 금지된 액션(예: [4, 5])의 확률은 정확히 0.000000이어야 함"""
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    dummy_inf = np.random.randn(1417).astype(np.float32)
    norm_inf = normalize_inf_state_v79(dummy_inf)
    
    legal_actions = [0, 1] # Fold, Call only
    t_state = torch.from_numpy(norm_inf).unsqueeze(0)
    logits, _ = model(t_state)
    mask = torch.full_like(logits, -1e9)
    for a in legal_actions: mask[0, a] = 0.0
    probs = F.softmax(logits + mask, dim=-1).squeeze(0).detach().numpy()
    
    for a in [2, 3, 4, 5]:
        assert probs[a] < 1e-7, f"Illegal action {a} must have 0 probability! Got {probs[a]}"
    assert abs(sum(probs[legal_actions]) - 1.0) < 1e-6, "Legal action sum must equal 1.0"
    print("[PASS] test_action_masking_integrity: 불법 액션 확률 0.000000 차단 확인.")

def test_multi_agent_zero_sum_invariant():
    """5인 동시 게임에서 매 핸드 전체 플레이어 손익의 합이 엄밀히 0.0인지 검증"""
    game = get_5p_fullgame()
    for h in range(50):
        rng = np.random.RandomState(43000 + h)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
            cp = st.current_player()
            legal = True6ActionAdapter.get_legal_6actions(st, cp)
            # Pick arbitrary legal action
            act_6 = rng.choice(legal)
            nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
            st.apply_action(nat_act)
            
        returns = st.returns()
        assert len(returns) == 5
        tot_ret = sum(returns)
        assert abs(tot_ret) < 1e-4, f"Zero-sum invariant broken! hand {h}, sum={tot_ret}"
    print("[PASS] test_multi_agent_zero_sum_invariant: 5인 전원 총손익 합 == 0.0 제로섬 50핸드 전수 검증 완료.")

def run_all_tests():
    print("=" * 70)
    print("RUNNING DIRECTIVE 043 PPO INPUT INTEGRITY & ZERO-SUM REGRESSION TESTS")
    print("=" * 70)
    test_normalization_betting_features()
    test_raw_vs_normalized_mismatch_detection()
    test_identical_weights_ratio_is_exact_one()
    test_action_masking_integrity()
    test_multi_agent_zero_sum_invariant()
    print("=" * 70)
    print("ALL DIRECTIVE 043 PRE-REQUISITE TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
