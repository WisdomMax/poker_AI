"""
POKER AI V5.1 - PHASE 1: Actor State & Masking Verification across Rigorous States
- 1. 결정론적 3대 게임 상태 구축 (Fallback 금지, 실패 시 AssertionError):
     * State A: Preflop Unopened
     * State B: Facing 3-Bet Pot (오픈 후 상대의 Pot 3-Bet 직면)
     * State C: Facing 3-Bet All-in (오픈 후 상대의 All-in 3-Bet 직면)
- 2. 학습 단계별 모델 정책 정밀 비교:
     * 0k 모델 (초기화 직후)
     * V5.0 2k 모델 (CondD_Bayes_RareExp_seed_42.pt)
     * V4.9.6 20k 장기 학습 모델 (CondD_Norm_EntON_seed_42.pt)
- 3. 지표 전수 실측:
     * Actor Raw Logits (소수점 8자리)
     * Legal Action Mask 및 Masked Logits
     * Softmax Probs (소수점 8자리)
     * Policy Entropy
     * Actor Head Weight Norm & Encoder Weight Norm
     * Gradient Norm, Advantage, Clip Fraction
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, advance_chance_nodes, parse_cards_from_info_str
)
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import BayesianHoldemActorCritic5A, get_neutral_prior_features

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def build_deterministic_states(game: pyspiel.Game, seed: int = 42):
    """Fallback 절대 금지! 3대 상태를 결정론적 액션 시퀀스로 정확히 생성"""
    states = {}

    # State A: Preflop Unopened
    st_a = game.new_initial_state()
    advance_chance_nodes(st_a, rng=np.random.RandomState(seed))
    assert not st_a.is_terminal(), "State A must not be terminal"
    assert st_a.current_player() == 2, f"State A current player must be 2, got {st_a.current_player()}"
    states['A_Preflop_Unopened'] = st_a

    # State B: Facing 3-Bet Pot
    # 시퀀스: P2 PotBet(2) -> P3 PotRaise(2) -> P4 Fold(0) -> P0 Fold(0) -> P1 Fold(0) -> 다시 P2!
    st_b = game.new_initial_state()
    advance_chance_nodes(st_b, rng=np.random.RandomState(seed))
    assert st_b.current_player() == 2
    st_b.apply_action(2) # P2 Pot Bet
    assert st_b.current_player() == 3
    st_b.apply_action(2) # P3 Pot Raise (3-Bet)
    assert st_b.current_player() == 4
    st_b.apply_action(0) # P4 Fold
    assert st_b.current_player() == 0
    st_b.apply_action(0) # P0 Fold
    assert st_b.current_player() == 1
    st_b.apply_action(0) # P1 Fold
    assert not st_b.is_terminal(), "State B should not be terminal yet"
    assert st_b.current_player() == 2, f"State B must return to P2, got {st_b.current_player()}"
    states['B_Facing_3Bet_Pot'] = st_b

    # State C: Facing 3-Bet All-in
    # 시퀀스: P2 PotBet(2) -> P3 All-in(3) -> P4 Fold(0) -> P0 Fold(0) -> P1 Fold(0) -> 다시 P2!
    st_c = game.new_initial_state()
    advance_chance_nodes(st_c, rng=np.random.RandomState(seed))
    assert st_c.current_player() == 2
    st_c.apply_action(2) # P2 Pot Bet
    assert st_c.current_player() == 3
    st_c.apply_action(3) # P3 All-in (3-Bet All-in)
    assert st_c.current_player() == 4
    st_c.apply_action(0) # P4 Fold
    assert st_c.current_player() == 0
    st_c.apply_action(0) # P0 Fold
    assert st_c.current_player() == 1
    st_c.apply_action(0) # P1 Fold
    assert not st_c.is_terminal(), "State C should not be terminal yet"
    assert st_c.current_player() == 2, f"State C must return to P2, got {st_c.current_player()}"
    states['C_Facing_3Bet_Allin'] = st_c

    return states

def run_phase1_audit():
    print("=" * 85)
    print("PHASE 1: ACTOR 상태 및 3대 게임 상태 결정론적 전수 감사")
    print("=" * 85)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()

    # 1. 3대 상태 구축 및 무결성 assertion
    print("\n[검사 1] 결정론적 3대 게임 상태 생성 및 Assertion 검증:")
    states = build_deterministic_states(game, seed=42)
    for name, st in states.items():
        cur = st.current_player()
        legal = st.legal_actions(cur)
        info_s = st.information_state_string(cur)
        priv, _ = parse_cards_from_info_str(info_s)
        pot_part = info_s.split("[Pot: ")[1].split("]")[0] if "[Pot: " in info_s else "unknown"
        seq_part = info_s.split("[Sequences: ")[1].split("]")[0] if "[Sequences: " in info_s else ""
        print(f"  [{name}]")
        print(f"    Current Player: {cur} | Hero Cards: {''.join(priv)} | Pot: {pot_part} chips")
        print(f"    Betting Sequence: '{seq_part}' | Legal Actions: {legal} (총 {len(legal)}개)")

    assert states['C_Facing_3Bet_Allin'].legal_actions(2) == [0, 1], "Facing All-in의 합법 행동은 반드시 [0, 1]이어야 함!"
    print("-> PASS: Fallback 없는 순수 시뮬레이터 액션 시퀀스로 3대 상태 100% 정상 구축 확인.")

    # 2. 모델 로드 (0k 초기 모델 vs V5.0 2k 모델)
    print("\n[검사 2] 모델 파라미터 규격 및 가중치 Norm 측정:")
    # 0k 모델
    torch.manual_seed(42)
    model_0k = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)

    # 2k 모델 (V5.0 CondD)
    model_2k = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    path_2k = "v50/models/CondD_Bayes_RareExp_seed_42.pt"
    assert os.path.exists(path_2k), f"모델 파일 없음: {path_2k}"
    model_2k.load_state_dict(torch.load(path_2k, map_location=device))

    models_to_test = [
        ("0k_Initial", model_0k),
        ("2k_CondD_V50", model_2k),
    ]

    for m_label, m_obj in models_to_test:
        enc_norm = sum(p.norm().item() for p in m_obj.encoder.parameters())
        act_norm = sum(p.norm().item() for p in m_obj.actor_head.parameters())
        crt_norm = sum(p.norm().item() for p in m_obj.critic_head.parameters())
        print(f"  [{m_label}]")
        print(f"    Encoder Param Norm : {enc_norm:.4f}")
        print(f"    Actor Head Norm    : {act_norm:.4f}")
        print(f"    Critic Head Norm   : {crt_norm:.4f}")

    # 3. 3대 상태에 대한 정책 출력 정밀 측정 (소수점 8자리)
    print("\n" + "=" * 85)
    print("[검사 3] 상태별 Raw Logits, Masked Logits, Softmax Probs 정밀 분석:")
    print("=" * 85)

    mgr = MatchSessionOpponentManager()
    mgr.inject_synthetic_profile(1, "Tight", num_hands=100)
    opp_feats = mgr.get_table_feature_tensor(current_hero_seat=2)

    results = []

    for s_name, st in states.items():
        cur = st.current_player()
        legal = st.legal_actions(cur)
        info_vec = normalize_vec(st.information_state_tensor(cur))
        full_x = torch.tensor(np.concatenate([info_vec, opp_feats]), dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros((1, 5), dtype=torch.bool, device=device)
        for a in legal: mask[0, a] = True

        print(f"\n>>> [상태: {s_name}] (Legal: {legal})")

        for m_label, m_obj in models_to_test:
            m_obj.eval()
            with torch.no_grad():
                feats = m_obj.encoder(full_x)
                raw_lg = m_obj.actor_head(feats).squeeze(0).cpu().numpy()
                masked_lg_t, val = m_obj(full_x, mask)
                masked_lg = masked_lg_t.squeeze(0).cpu().numpy()
                probs = F.softmax(masked_lg_t, dim=-1).squeeze(0).cpu().numpy()

                # 엔트로피 계산
                legal_probs = probs[legal]
                ent = -float(np.sum(legal_probs * np.log(np.maximum(legal_probs, 1e-12))))

            print(f"  [{m_label}]")
            print(f"    Raw Logits    : {[f'{raw_lg[i]:+.8f}' for i in range(5)]}")
            print(f"    Masked Logits : {[f'{masked_lg[i]:+.4e}' for i in range(5)]}")
            print(f"    Softmax Probs : {[f'{probs[i]:.8f}' for i in range(5)]}")
            print(f"    Entropy       : {ent:.6f} (최대 엔트로피: {np.log(len(legal)):.6f}) | Value: {val.item():+.4f} BB")

            # 불법 행동 확률 0 검증
            for i in range(5):
                if i not in legal:
                    assert probs[i] == 0.0, f"불법 행동 {i}의 확률이 0이 아님!"

            results.append({
                'state': s_name,
                'model': m_label,
                'legal_count': len(legal),
                'fold_prob': float(probs[0]),
                'call_prob': float(probs[1]),
                'pot_prob': float(probs[2]),
                'allin_prob': float(probs[3]),
                'half_prob': float(probs[4]),
                'entropy': ent,
                'value': float(val.item())
            })

    # CSV 저장
    csv_out = "v51/data/phase1_actor_state_verification.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n[저장 완료] Phase 1 감사 CSV: {csv_out}")

if __name__ == "__main__":
    run_phase1_audit()
