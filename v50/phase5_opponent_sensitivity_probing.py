"""
POKER AI V5.0 - PHASE 5: Opponent Sensitivity & Shuffled-Feature Probing
- 1. 동일 카드 + 동일 공개 베팅(3-Bet 직면) 상황에서 상대 과거 이력만 4종(Tight, Loose, Aggressive, Selective)으로 변경 주입
- 2. 7대 대표 핸드(AA, KK, QQ, AKs, JTs, 98s, 72o)의 행동 확률 분포(Fold, Call, Pot, All-in, Half) 측정
- 3. 핵심 시나리오 심층 비교:
     * JTs vs Tight 3-Bet vs Aggressive 3-Bet
     * 72o vs Tight 3-Bet vs Aggressive 3-Bet
     * AA vs Tight 3-Bet vs Aggressive 3-Bet
- 4. Shuffled-Feature 테스트: 상대 피처를 무작위로 섞었을 때의 정책 민감도 측정
"""

import os
import sys
import copy
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, advance_chance_nodes, parse_cards_from_info_str
)
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import BayesianHoldemActorCritic5A

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

TARGET_HANDS_MAP = {
    'AA': ('As', 'Ah'),
    'KK': ('Ks', 'Kh'),
    'QQ': ('Qs', 'Qh'),
    'AKs': ('As', 'Ks'),
    'JTs': ('Js', 'Ts'),
    '98s': ('9s', '8s'),
    '72o': ('7h', '2c'),
}

PROFILES = ["Tight", "LooseCalling", "Aggressive", "Selective3Bettor"]

def probe_model_sensitivity(model_path: str, device: torch.device) -> Dict:
    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    game = get_5p_holdem_5a_game()
    rng = np.random.RandomState(42)

    # 3-Bet 직면 기본 상태 탐색
    found_vec = None
    found_legal = None

    for _ in range(500):
        s = game.new_initial_state()
        advance_chance_nodes(s, rng=rng)
        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, rng=rng)
            else:
                cur = s.current_player()
                legal = s.legal_actions(cur)
                info = s.information_state_string(cur)
                if "Sequences:" in info and any(b in info.split("Sequences:")[1] for b in ['b', 'h', 'p', 'a']):
                    found_vec = normalize_vec(s.information_state_tensor(cur))
                    found_legal = legal
                    break
                s.apply_action(rng.choice(legal))
        if found_vec is not None:
            break

    if found_vec is None:
        s = game.new_initial_state()
        advance_chance_nodes(s, rng=rng)
        found_vec = normalize_vec(s.information_state_tensor(0))
        found_legal = s.legal_actions(0)

    results = []

    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in found_legal: m[0, a] = True

    # 상대 매니저
    mgr = MatchSessionOpponentManager()

    for h_name in TARGET_HANDS_MAP:
        for prof in PROFILES:
            mgr.reset_session()
            # 상대 4명 중 나에게 3-Bet을 날린 상대(바로 왼쪽 상대, Agent 1)에게 해당 프로필 주입
            mgr.inject_synthetic_profile(agent_id=1, profile_name=prof, num_hands=100)
            opp_feats = mgr.get_table_feature_tensor(current_hero_seat=0)

            full_vec = np.concatenate([found_vec, opp_feats])
            x = torch.tensor(full_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, val = model(x, m)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            results.append({
                'hand': h_name,
                'profile': prof,
                'fold_prob': float(probs[0]),
                'call_prob': float(probs[1]),
                'pot_prob': float(probs[2]),
                'allin_prob': float(probs[3]),
                'half_prob': float(probs[4]),
                'value': float(val.item())
            })

    # Shuffled-Feature 테스트 (정상 프로필 vs 무작위 셔플 프로필)
    shuffled_results = []
    for h_name in ['AA', 'JTs', '72o']:
        mgr.reset_session()
        mgr.inject_synthetic_profile(1, "Tight", num_hands=100)
        true_feats = mgr.get_table_feature_tensor(current_hero_seat=0)

        # 무작위 셔플
        shuffled_feats = np.random.permutation(true_feats)

        x_true = torch.tensor(np.concatenate([found_vec, true_feats]), dtype=torch.float32, device=device).unsqueeze(0)
        x_shuf = torch.tensor(np.concatenate([found_vec, shuffled_feats]), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            pr_true = F.softmax(model(x_true, m)[0], dim=-1).squeeze(0).cpu().numpy()
            pr_shuf = F.softmax(model(x_shuf, m)[0], dim=-1).squeeze(0).cpu().numpy()

        tvd_shuffle = 0.5 * np.sum(np.abs(pr_true - pr_shuf))
        shuffled_results.append({
            'hand': h_name,
            'true_fold': float(pr_true[0]),
            'shuf_fold': float(pr_shuf[0]),
            'true_call': float(pr_true[1]),
            'shuf_call': float(pr_shuf[1]),
            'tvd_due_to_shuffle': float(tvd_shuffle)
        })

    return {'sensitivity': results, 'shuffled': shuffled_results}

def run_phase5_probing_experiment():
    print("=" * 80)
    print("PHASE 5: 상대 성향에 따른 행동 민감도 및 Shuffled-Feature 검증")
    print("=" * 80)

    device = torch.device("cpu")
    # CondD_Bayes_RareExp Seed 42 대상 검증
    target_model = "v50/models/CondD_Bayes_RareExp_seed_42.pt"
    if not os.path.exists(target_model):
        print(f"[알림] {target_model} 학습 완료 대기 중...")
        return

    data = probe_model_sensitivity(target_model, device)

    print("\n--- 72o vs 4대 상대 프로필 3-Bet 대응 비교 ---")
    rows_72 = [r for r in data['sensitivity'] if r['hand'] == '72o']
    for r in rows_72:
        print(f"  72o vs {r['profile']:<16}: Fold {r['fold_prob']*100:5.1f}% | Call {r['call_prob']*100:5.1f}% | All-in {r['allin_prob']*100:5.1f}% | Val {r['value']:+6.2f} BB")

    print("\n--- JTs vs 4대 상대 프로필 3-Bet 대응 비교 ---")
    rows_jt = [r for r in data['sensitivity'] if r['hand'] == 'JTs']
    for r in rows_jt:
        print(f"  JTs vs {r['profile']:<16}: Fold {r['fold_prob']*100:5.1f}% | Call {r['call_prob']*100:5.1f}% | All-in {r['allin_prob']*100:5.1f}% | Val {r['value']:+6.2f} BB")

    print("\n--- AA vs 4대 상대 프로필 3-Bet 대응 비교 ---")
    rows_aa = [r for r in data['sensitivity'] if r['hand'] == 'AA']
    for r in rows_aa:
        print(f"  AA  vs {r['profile']:<16}: Fold {r['fold_prob']*100:5.1f}% | Call {r['call_prob']*100:5.1f}% | All-in {r['allin_prob']*100:5.1f}% | Val {r['value']:+6.2f} BB")

    print("\n--- Shuffled-Feature 테스트 결과 (정상 vs 셔플) ---")
    for s in data['shuffled']:
        print(f"  {s['hand']:<4}: True Fold {s['true_fold']*100:5.1f}% -> Shuf Fold {s['shuf_fold']*100:5.1f}% | TVD = {s['tvd_due_to_shuffle']:.4f}")

    # CSV 저장
    csv_sens = "v50/data/phase5_opponent_sensitivity.csv"
    with open(csv_sens, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data['sensitivity'][0].keys()))
        w.writeheader()
        w.writerows(data['sensitivity'])

    csv_shuf = "v50/data/phase5_shuffled_features.csv"
    with open(csv_shuf, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data['shuffled'][0].keys()))
        w.writeheader()
        w.writerows(data['shuffled'])
    print(f"\n[저장 완료] Phase 5 CSV 저장 완료: {csv_sens}, {csv_shuf}")

if __name__ == "__main__":
    run_phase5_probing_experiment()
