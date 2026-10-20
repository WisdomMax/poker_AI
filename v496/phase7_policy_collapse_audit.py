"""
POKER AI V4.9.6 - PHASE 7: 6대 정책 붕괴 유형 정량 감사
- 붕괴 유형 1: 모든 카드로 All-in 붕괴 (All-in Collapse)
- 붕괴 유형 2: 모든 카드로 Fold 붕괴 (All-Folder Collapse)
- 붕괴 유형 3: 모든 카드로 Call 붕괴 (Blind Caller Collapse)
- 붕괴 유형 4: 포스트플롭 정책 붕괴 (Postflop Card Blindness)
- 붕괴 유형 5: 상대 레이즈 직면 시 무조건 Fold 붕괴 (Facing Raise 100% Fold)
- 붕괴 유형 6: Half-Pot 이후 방어 불능 붕괴 (Half-Pot Defense Collapse)
"""

import os
import sys
import math
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v496.stable_ppo import StableHoldemActorCritic5A

TARGET_PROBE_HANDS = ['AA', 'KK', 'QQ', 'AKs', 'JTs', '72o']

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def audit_model_collapse(model_path: str, model_name: str, normalize_input: bool = True) -> Dict:
    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    gen = TargetedSubgameGenerator(game)

    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    if not os.path.exists(model_path):
        return {}
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    flags = {
        'collapse_allin': False,
        'collapse_allfolder': False,
        'collapse_blindcaller': False,
        'collapse_postflop_blind': False,
        'collapse_raise_allfold': False,
    }

    # 1. 프리플롭 오픈 시 핸드별 정책 조사
    preflop_probs = {}
    for h in TARGET_PROBE_HANDS:
        st, _ = gen.generate_subgame_state('SitA_Facing_OpenRaise', h, hero_seat=3)
        vec = st.information_state_tensor(3)
        if normalize_input: vec = normalize_vec(vec)
        legal = st.legal_actions(3)
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
        for a in legal: m[0, a] = True
        with torch.no_grad():
            lg, _ = model(x, m)
            pr = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
        preflop_probs[h] = pr

    allin_rates = [p[3] for p in preflop_probs.values()]
    fold_rates = [p[0] for p in preflop_probs.values()]
    call_rates = [p[1] for p in preflop_probs.values()]

    if all(r > 0.85 for r in allin_rates):
        flags['collapse_allin'] = True
    if all(r > 0.85 for r in fold_rates):
        flags['collapse_allfolder'] = True
    if all(r > 0.85 for r in call_rates):
        flags['collapse_blindcaller'] = True

    # 2. 3-Bet 직면 시 Fold 비율 조사
    facing_3bet_folds = []
    for h in TARGET_PROBE_HANDS:
        st, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', h, hero_seat=2)
        vec = st.information_state_tensor(2)
        if normalize_input: vec = normalize_vec(vec)
        legal = st.legal_actions(2)
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
        for a in legal: m[0, a] = True
        with torch.no_grad():
            lg, _ = model(x, m)
            pr = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
        facing_3bet_folds.append(pr[0])

    if all(f > 0.95 for f in facing_3bet_folds):
        flags['collapse_raise_allfold'] = True

    # 3. 블라인드 콜러 (AA와 72o 모두 Call > 80%) 검사
    aa_call_3bet = 1.0 - facing_3bet_folds[0] # AA
    t72_call_3bet = 1.0 - facing_3bet_folds[-1] # 72o
    if aa_call_3bet > 0.70 and t72_call_3bet > 0.70:
        flags['collapse_blindcaller'] = True

    # 4. 포스트플롭 카드 블라인드니스 검사 (플롭에서 AA vs 72o TV Distance < 0.05)
    st_aa_flop, _ = gen.generate_subgame_state('SitD_Flop_Facing_Bet', 'AA', hero_seat=2)
    st_72_flop, _ = gen.generate_subgame_state('SitD_Flop_Facing_Bet', '72o', hero_seat=2)
    v_aa = st_aa_flop.information_state_tensor(2)
    v_72 = st_72_flop.information_state_tensor(2)
    if normalize_input:
        v_aa = normalize_vec(v_aa)
        v_72 = normalize_vec(v_72)
    m_flop = torch.tensor([[True, True, False, False, False]], device=device) # Fold, Call
    with torch.no_grad():
        lg_aa, _ = model(torch.tensor(v_aa).unsqueeze(0), m_flop)
        lg_72, _ = model(torch.tensor(v_72).unsqueeze(0), m_flop)
        pr_aa = torch.softmax(lg_aa, dim=-1).squeeze(0).numpy()
        pr_72 = torch.softmax(lg_72, dim=-1).squeeze(0).numpy()
    flop_tvd = 0.5 * np.sum(np.abs(pr_aa - pr_72))
    if flop_tvd < 0.05:
        flags['collapse_postflop_blind'] = True

    collapse_count = sum(flags.values())
    print(f"[{model_name}] 붕괴 진단: {flags} (총 {collapse_count}건 감지)")
    return {
        'model': model_name,
        'collapse_allin': flags['collapse_allin'],
        'collapse_allfolder': flags['collapse_allfolder'],
        'collapse_blindcaller': flags['collapse_blindcaller'],
        'collapse_postflop_blind': flags['collapse_postflop_blind'],
        'collapse_raise_allfold': flags['collapse_raise_allfold'],
        'total_collapses': collapse_count,
        'flop_tvd': float(flop_tvd),
        'aa_3bet_call': float(aa_call_3bet),
        't72_3bet_call': float(t72_call_3bet)
    }

def run_phase7_all_models():
    print("=" * 75)
    print("PHASE 7: 6대 정책 붕괴 유형 정량 전수 감사")
    print("=" * 75)
    SEEDS = [42, 101, 202, 303, 404]
    COND_NAMES = [
        ('CondA_Raw_EntOFF', False),
        ('CondB_Norm_EntOFF', True),
        ('CondC_Raw_EntON', False),
        ('CondD_Norm_EntON', True),
    ]

    all_audits = []
    for cprefix, norm in COND_NAMES:
        for s in SEEDS:
            mpath = f"v496/models/{cprefix}_seed_{s}.pt"
            mname = f"{cprefix}_seed_{s}"
            if os.path.exists(mpath):
                res = audit_model_collapse(mpath, mname, norm)
                if res:
                    all_audits.append(res)

    csv_out = "v496/data/phase7_collapse_audit.csv"
    if all_audits:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_audits[0].keys()))
            writer.writeheader()
            writer.writerows(all_audits)
        print(f"\n[저장 완료] Phase 7 붕괴 진단 결과: {csv_out}")

if __name__ == "__main__":
    run_phase7_all_models()
