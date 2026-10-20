import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
    TightPassiveBot
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v83.run_v83_robust_range_and_cfr_audit import RobustBayesianRangeModel
from v86.run_v86_meta_controller_and_independent_audit import BalancedBot
from v87.run_v87_adversarial_red_team_arena import UltraAggressiveBot
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v106.run_v106_deterministic_replay import DeterministicBalancedBot

# =====================================================================
# DETERMINISTIC AGENTS & RNG ISOLATION
# =====================================================================
class DeterministicTrueNFSP_Agent(TrueNFSP_6Action_Agent):
    """전역 np.random을 전혀 사용하지 않고 독립 RandomState로 샘플링하는 NFSP 에이전트"""
    def select_action_deterministic(self, info: np.ndarray, legal_6: list, rng: np.random.RandomState):
        norm_info = normalize_inf_state_v79(info)
        x = torch.tensor(norm_info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True

        self.sl_net.eval()
        with torch.no_grad():
            lg = self.sl_net(x).squeeze(0)
            lg[~mask] = -1e9
            probs = F.softmax(lg, dim=-1).numpy()
        
        return int(rng.choice(self.action_dim, p=probs))

def estimate_range_aware_equity_deterministic(
    hero_cards, board_cards, opp_combos, opp_probs, num_active_opponents,
    num_sims=100, rng_stream=None
):
    known_cards = set(hero_cards) | set(board_cards)
    unknown_deck = list(set(range(52)) - known_cards)
    cards_needed_board = 5 - len(board_cards)
    
    if len(unknown_deck) < (2 * num_active_opponents + cards_needed_board):
        return 0.50

    if rng_stream is None:
        raise ValueError("CRITICAL: rng_stream (RandomState) must be explicitly provided!")

    wins = 0
    ties = 0
    num_combos = len(opp_combos)
    if num_combos == 0:
        return 0.50

    for _ in range(num_sims):
        opp1_idx = rng_stream.choice(num_combos, p=opp_probs)
        opp1_hole = list(opp_combos[opp1_idx])

        current_blocked = known_cards | set(opp1_hole)
        rem_deck = [c for c in unknown_deck if c not in current_blocked]

        extra_opps = num_active_opponents - 1
        needed_cards = cards_needed_board + 2 * extra_opps
        if len(rem_deck) < needed_cards:
            continue
        
        sim_sampled = rng_stream.choice(rem_deck, size=needed_cards, replace=False)
        sim_board = list(board_cards) + list(sim_sampled[:cards_needed_board])
        
        hero_full = list(hero_cards) + sim_board
        hero_score = eval_7cards(hero_full)

        hero_won = True
        hero_tied = False

        opp1_full = opp1_hole + sim_board
        opp1_score = eval_7cards(opp1_full)
        if opp1_score > hero_score:
            hero_won = False
        elif opp1_score == hero_score:
            hero_tied = True

        if hero_won or hero_tied:
            for i in range(extra_opps):
                o_hole = list(sim_sampled[cards_needed_board + 2*i : cards_needed_board + 2*i + 2])
                o_score = eval_7cards(o_hole + sim_board)
                if o_score > hero_score:
                    hero_won = False
                    hero_tied = False
                    break
                elif o_score == hero_score:
                    hero_tied = True

        if hero_won and not hero_tied:
            wins += 1
        elif hero_tied:
            ties += 1

    return float(wins + 0.5 * ties) / float(num_sims)


class V108_Deterministic_TempHybrid:
    """Model A: 결정론적 Temperature Hybrid T=2.0 기준 모델"""
    def __init__(self, bp_agent: DeterministicTrueNFSP_Agent, num_sims=100):
        self.blueprint = bp_agent
        self.temperature = 2.0
        self.num_sims = num_sims

    def select_action(self, st, cp, legal_6, folded_players=None, hand_seed=0, decision_idx=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)

        # 1. 수비 상황 (To-call > 0) -> Temp-Hybrid 수비 엔진
        if to_call > 0 and 0 in legal_6:
            hero_cards = list(np.where(inf[5:57] == 1.0)[0])
            board_cards = list(np.where(inf[57:109] == 1.0)[0])
            active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            
            eq_seed = (hand_seed * 10007 + cp * 101 + decision_idx * 17) & 0xFFFFFFFF
            eq_rng = np.random.RandomState(eq_seed)
            equity = estimate_range_aware_equity_deterministic(
                hero_cards, board_cards, combos, probs, active_opps,
                num_sims=self.num_sims, rng_stream=eq_rng
            )

            if equity > pot_odds + 0.05:
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if (raises and equity > 0.65) else (1 if 1 in legal_6 else legal_6[0])
            elif equity < pot_odds - 0.05:
                return 0 # Clean Fold
            else:
                return 1 if 1 in legal_6 else legal_6[0]

        # 2. 자유 행동 상황 (To-call == 0) -> Blueprint 신경망 (결정론적 샘플링)
        bp_seed = (hand_seed * 20011 + cp * 307 + decision_idx * 31) & 0xFFFFFFFF
        bp_rng = np.random.RandomState(bp_seed)
        return self.blueprint.select_action_deterministic(inf, legal_6, rng=bp_rng)

class DeterministicTightPassiveBot:
    """결정론적 RNG 인스턴스를 사용하는 TightPassiveBot"""
    def __init__(self, rng_seed=20267902):
        self.rng = random.Random(rng_seed)

    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        if to_call > 150 and 0 in legal_6 and self.rng.random() < 0.55:
            return 0
        return 1

class DeterministicUltraAggressiveBot:
    """결정론적 RNG 인스턴스를 사용하는 UltraAggressiveBot"""
    def __init__(self, rng_seed=20268704):
        self.rng = random.Random(rng_seed)

    def select_action(self, st, cp, legal_6: list):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        if to_call > 0:
            raises = [a for a in legal_6 if a in [2, 3, 5]]
            if raises and self.rng.random() < 0.80:
                return raises[-1]
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else 1)
        else:
            bets = [a for a in legal_6 if a in [2, 3]]
            if bets and self.rng.random() < 0.85:
                return bets[-1]
            return 1

def get_deterministic_fixed_opponents_v108(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 100 + 1),
        DeterministicTightPassiveBot(rng_seed=seed_h * 100 + 2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(rng_seed=seed_h * 100 + 4)
    ]

def test_self_regression():
    print("=" * 80)
    print("PHASE 2: SELF-REGRESSION BIT-EXACT REPRODUCIBILITY TEST (100 HANDS)")
    print("=" * 80)

    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    model_a_test = V108_Deterministic_TempHybrid(bp_agent, num_sims=100)

    regression_identical = True
    max_diff = 0.0
    action_seq_mismatches = 0
    chip_flow_mismatches = 0

    for h in range(100):
        seed_h = 20261100 + h
        seat_test = h % 5

        # Run 1
        opps_1 = get_deterministic_fixed_opponents_v108(seed_h)
        st_1 = game.new_initial_state()
        deck_rng_1 = np.random.RandomState(seed_h)
        advance_chance_nodes(st_1, deck_rng_1)
        f_1 = set()
        dec_1 = 0
        acts_1 = []
        bets_1 = []

        while not st_1.is_terminal():
            if st_1.is_chance_node():
                advance_chance_nodes(st_1, deck_rng_1)
            else:
                cp = st_1.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_1, cp)
                if cp == seat_test:
                    a = model_a_test.select_action(st_1, cp, l6, folded_players=f_1, hand_seed=seed_h, decision_idx=dec_1)
                    dec_1 += 1
                else:
                    opp = opps_1[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_1, cp, l6)
                if a == 0: f_1.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_1, cp, a)
                acts_1.append((cp, a))
                bets_1.append((cp, eng_amt))
                st_1.apply_action(eng_amt)
        ret_1 = st_1.returns()[seat_test] / 100.0

        # Run 2 (동일 조건 독립 재실행)
        opps_2 = get_deterministic_fixed_opponents_v108(seed_h)
        st_2 = game.new_initial_state()
        deck_rng_2 = np.random.RandomState(seed_h)
        advance_chance_nodes(st_2, deck_rng_2)
        f_2 = set()
        dec_2 = 0
        acts_2 = []
        bets_2 = []

        while not st_2.is_terminal():
            if st_2.is_chance_node():
                advance_chance_nodes(st_2, deck_rng_2)
            else:
                cp = st_2.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_2, cp)
                if cp == seat_test:
                    a = model_a_test.select_action(st_2, cp, l6, folded_players=f_2, hand_seed=seed_h, decision_idx=dec_2)
                    dec_2 += 1
                else:
                    opp = opps_2[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_2, cp, l6)
                if a == 0: f_2.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_2, cp, a)
                acts_2.append((cp, a))
                bets_2.append((cp, eng_amt))
                st_2.apply_action(eng_amt)
        ret_2 = st_2.returns()[seat_test] / 100.0

        diff = abs(ret_1 - ret_2)
        if diff > max_diff:
            max_diff = diff
        if acts_1 != acts_2:
            action_seq_mismatches += 1
        if bets_1 != bets_2:
            chip_flow_mismatches += 1
        if diff > 1e-9:
            regression_identical = False

    print(f"  * Self-Regression 100 Hands Test : {'PASSED (Bit-Exact 0.000000 BB)' if regression_identical else 'FAILED'}")
    print(f"  * Action Sequence Mismatches     : {action_seq_mismatches} / 100")
    print(f"  * Chip Flow Amount Mismatches    : {chip_flow_mismatches} / 100")
    print(f"  * Max Payoff Difference Detected : {max_diff:.8f} BB")

    res = {
        "passed": regression_identical,
        "max_diff": max_diff,
        "action_seq_mismatches": action_seq_mismatches,
        "chip_flow_mismatches": chip_flow_mismatches
    }
    with open("v108/data/phase2_self_regression.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    test_self_regression()
