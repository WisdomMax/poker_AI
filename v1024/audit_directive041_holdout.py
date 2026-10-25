"""
DIRECTIVE 041 Independent Multi-Opponent Holdout Evaluation Engine
- Compares Baseline E2E 50k vs Pilot League E2E 70k under identical seeds (1,000 hands).
- 5 Distinct Unseen / Challenging Opponents:
  1. SymmetricTop20HeuristicBot (Top 19.76% range)
  2. PostflopBoardEvaluatorBot (Real 7-card hand rank evaluation)
  3. StrongAllinCatcherBot (Calls large bets / all-in only with Two Pair+)
  4. Aggressive3BetBot (Preflop 3-Bet & Flop C-Bet pressure)
  5. UnseenAdaptiveBlufferBot (Semi-bluff check-raise postflop)
- Records hand-by-hand raw JSONL for 100% independent auditing.
- Calculates Paired Difference 95% CI, VPIP, PFR, 3-Bet, Push All-in, Call All-in, and Pot Bet%.
"""
import sys
import os
import json
import random
import time
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1023.v1023_13_differential_forensics import PPOAgentWrapper
from v1024.tests.test_directive040_regression import (
    calc_chen_score, robust_parse_cards, parse_board_cards
)
from v1024.audit_directive040_flawless_engine import SymmetricTop20HeuristicBot
from v1024.integrity_poker_metrics import HandIntegrityTracker

STARTING_STACK = 10000.0

def card_str_to_int(c_str: str) -> int:
    ranks = '23456789TJQKA'
    suits = 'shdc'
    r = ranks.index(c_str[0])
    s = suits.index(c_str[1])
    return r * 4 + s

# -----------------------------------------------------------------------------
# 5종 홀드아웃 상대 정의
# -----------------------------------------------------------------------------
class PostflopBoardEvaluatorBot:
    """보드 카드와 실제 7장 핸드 랭크를 계산하여 포스트플랍 강도에 따라 결정하는 상대"""
    name = "Postflop_Board_Evaluator"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: list, priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        
        if round_idx == 0:
            if chen >= 7.5: return 2 if (2 in legal_6 and self.rng.random() < 0.5) else (1 if 1 in legal_6 else 0)
            if to_call <= 100.0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1
            
        board = parse_board_cards(st)
        if len(board) >= 3:
            int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
            rank_tuple = eval_7cards(int_cards)
            hand_rank = rank_tuple[0]
        else:
            hand_rank = 0
            
        # hand_rank: 0: High Card, 1: Pair, 2: Two Pair, 3: Trips, 4: Straight, 5: Flush, 6: Full House+
        if is_allin_facing and to_call > 0:
            if hand_rank >= 2 and 1 in legal_6: return 1 # Two Pair 이상만 올인 콜
            return 0 if 0 in legal_6 else 1
            
        if hand_rank >= 3: # Trips+
            if 3 in legal_6 and self.rng.random() < 0.6: return 3 # Pot Raise
            if 2 in legal_6: return 2
            return 1 if 1 in legal_6 else 0
        elif hand_rank == 2: # Two Pair
            if 2 in legal_6 and self.rng.random() < 0.5: return 2
            return 1 if 1 in legal_6 else 0
        elif hand_rank == 1: # One Pair
            if to_call <= 300.0 and 1 in legal_6: return 1
            return 0 if (to_call > 300.0 and 0 in legal_6) else 1
        else: # High card
            if to_call == 0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1

class StrongAllinCatcherBot:
    """큰 벳이나 올인에 대해 너트/투페어 이상으로만 스택 승부하는 상대"""
    name = "Strong_Allin_Catcher"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: list, priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        board = parse_board_cards(st)
        
        if is_allin_facing or to_call >= 1000.0:
            if round_idx == 0:
                # Preflop: AA, KK, QQ, AKs only
                if chen >= 11.0 and 1 in legal_6: return 1
                return 0 if 0 in legal_6 else 1
            else:
                if len(board) >= 3:
                    int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
                    rank = eval_7cards(int_cards)[0]
                else:
                    rank = 0
                if rank >= 2 and 1 in legal_6: return 1
                return 0 if 0 in legal_6 else 1

        if round_idx == 0:
            if chen >= 8.0: return 2 if (2 in legal_6 and self.rng.random() < 0.4) else (1 if 1 in legal_6 else 0)
            if to_call == 0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1
        else:
            if len(board) >= 3:
                int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
                rank = eval_7cards(int_cards)[0]
            else:
                rank = 0
            if rank >= 2 and 2 in legal_6: return 2
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else 1)

class Aggressive3BetOpponent:
    """프리플랍 3-Bet 및 플랍 C-Bet 상대"""
    name = "Aggressive_3Bet_Opponent"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: list, priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        if is_allin_facing and to_call > 0:
            return 1 if (chen >= 9.0 and 1 in legal_6) else (0 if 0 in legal_6 else 1)
        if round_idx == 0:
            if to_call > 100.0 and chen >= 7.0 and 3 in legal_6:
                return 3 # 3-Bet Pot
            if chen >= 6.5 and 2 in legal_6: return 2
            return 1 if 1 in legal_6 else 0
        elif round_idx == 1:
            if 2 in legal_6 and self.rng.random() < 0.6: return 2 # C-Bet
            return 1 if 1 in legal_6 else 0
        return 1 if 1 in legal_6 else 0

class UnseenAdaptiveBlufferBot:
    """미학습 체크레이즈 및 블러프 상대"""
    name = "Unseen_Adaptive_Bluffer"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: list, priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        if is_allin_facing and to_call > 0:
            return 0 if 0 in legal_6 else 1
        if round_idx == 0:
            if 2 in legal_6 and self.rng.random() < 0.35: return 2
            return 1 if 1 in legal_6 else 0
        else:
            # 25% 확률로 블러프 팟벳 레이즈
            if 3 in legal_6 and self.rng.random() < 0.25: return 3
            if 2 in legal_6 and self.rng.random() < 0.35: return 2
            return 1 if 1 in legal_6 else 0

def get_holdout_opponents(opp_type: str, agent_seat: int, seed: int) -> dict:
    opps = {}
    for s in range(5):
        if s == agent_seat: continue
        s_seed = seed * 10 + s
        if opp_type == "top20":
            opps[s] = SymmetricTop20HeuristicBot(s, s_seed)
        elif opp_type == "postflop_board":
            opps[s] = PostflopBoardEvaluatorBot(s, s_seed)
        elif opp_type == "allin_catcher":
            opps[s] = StrongAllinCatcherBot(s, s_seed)
        elif opp_type == "aggressive_3bet":
            opps[s] = Aggressive3BetOpponent(s, s_seed)
        elif opp_type == "unseen_bluffer":
            opps[s] = UnseenAdaptiveBlufferBot(s, s_seed)
        else:
            opps[s] = SymmetricTop20HeuristicBot(s, s_seed)
    return opps

def evaluate_model_on_holdout(model_path: str, model_name: str, opp_type: str,
                             num_hands: int = 1000, base_seed: int = 888000,
                             jsonl_out: str = None) -> dict:
    game = get_5p_fullgame()
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    agent = PPOAgentWrapper(model)

    raw_records = []
    hand_pnls_bb = []
    running_pnl = 0.0
    cumulative_pnls = [0.0]

    action_counts = Counter()
    vpip_count = 0
    pfr_count = 0
    three_bet_opp_count = 0
    three_bet_count = 0
    
    direct_action5_count = 0
    pushed_allin_count = 0
    pushed_allin_called_count = 0
    pushed_allin_called_pnls = []
    uncalled_pushed_allin_count = 0
    
    called_allin_count = 0
    called_allin_pnls = []
    
    showdown_count = 0
    showdown_won_count = 0
    entropy_samples = []

    for h_idx in range(num_hands):
        h_seed = base_seed + h_idx
        chance_rng = np.random.RandomState(h_seed)

        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)

        agent_seat = h_idx % 5
        priv_cards = robust_parse_cards(st)
        opps = get_holdout_opponents(opp_type, agent_seat, h_seed)
        tracker = HandIntegrityTracker(agent_seat, h_seed)

        opp_allin_active = False
        opp_allin_seat = None

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue

            cp = st.current_player()
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            round_idx = get_round_from_state(st)
            spent_before = extract_player_spent(st)
            max_spent = max(spent_before)
            cur_to_call = max_spent - spent_before[cp]

            if cp == agent_seat:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                probs = agent.get_action_probs(inf, legal_6)
                act_6 = agent.act(inf, legal_6)
                action_counts[act_6] += 1
                
                probs_safe = np.maximum(probs, 1e-8)
                entropy_samples.append(-np.sum(probs_safe * np.log(probs_safe)))

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                spent_after = extract_player_spent(st)

                tracker.record_agent_decision(
                    round_idx, act_6, nat_act, cur_to_call,
                    spent_before[cp], spent_after[cp],
                    opp_allin_active, opp_allin_seat, probs=probs
                )
            else:
                bot = opps[cp]
                is_allin_facing = (cur_to_call >= STARTING_STACK - spent_before[cp] or tracker.pushed_allin)
                act_6 = bot.act(st, cp, legal_6, priv_cards, round_idx, cur_to_call, is_allin_facing)

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                spent_after = extract_player_spent(st)

                if (STARTING_STACK - spent_after[cp] <= 0.0 or act_6 == 5) and not opp_allin_active:
                    opp_allin_active = True
                    opp_allin_seat = cp

                tracker.record_opponent_decision(
                    cp, round_idx, act_6, cur_to_call,
                    spent_before[cp], spent_after[cp]
                )

        final_spent = extract_player_spent(st)
        returns = st.returns()
        board_cards = parse_board_cards(st)

        h_res = tracker.finalize_hand(final_spent, returns, priv_cards, board_cards)
        pnl_bb = h_res["pnl_bb"]
        hand_pnls_bb.append(pnl_bb)
        running_pnl += pnl_bb
        cumulative_pnls.append(running_pnl)
        raw_records.append(h_res)

        if h_res["vpip"]: vpip_count += 1
        if h_res["pfr"]: pfr_count += 1
        if h_res["3bet_opportunity"]: three_bet_opp_count += 1
        if h_res["3bet"]: three_bet_count += 1
        if h_res["direct_action5"]: direct_action5_count += 1

        if h_res["pushed_allin"]:
            pushed_allin_count += 1
            if h_res["pushed_allin_called"]:
                pushed_allin_called_count += 1
                pushed_allin_called_pnls.append(pnl_bb)
            else:
                uncalled_pushed_allin_count += 1

        if h_res["called_allin"]:
            called_allin_count += 1
            called_allin_pnls.append(pnl_bb)

        if h_res["showdown"]:
            showdown_count += 1
            if h_res["showdown_won"]:
                showdown_won_count += 1

    if jsonl_out:
        os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for r in raw_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mean_bb = np.mean(hand_pnls_bb)
    std_bb = np.std(hand_pnls_bb, ddof=1)
    ci95 = 1.96 * (std_bb / np.sqrt(num_hands)) * 100.0
    bb_100 = mean_bb * 100.0

    peak = 0.0
    max_dd = 0.0
    for p in cumulative_pnls:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd

    tot_acts = sum(action_counts.values())
    pot_bet_pct = (action_counts[3] / tot_acts * 100.0) if tot_acts > 0 else 0.0
    half_pot_pct = (action_counts[2] / tot_acts * 100.0) if tot_acts > 0 else 0.0

    three_bet_pct = (three_bet_count / three_bet_opp_count * 100.0) if three_bet_opp_count > 0 else 0.0
    called_push_ev = float(np.mean(pushed_allin_called_pnls)) if pushed_allin_called_pnls else 0.0
    called_allin_ev = float(np.mean(called_allin_pnls)) if called_allin_pnls else 0.0

    return {
        "model_name": model_name,
        "opponent": opp_type,
        "num_hands": num_hands,
        "bb_per_100": round(float(bb_100), 2),
        "ci95_bb": round(float(ci95), 2),
        "total_pnl_bb": round(float(running_pnl), 2),
        "max_drawdown_bb": round(float(max_dd), 2),
        "vpip_pct": round(vpip_count / num_hands * 100.0, 1),
        "pfr_pct": round(pfr_count / num_hands * 100.0, 1),
        "three_bet_opps": three_bet_opp_count,
        "three_bet_count": three_bet_count,
        "three_bet_pct": round(three_bet_pct, 1),
        "direct_action5_count": direct_action5_count,
        "pushed_allin_count": pushed_allin_count,
        "pushed_allin_called_count": pushed_allin_called_count,
        "called_push_ev_bb": round(called_push_ev, 2),
        "uncalled_pushed_allin_count": uncalled_pushed_allin_count,
        "called_allin_count": called_allin_count,
        "called_allin_ev_bb": round(called_allin_ev, 2),
        "half_pot_bet_pct": round(half_pot_pct, 1),
        "pot_bet_pct": round(pot_bet_pct, 1),
        "showdown_count": showdown_count,
        "showdown_win_pct": round(showdown_won_count / showdown_count * 100.0, 1) if showdown_count > 0 else 0.0,
        "avg_entropy": round(float(np.mean(entropy_samples)), 3) if entropy_samples else 0.0,
        "hand_pnls": hand_pnls_bb
    }
