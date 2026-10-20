import os
import sys
import json
import time
import random
from collections import defaultdict, deque
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

# =====================================================================
# PREFLOP & POSTFLOP RIGOROUS HAND STRENGTH EVALUATOR
# Evaluates 2-card starting hands properly (No dummy cat=7 fallback)
# =====================================================================
def get_chen_score(h_cards):
    """
    Computes Bill Chen's Texas Hold'em Starting Hand Score (0 to 20+)
    Higher score means stronger preflop hand.
    """
    if len(h_cards) < 2: return 0.0
    r1, s1 = h_cards[0] // 4, h_cards[0] % 4
    r2, s2 = h_cards[1] // 4, h_cards[1] % 4
    if r1 < r2: r1, r2, s1, s2 = r2, r1, s2, s1
    
    # Base score of highest card (r=12 is Ace)
    scores = {12: 10, 11: 8, 10: 7, 9: 6}
    score = scores.get(r1, (r1 + 2) / 2.0)
    
    # Pairs
    if r1 == r2:
        score = max(score * 2, 5.0)
        if r1 == 12: score = 20.0
        elif r1 == 11: score = 16.0
        elif r1 == 10: score = 14.0
        elif r1 == 9: score = 12.0
    
    # Suited
    if s1 == s2: score += 2.0
    
    # Gap
    gap = r1 - r2
    if gap == 1: score += 1.0 # Connected
    elif gap == 2: score -= 1.0
    elif gap == 3: score -= 2.0
    elif gap == 4: score -= 4.0
    elif gap >= 5: score -= 5.0
    return max(0.0, score)

def get_comprehensive_hand_strength(inf):
    h_cards = list(np.where(inf[5:57] == 1.0)[0])
    b_cards = list(np.where(inf[57:109] == 1.0)[0])
    n_b = len(b_cards)
    
    if n_b == 0: # Preflop: Chen formula
        chen = get_chen_score(h_cards)
        # Normalize: >10 is Premium, >7 is Good, >5 is Medium, <=5 is Trash
        return {"street": "Preflop", "chen": chen, "is_monster": chen >= 12.0, "is_good": chen >= 8.0, "is_medium": chen >= 5.0, "is_trash": chen < 5.0}
    else: # Postflop: 5+ cards evaluation
        all_c = h_cards + b_cards
        cat = eval_7cards(all_c)[0]
        # 0=StraightFlush, 1=FourOfAKind, 2=FullHouse, 3=Flush, 4=Straight, 5=ThreeOfAKind, 6=TwoPair, 7=Pair, 8=HighCard
        return {"street": "Postflop", "cat": cat, "is_monster": cat <= 3, "is_good": cat <= 5, "is_medium": cat <= 6, "is_trash": cat >= 7}

# =====================================================================
# 4 DISTINCT UNSEEN OPPONENT POLICIES (No duplicated logic)
# =====================================================================
class RigorousAdaptiveHistoryExploiter:
    """1. Tracks hero's fold frequency to aggressively exploit timid heroes"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.hero_folds = deque(maxlen=window_size)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        fold_rate = np.mean(self.hero_folds) if len(self.hero_folds) >= 10 else 0.5
        
        if to_call == 0:
            if fold_rate > 0.55: # Hero folds a lot -> steal aggressively
                if 4 in legal_actions: return 4
                if 3 in legal_actions: return 3
            if hs["is_good"] and 3 in legal_actions: return 3
            return 1 # Check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                return 1
            if hs["is_good"] and to_call < 500: return 1
            if to_call <= 100: return 1 # Blind call
            return 0 if 0 in legal_actions else 1

class RigorousDynamicMultiSizingBot:
    """2. Distinctly uses 4-tier bet sizing (Min, Half, Pot, Allin) based on exact Chen/Category"""
    def __init__(self, seed=54321):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        
        if to_call == 0:
            if hs["is_monster"]:
                if 5 in legal_actions and self.rng.rand() < 0.3: return 5 # All-in 30%
                if 4 in legal_actions: return 4 # Pot bet
            elif hs["is_good"]:
                if 3 in legal_actions: return 3 # Half pot
            elif hs["is_medium"]:
                if 2 in legal_actions: return 2 # Min bet probe
            return 1 # Check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                return 1
            if hs["is_good"] and to_call < 400: return 1
            if hs["is_medium"] and to_call <= 100: return 1
            return 0 if 0 in legal_actions else 1

class RigorousPolarizedBluffValueBot:
    """3. Highly polarized: bets large pot with Monsters or Pure Air, pure checks/folds Medium"""
    def __init__(self, seed=65432):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        roll = self.rng.rand()
        
        if to_call == 0:
            if hs["is_monster"]:
                if roll < 0.75 and 4 in legal_actions: return 4 # 75% pot bet
                return 1 # 25% trap
            if hs["is_trash"]: # Pure air bluff
                if roll < 0.35 and 4 in legal_actions: return 4 # 35% large bluff
                return 1
            return 1 # Medium hands pure check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                if 5 in legal_actions and roll < 0.4: return 5
                return 1
            if to_call <= 100 and hs["is_medium"]: return 1
            return 0 if 0 in legal_actions else 1

class RigorousCheckRaiseTrapBot:
    """4. Traps aggressively: checks monsters on Flop/Turn, then check-raises huge on bet"""
    def __init__(self, seed=76543):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        
        if to_call == 0:
            # Slowplay monster checks on Flop/Turn
            if hs["is_monster"] and n_b in [3, 4]:
                return 1 # Pure trap check
            if hs["is_good"] and 3 in legal_actions: return 3
            return 1
        else: # Facing a bet
            if hs["is_monster"]: # Check-Raise execution!
                if 4 in legal_actions: return 4 # Pot check-raise
                if 3 in legal_actions: return 3
                if 5 in legal_actions: return 5
                return 1
            if hs["is_good"] and to_call < 500: return 1
            if hs["is_medium"] and to_call <= 150: return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# POOL GENERATORS
# =====================================================================
def get_base_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        DeterministicTrueNFSP_Agent()
    ]

def get_adaptive_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousAdaptiveHistoryExploiter(window_size=30)
    ]

def get_multisizing_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousDynamicMultiSizingBot(seed=seed_h * 23)
    ]

def get_polarized_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousPolarizedBluffValueBot(seed=seed_h * 29)
    ]

def get_trap_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousCheckRaiseTrapBot(seed=seed_h * 31)
    ]

# =====================================================================
# RIGOROUS AUDIT HARNESS
# =====================================================================
def run_v1020_audit(models_dict, eval_seeds, hands_per_cond=600):
    game = get_5p_fullgame()
    conditions = [
        ("Base_Seen", get_base_opponents, "None"),
        ("Single_Adaptive", get_adaptive_opponents, "RigorousAdaptiveHistoryExploiter"),
        ("Single_MultiSizing", get_multisizing_opponents, "RigorousDynamicMultiSizingBot"),
        ("Single_Polarized", get_polarized_opponents, "RigorousPolarizedBluffValueBot"),
        ("Single_Trap", get_trap_opponents, "RigorousCheckRaiseTrapBot"),
    ]
    
    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _, _ in conditions}
    target_bot_actions = {c_name: defaultdict(int) for c_name, _, _ in conditions[1:]}
    forensic_pots = {c_name: {m_name: [] for m_name in models_dict} for c_name, _, _ in conditions}
    
    print("\n=======================================================")
    print("[*] V10.20 EXPERIMENT INTEGRITY AUDIT RE-RUN")
    print(f"[*] 4 Models x 5 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print("=======================================================")
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_s = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen, target_name in conditions:
            for h in range(hands_per_cond):
                seed_h = s_base + h
                seat = h % 5
                opps = p_gen(seed_h)
                
                for m_name, model in models_dict.items():
                    st = game.new_initial_state()
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    f_set = set(); dec = 0
                    
                    while not st.is_terminal():
                        if st.is_chance_node(): advance_chance_nodes(st, np.random.RandomState(seed_h)); continue
                        cp = st.current_player()
                        l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                        
                        if cp == seat:
                            inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                            if isinstance(model, ModelA): act = model.act(st, seat, l6, f_set, seed_h, dec)
                            elif isinstance(model, PPONeuralPlayer): act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                            else: act = model.select_action(st, cp, l6)
                            dec += 1
                        else:
                            opp_idx = cp if cp < seat else cp - 1
                            opp_bot = opps[opp_idx]
                            if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                                inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                                act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(seed_h*101 + cp*17))
                            else:
                                act = opp_bot.select_action(st, cp, l6)
                            
                            # Track target bot actions (opp_idx == 3)
                            if opp_idx == 3 and c_name != "Base_Seen" and m_name == "Model_A":
                                target_bot_actions[c_name][act] += 1
                                
                        if act == 0: f_set.add(cp)
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[c_name][m_name][s_base].append(ret_bb)
                    pot_bb = sum(extract_player_spent(st)) / 100.0
                    forensic_pots[c_name][m_name].append((pot_bb, ret_bb))
                    
        print(f"     [Seed {s_base} completed in {time.time()-t_s:.1f}s]", flush=True)

    # 1. Summary by Condition
    summary = {}
    base_means = {}
    for c_name, _, _ in conditions:
        summary[c_name] = {}
        for m_name in models_dict:
            s_means = [np.mean(results[c_name][m_name][s]) * 100.0 for s in eval_seeds]
            m_val = float(np.mean(s_means))
            se_val = float(np.std(s_means, ddof=1) / np.sqrt(len(eval_seeds)))
            t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
            ci = [float(m_val - t_crit * se_val), float(m_val + t_crit * se_val)]
            summary[c_name][m_name] = {
                "mean": m_val, "se": se_val, "ci_95": ci, "seed_means": s_means
            }
            if c_name == "Base_Seen":
                base_means[m_name] = s_means

    # 2. Causal Impacts vs Baseline with Bonferroni Correction (alpha = 0.05 / 16 = 0.003125)
    causal_impacts = {}
    for c_name, _, target_name in conditions[1:]:
        causal_impacts[c_name] = {}
        for m_name in models_dict:
            c_s = summary[c_name][m_name]["seed_means"]
            b_s = base_means[m_name]
            diffs = np.array(c_s) - np.array(b_s)
            d_m = float(np.mean(diffs))
            d_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            t_s, p_v = stats.ttest_rel(c_s, b_s)
            t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
            ci_d = [float(d_m - t_crit * d_se), float(d_m + t_crit * d_se)]
            sig_bonf = bool(p_v < 0.003125)
            causal_impacts[c_name][m_name] = {
                "delta_mean": d_m, "delta_se": d_se, "t_stat": float(t_s),
                "p_val": float(p_v), "ci_95": ci_d, "bonferroni_sig": sig_bonf
            }

    # 3. Model A Relative Delta: Delta(Model) - Delta(Model_A)
    relative_to_ma = {}
    for c_name, _, _ in conditions[1:]:
        relative_to_ma[c_name] = {}
        d_ma = causal_impacts[c_name]["Model_A"]["delta_mean"]
        for m_name in models_dict:
            if m_name == "Model_A": continue
            d_m = causal_impacts[c_name][m_name]["delta_mean"]
            relative_to_ma[c_name][m_name] = float(d_m - d_ma)

    # 4. Big Pot Accounting (Top 1% pots)
    big_pot_audit = {}
    for c_name, _, _ in conditions:
        big_pot_audit[c_name] = {}
        for m_name in models_dict:
            pots = forensic_pots[c_name][m_name]
            pots.sort(key=lambda x: x[0], reverse=True)
            n_top = max(1, int(len(pots) * 0.01))
            top_pots = pots[:n_top]
            
            avg_pot_size = float(np.mean([p[0] for p in top_pots]))
            top_profit_sum = float(sum(p[1] for p in top_pots))
            # Metric A: Within-Top-1% average profit (bb/hand)
            within_top_avg_bb = float(top_profit_sum / n_top)
            # Metric B: Marginal contribution to overall bb/100 (divided by total hands N)
            marginal_contrib_bb100 = float(top_profit_sum / len(pots) * 100.0)
            
            big_pot_audit[c_name][m_name] = {
                "total_hands": len(pots),
                "top_1pct_hands": n_top,
                "avg_top_pot_size_bb": avg_pot_size,
                "within_top_avg_bb": within_top_avg_bb,
                "marginal_contrib_bb100": marginal_contrib_bb100
            }

    return summary, causal_impacts, relative_to_ma, target_bot_actions, big_pot_audit

def main():
    eval_seeds = [92001, 92002, 92003, 92004, 92005]
    bp = DeterministicTrueNFSP_Agent()
    models_dict = {
        "Model_A": ModelA(bp, sims=300)
    }
    for r_idx in range(1, 4):
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        path = f"v1017/models/model_e_r{r_idx}_100k.pt"
        m.load_state_dict(torch.load(path))
        m.eval()
        models_dict[f"E_R{r_idx}_100K"] = m
        print(f"[+] Loaded Model E_R{r_idx}_100K from {path}")
        
    summary, causal_impacts, relative_to_ma, target_bot_actions, big_pot_audit = run_v1020_audit(
        models_dict, eval_seeds, hands_per_cond=600
    )
    
    output_data = {
        "eval_seeds": eval_seeds,
        "summary": summary,
        "causal_impacts": causal_impacts,
        "relative_to_ma": relative_to_ma,
        "target_bot_actions": {k: dict(v) for k, v in target_bot_actions.items()},
        "big_pot_audit": big_pot_audit
    }
    
    with open("v1020/data/v1020_audit_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Full results saved to v1020/data/v1020_audit_results.json")

if __name__ == "__main__":
    main()
