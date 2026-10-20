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
    get_pool_standard, DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

# =====================================================================
# UNSEEN OPPONENT 1: Adaptive History Exploiter
# Tracks recent fold/aggression rates of opponents to exploit them
# =====================================================================
class AdaptiveHistoryExploiter:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.opponent_folds = deque(maxlen=window_size)
        self.opponent_aggressions = deque(maxlen=window_size)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        
        # Estimate recent fold tendency
        fold_rate = np.mean(self.opponent_folds) if len(self.opponent_folds) >= 10 else 0.5
        
        if to_call == 0: # Free check or bet
            if fold_rate > 0.6: # Opponent folds easily -> Steal aggressively
                if 4 in legal_actions: return 4 # Pot bet
                if 3 in legal_actions: return 3
                if 2 in legal_actions: return 2
            # Otherwise balanced probe
            if 3 in legal_actions and n_b >= 3:
                return 3
            return 1 # Check
        else: # Facing a bet
            pot = sum(sp)
            pot_odds = to_call / (pot + to_call)
            # Hand evaluation: returns tuple (category, ...) where 0=StraightFlush ... 8=HighCard
            h_cards = list(np.where(inf[5:57] == 1.0)[0])
            b_cards = list(np.where(inf[57:109] == 1.0)[0])
            all_c = h_cards + b_cards
            cat = eval_7cards(all_c)[0] if len(all_c) >= 5 else 7 # 7 = One Pair default
            
            # If hand is reasonable (category <= 6 means Two Pair or better)
            if cat <= 6:
                if 4 in legal_actions and cat <= 3: return 4 # Monster
                if 1 in legal_actions: return 1
            if to_call < 200 and 1 in legal_actions:
                return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# UNSEEN OPPONENT 2: Dynamic Multi-Sizing Bot
# Uses precise multi-street sizing (Min, Half-pot, Pot, Allin) based on board & texture
# =====================================================================
class DynamicMultiSizingBot:
    def __init__(self, seed=54321):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        h_cards = list(np.where(inf[5:57] == 1.0)[0])
        b_cards = list(np.where(inf[57:109] == 1.0)[0])
        all_c = h_cards + b_cards
        cat = eval_7cards(all_c)[0] if len(all_c) >= 5 else 7
        
        if to_call == 0:
            if cat <= 2: # Monster (Full House+) -> Pot or AllIn
                if 4 in legal_actions: return 4
                if 5 in legal_actions: return 5
            elif cat <= 5: # Good hand (Straight, Three-of-a-kind) -> Half pot
                if 3 in legal_actions: return 3
            elif cat <= 6: # Two Pair / Pair -> Min bet probe
                if 2 in legal_actions: return 2
            return 1 # Check
        else:
            if cat <= 2: # Nut hand -> Raise or Allin
                if 4 in legal_actions: return 4
                if 5 in legal_actions: return 5
            if cat <= 6:
                return 1 if 1 in legal_actions else 0
            if to_call <= 100 and 1 in legal_actions:
                return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# UNSEEN OPPONENT 3: Polarized GTO Bluff/Value Controller
# Polarizes range: bets huge with monsters and pure bluffs, checks mid-strength
# =====================================================================
class PolarizedBluffValueBot:
    def __init__(self, seed=65432):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        h_cards = list(np.where(inf[5:57] == 1.0)[0])
        b_cards = list(np.where(inf[57:109] == 1.0)[0])
        all_c = h_cards + b_cards
        cat = eval_7cards(all_c)[0] if len(all_c) >= 5 else 7
        
        is_monster = (cat <= 3) # Flush or better
        is_air = (cat == 8)     # High card
        roll = self.rng.rand()
        
        if to_call == 0:
            # Monster: 70% pot bet, 30% check-trap
            if is_monster:
                if roll < 0.70 and 4 in legal_actions: return 4
                return 1
            # Pure air bluff: 25% pot bet bluff
            if is_air:
                if roll < 0.25 and 4 in legal_actions: return 4
                return 1
            # Medium hands: pure check
            return 1
        else:
            if is_monster:
                if 4 in legal_actions: return 4
                if 5 in legal_actions and roll < 0.4: return 5
                return 1 if 1 in legal_actions else 0
            if cat <= 6 and 1 in legal_actions:
                return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# UNSEEN OPPONENT 4: Trap & Check-Raise Specialist
# Checks monsters on Flop/Turn to launch devastating check-raises
# =====================================================================
class CheckRaiseTrapBot:
    def __init__(self, seed=76543):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        h_cards = list(np.where(inf[5:57] == 1.0)[0])
        b_cards = list(np.where(inf[57:109] == 1.0)[0])
        all_c = h_cards + b_cards
        cat = eval_7cards(all_c)[0] if len(all_c) >= 5 else 7
        n_b = len(b_cards)
        
        is_monster = (cat <= 3)
        
        if to_call == 0:
            # Trap on Flop/Turn (check instead of bet)
            if is_monster and n_b in [3, 4]:
                return 1 # Slowplay check
            if cat <= 5 and 3 in legal_actions:
                return 3
            return 1
        else: # Facing bet -> Execute Check-Raise!
            if is_monster:
                if 4 in legal_actions: return 4 # Check-Raise Pot!
                if 3 in legal_actions: return 3
                if 5 in legal_actions: return 5
                return 1 if 1 in legal_actions else 0
            if cat <= 6 and 1 in legal_actions:
                return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# POOL GENERATORS
# =====================================================================
def get_seen_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        DeterministicTrueNFSP_Agent()
    ]

def get_unseen_pool(seed_h):
    return [
        AdaptiveHistoryExploiter(window_size=30),
        DynamicMultiSizingBot(seed=seed_h * 23),
        PolarizedBluffValueBot(seed=seed_h * 29),
        CheckRaiseTrapBot(seed=seed_h * 31)
    ]

# =====================================================================
# EVALUATION ENGINE
# =====================================================================
def evaluate_generalization(models_dict, eval_seeds, hands_per_pool=1000):
    game = get_5p_fullgame()
    pools = [("Seen_Pool", get_seen_pool), ("Unseen_Pool", get_unseen_pool)]
    
    # Structure: pool -> model -> seed -> list of returns
    results = {
        p_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict}
        for p_name, _ in pools
    }
    
    # Detailed action tracking for Six-Action semantic audit
    action_records = {m_name: defaultdict(int) for m_name in models_dict}
    
    print("\n=======================================================")
    print("[*] V10.18 UNSEEN OPPONENT GENERALIZATION CHAMPIONSHIP")
    print(f"[*] Models: {list(models_dict.keys())}")
    print(f"[*] Evaluation: {len(eval_seeds)} Seeds x {len(pools)} Pools x {hands_per_pool} Hands = {len(eval_seeds)*len(pools)*hands_per_pool} Hands/Model")
    print("=======================================================")
    
    t0 = time.time()
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_seed = time.time()
        print(f"  -> Progress: Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for p_name, p_gen in pools:
            for h in range(hands_per_pool):
                seed_h = s_base + h
                seat = h % 5
                opps = p_gen(seed_h)
                
                for m_name, model in models_dict.items():
                    st = game.new_initial_state()
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    f_set = set()
                    dec = 0
                    
                    while not st.is_terminal():
                        if st.is_chance_node():
                            advance_chance_nodes(st, np.random.RandomState(seed_h))
                            continue
                        cp = st.current_player()
                        l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                        
                        if cp == seat:
                            inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                            if isinstance(model, ModelA):
                                act = model.act(st, seat, l6, f_set, seed_h, dec)
                            elif isinstance(model, PPONeuralPlayer):
                                act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                            else:
                                act = model.select_action(st, cp, l6)
                            action_records[m_name][act] += 1
                            dec += 1
                        else:
                            opp_idx = cp if cp < seat else cp - 1
                            opp_bot = opps[opp_idx]
                            if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                                inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                                act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(seed_h*101 + cp*17))
                            else:
                                act = opp_bot.select_action(st, cp, l6)
                                
                        if act == 0:
                            f_set.add(cp)
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[p_name][m_name][s_base].append(ret_bb)
                    
        print(f"     [Seed {s_base} finished in {time.time()-t_seed:.1f}s]", flush=True)

    # Compile Summary
    summary = {}
    for p_name, _ in pools:
        summary[p_name] = {}
        print(f"\n--- [{p_name} Results] ---")
        for m_name in models_dict:
            seed_means = [np.mean(results[p_name][m_name][s]) * 100.0 for s in eval_seeds]
            mean_val = float(np.mean(seed_means))
            se_val = float(np.std(seed_means, ddof=1) / np.sqrt(len(eval_seeds)))
            t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
            ci = [float(mean_val - t_crit * se_val), float(mean_val + t_crit * se_val)]
            
            summary[p_name][m_name] = {
                "mean": mean_val,
                "se": se_val,
                "ci_95": ci,
                "seed_means": seed_means
            }
            print(f"  * {m_name:12s}: {mean_val:+7.2f} ± {se_val:5.2f} bb/100 | CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}] | Seeds: {[round(x,1) for x in seed_means]}")
            
    # Model A Delta in Unseen Pool
    print("\n--- [Delta EV vs Model A in UNSEEN POOL] ---")
    ma_unseen = summary["Unseen_Pool"]["Model_A"]["seed_means"]
    delta_vs_ma = {}
    for m_name in models_dict:
        if m_name == "Model_A": continue
        m_unseen = summary["Unseen_Pool"][m_name]["seed_means"]
        diffs = np.array(m_unseen) - np.array(ma_unseen)
        d_mean = float(np.mean(diffs))
        d_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        t_stat, p_val = stats.ttest_rel(m_unseen, ma_unseen)
        t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
        ci_diff = [float(d_mean - t_crit * d_se), float(d_mean + t_crit * d_se)]
        
        delta_vs_ma[m_name] = {
            "d_mean": d_mean,
            "d_se": d_se,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "ci_95": ci_diff
        }
        print(f"  * {m_name:12s} - Model_A: {d_mean:+7.2f} ± {d_se:5.2f} bb/100 (95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}], t={t_stat:.3f}, p={p_val:.4f})")

    # Action Semantic Summary
    action_table = {}
    for m_name, counts in action_records.items():
        tot = sum(counts.values())
        action_table[m_name] = {
            act: {"count": counts[act], "pct": float(counts[act] / tot * 100.0)}
            for act in range(6)
        }

    return summary, delta_vs_ma, action_table, results

def main():
    eval_seeds = [91001, 91002, 91003, 91004, 91005]
    
    # Load Frozen 4 Models
    bp = DeterministicTrueNFSP_Agent()
    eval_models = {
        "Model_A": ModelA(bp, sims=300)
    }
    
    for r_idx in range(1, 4):
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        path = f"v1017/models/model_e_r{r_idx}_100k.pt"
        m.load_state_dict(torch.load(path))
        m.eval()
        eval_models[f"E_R{r_idx}_100K"] = m
        print(f"[+] Loaded Frozen Model E_R{r_idx}_100K from {path}")
        
    summary, delta_vs_ma, action_table, _ = evaluate_generalization(
        eval_models, eval_seeds, hands_per_pool=1000
    )
    
    # Save results to JSON
    output_data = {
        "eval_seeds": eval_seeds,
        "summary": summary,
        "delta_vs_ma": delta_vs_ma,
        "action_table": action_table
    }
    
    with open("v1018/data/v1018_unseen_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Full results saved to v1018/data/v1018_unseen_results.json")

if __name__ == "__main__":
    main()
