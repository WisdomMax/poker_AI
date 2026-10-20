import os
import sys
import json
import time
from collections import defaultdict, deque
import numpy as np
import scipy.stats as stats
import torch

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
from v1020.run_v1020_experiment_integrity_audit import (
    get_chen_score, get_comprehensive_hand_strength,
    RigorousDynamicMultiSizingBot, RigorousPolarizedBluffValueBot, RigorousCheckRaiseTrapBot
)

# =====================================================================
# ADAPTIVE BOT WITH TOGGLEABLE ADAPTATION (Ablation Controller)
# =====================================================================
class CausalAdaptiveExploiter:
    def __init__(self, window_size=30, enable_adaptive=True):
        self.window_size = window_size
        self.enable_adaptive = enable_adaptive
        self.hero_folds = deque(maxlen=window_size)
        self.total_steal_attempts = 0
        self.steal_successes = 0
        self.steal_profit_chips = 0.0
        
    def record_hero_action(self, hero_act):
        # 0 is fold
        self.hero_folds.append(1 if hero_act == 0 else 0)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        
        # When adaptive is enabled, track sliding window fold rate
        fold_rate = np.mean(self.hero_folds) if (self.enable_adaptive and len(self.hero_folds) >= 10) else 0.50
        
        if to_call == 0:
            if fold_rate > 0.55: # EXPLOIT TRIGGER: Steal aggressively!
                self.total_steal_attempts += 1
                if 4 in legal_actions: return 4 # Pot steal
                if 3 in legal_actions: return 3 # Half pot steal
            if hs["is_good"] and 3 in legal_actions: return 3
            return 1 # Check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                return 1
            if hs["is_good"] and to_call < 500: return 1
            if to_call <= 100: return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# POOL FACTORIES
# =====================================================================
def get_base_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        DeterministicTrueNFSP_Agent()
    ]

def get_adaptive_on_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        CausalAdaptiveExploiter(window_size=30, enable_adaptive=True)
    ]

def get_adaptive_off_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        CausalAdaptiveExploiter(window_size=30, enable_adaptive=False)
    ]

def get_multisizing_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousDynamicMultiSizingBot(seed=seed_h * 23)
    ]

def get_trap_pool(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        RigorousCheckRaiseTrapBot(seed=seed_h * 31)
    ]

# =====================================================================
# EXECUTION & STEAL TELEMETRY HARNESS
# =====================================================================
def run_v1020_1_audit(models_dict, eval_seeds, hands_per_cond=600):
    game = get_5p_fullgame()
    conditions = [
        ("Base_Seen", get_base_pool),
        ("Adaptive_ON", get_adaptive_on_pool),
        ("Adaptive_OFF", get_adaptive_off_pool),
        ("MultiSizing", get_multisizing_pool),
        ("Trap", get_trap_pool)
    ]
    
    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _ in conditions}
    forensic_pots = {c_name: {m_name: [] for m_name in models_dict} for c_name, _ in conditions}
    
    # Specific Steal Telemetry for Adaptive conditions
    steal_telemetry = {
        c_name: {
            m_name: {
                "steal_attempts": 0,
                "hero_folds_to_steal": 0,
                "steal_profit_bb": 0.0
            }
            for m_name in models_dict
        }
        for c_name in ["Adaptive_ON", "Adaptive_OFF"]
    }
    
    print("\n=======================================================")
    print("[*] V10.20.1 ADAPTIVE CAUSAL VERIFICATION RUN")
    print(f"[*] 4 Models x 5 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print(f"[*] Total Evaluated: {4 * 5 * len(eval_seeds) * hands_per_cond:,} Model-Hands")
    print("=======================================================")
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_s = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen in conditions:
            for h in range(hands_per_cond):
                seed_h = s_base + h
                seat = h % 5
                opps = p_gen(seed_h)
                target_bot = opps[3]
                
                for m_name, model in models_dict.items():
                    st = game.new_initial_state()
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    f_set = set(); dec = 0
                    hero_last_action = None
                    target_stole_this_hand = False
                    
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
                            hero_last_action = act
                            
                            # If target bot stole before, did hero fold?
                            if target_stole_this_hand and act == 0 and c_name in steal_telemetry:
                                steal_telemetry[c_name][m_name]["hero_folds_to_steal"] += 1
                                
                            # Feed hero action to adaptive bot
                            if isinstance(target_bot, CausalAdaptiveExploiter):
                                target_bot.record_hero_action(act)
                        else:
                            opp_idx = cp if cp < seat else cp - 1
                            opp_bot = opps[opp_idx]
                            
                            if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                                inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                                act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(seed_h*101 + cp*17))
                            else:
                                act = opp_bot.select_action(st, cp, l6)
                                
                            # Check if adaptive bot triggered a steal
                            if opp_idx == 3 and isinstance(opp_bot, CausalAdaptiveExploiter):
                                sp_now = extract_player_spent(st)
                                tc_now = max(sp_now) - sp_now[cp]
                                fold_r = np.mean(opp_bot.hero_folds) if (opp_bot.enable_adaptive and len(opp_bot.hero_folds) >= 10) else 0.50
                                if tc_now == 0 and fold_r > 0.55 and act in [3, 4]:
                                    target_stole_this_hand = True
                                    if c_name in steal_telemetry:
                                        steal_telemetry[c_name][m_name]["steal_attempts"] += 1
                                        
                        if act == 0: f_set.add(cp)
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[c_name][m_name][s_base].append(ret_bb)
                    pot_bb = sum(extract_player_spent(st)) / 100.0
                    forensic_pots[c_name][m_name].append((pot_bb, ret_bb))
                    
                    if target_stole_this_hand and c_name in steal_telemetry:
                        # Target bot return (index 3 is cp 4 or 3)
                        target_cp = 3 if 3 < seat else 4
                        t_ret_bb = st.returns()[target_cp] / 100.0
                        steal_telemetry[c_name][m_name]["steal_profit_bb"] += t_ret_bb
                        
        print(f"     [Seed {s_base} completed in {time.time()-t_s:.1f}s]", flush=True)

    # 1. Summary by Condition
    summary = {}
    base_means = {}
    for c_name, _ in conditions:
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
            if c_name == "Base_Seen": base_means[m_name] = s_means

    # 2. Causal Differences: Adaptive_ON vs Adaptive_OFF (The pure effect of adaptation!)
    adaptive_ablation = {}
    for m_name in models_dict:
        on_s = summary["Adaptive_ON"][m_name]["seed_means"]
        off_s = summary["Adaptive_OFF"][m_name]["seed_means"]
        diffs = np.array(on_s) - np.array(off_s)
        d_m = float(np.mean(diffs))
        d_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        t_s, p_v = stats.ttest_rel(on_s, off_s)
        t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
        ci_d = [float(d_m - t_crit * d_se), float(d_m + t_crit * d_se)]
        adaptive_ablation[m_name] = {
            "on_mean": summary["Adaptive_ON"][m_name]["mean"],
            "off_mean": summary["Adaptive_OFF"][m_name]["mean"],
            "adapt_effect_delta": d_m,
            "adapt_effect_se": d_se,
            "t_stat": float(t_s),
            "p_val": float(p_v),
            "ci_95": ci_d
        }

    # 3. Model A Difference-in-Differences (DiD)
    did_analysis = {}
    for c_name in ["Adaptive_ON", "Adaptive_OFF", "MultiSizing", "Trap"]:
        did_analysis[c_name] = {}
        d_ma = summary[c_name]["Model_A"]["mean"] - summary["Base_Seen"]["Model_A"]["mean"]
        for m_name in models_dict:
            if m_name == "Model_A": continue
            d_m = summary[c_name][m_name]["mean"] - summary["Base_Seen"][m_name]["mean"]
            did_analysis[c_name][m_name] = {
                "model_delta": d_m,
                "model_a_delta": d_ma,
                "did_relative_delta": float(d_m - d_ma)
            }

    return summary, adaptive_ablation, did_analysis, steal_telemetry

def main():
    eval_seeds = [93001, 93002, 93003, 93004, 93005]
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
        
    summary, adaptive_ablation, did_analysis, steal_telemetry = run_v1020_1_audit(
        models_dict, eval_seeds, hands_per_cond=600
    )
    
    output_data = {
        "eval_seeds": eval_seeds,
        "summary": summary,
        "adaptive_ablation": adaptive_ablation,
        "did_analysis": did_analysis,
        "steal_telemetry": steal_telemetry
    }
    
    with open("v1020_1/data/v1020_1_causal_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Full results saved to v1020_1/data/v1020_1_causal_results.json")

if __name__ == "__main__":
    main()
