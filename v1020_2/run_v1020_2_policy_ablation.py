import os
import sys
import json
import time
from collections import defaultdict
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
from v1020.run_v1020_experiment_integrity_audit import get_comprehensive_hand_strength

# =====================================================================
# FACTORIZED ABLATION BOTS (Limp vs Probe Decomposition)
# =====================================================================
class FactorizedLimpProbeBot:
    def __init__(self, enable_limp=True, enable_probe=True):
        self.enable_limp = enable_limp
        self.enable_probe = enable_probe
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        
        if to_call == 0:
            # Probe factor: bet half pot if checked to
            if self.enable_probe and hs["is_good"] and 3 in legal_actions:
                return 3 # Half pot probe
            return 1 # Check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                return 1
            if hs["is_good"] and to_call < 500: return 1
            # Limp factor: call 1 BB (to_call <= 100 chips)
            if self.enable_limp and to_call <= 100:
                return 1 # Limp call
            return 0 if 0 in legal_actions else 1

# =====================================================================
# POOL FACTORIES
# =====================================================================
def get_pool_full(seed_h): # Limp ON, Probe ON
    return [DeterministicBalancedBot(rng_seed=seed_h*13), DeterministicTightPassiveBot(seed=seed_h*17),
            DeterministicUltraAggressiveBot(seed=seed_h*19), FactorizedLimpProbeBot(enable_limp=True, enable_probe=True)]

def get_pool_nolimp(seed_h): # Limp OFF, Probe ON
    return [DeterministicBalancedBot(rng_seed=seed_h*13), DeterministicTightPassiveBot(seed=seed_h*17),
            DeterministicUltraAggressiveBot(seed=seed_h*19), FactorizedLimpProbeBot(enable_limp=False, enable_probe=True)]

def get_pool_noprobe(seed_h): # Limp ON, Probe OFF
    return [DeterministicBalancedBot(rng_seed=seed_h*13), DeterministicTightPassiveBot(seed=seed_h*17),
            DeterministicUltraAggressiveBot(seed=seed_h*19), FactorizedLimpProbeBot(enable_limp=True, enable_probe=False)]

def get_pool_none(seed_h): # Limp OFF, Probe OFF
    return [DeterministicBalancedBot(rng_seed=seed_h*13), DeterministicTightPassiveBot(seed=seed_h*17),
            DeterministicUltraAggressiveBot(seed=seed_h*19), FactorizedLimpProbeBot(enable_limp=False, enable_probe=False)]

# =====================================================================
# RUN ABLATION EXPERIMENT
# =====================================================================
def run_policy_ablation(models_dict, eval_seeds, hands_per_cond=500):
    game = get_5p_fullgame()
    conditions = [
        ("Full_Limp_Probe", get_pool_full),
        ("NoLimp_ProbeOnly", get_pool_nolimp),
        ("LimpOnly_NoProbe", get_pool_noprobe),
        ("NoLimp_NoProbe", get_pool_none),
    ]
    
    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _ in conditions}
    
    print("\n=======================================================")
    print("[*] V10.20.2 STATIC POLICY ABLATION (Limp vs Probe)")
    print(f"[*] 4 Models x 4 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print("=======================================================")
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_s = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen in conditions:
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
                            act = opps[opp_idx].select_action(st, cp, l6)
                            
                        if act == 0: f_set.add(cp)
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[c_name][m_name][s_base].append(ret_bb)
                    
        print(f"     [Seed {s_base} completed in {time.time()-t_s:.1f}s]", flush=True)

    summary = {}
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

    # Factor Analysis: Main Effect of Limp & Main Effect of Probe
    factor_effects = {}
    for m_name in models_dict:
        # Limp Effect = Mean(Limp ON) - Mean(Limp OFF)
        limp_on = (summary["Full_Limp_Probe"][m_name]["mean"] + summary["LimpOnly_NoProbe"][m_name]["mean"]) / 2.0
        limp_off = (summary["NoLimp_ProbeOnly"][m_name]["mean"] + summary["NoLimp_NoProbe"][m_name]["mean"]) / 2.0
        limp_effect = limp_on - limp_off # Impact on hero winrate
        
        # Probe Effect = Mean(Probe ON) - Mean(Probe OFF)
        probe_on = (summary["Full_Limp_Probe"][m_name]["mean"] + summary["NoLimp_ProbeOnly"][m_name]["mean"]) / 2.0
        probe_off = (summary["LimpOnly_NoProbe"][m_name]["mean"] + summary["NoLimp_NoProbe"][m_name]["mean"]) / 2.0
        probe_effect = probe_on - probe_off
        
        factor_effects[m_name] = {
            "limp_effect_on_hero": float(limp_effect),
            "probe_effect_on_hero": float(probe_effect)
        }

    return summary, factor_effects

def main():
    eval_seeds = [94001, 94002, 94003, 94004, 94005]
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
        
    summary, factor_effects = run_policy_ablation(models_dict, eval_seeds, hands_per_cond=500)
    
    output_data = {
        "eval_seeds": eval_seeds,
        "summary": summary,
        "factor_effects": factor_effects
    }
    
    with open("v1020_2/data/v1020_2_ablation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Full results saved to v1020_2/data/v1020_2_ablation_results.json")

if __name__ == "__main__":
    main()
