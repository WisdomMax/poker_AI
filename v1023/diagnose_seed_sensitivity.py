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
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1020.run_v1020_experiment_integrity_audit import (
    RigorousCheckRaiseTrapBot, RigorousPolarizedBluffValueBot,
)
from v1021.run_v1021_anti_limper_league import LooseLimperBot

# =====================================================================
# ENSEMBLE MODEL DEFINITION
# =====================================================================
class EnsemblePPONeuralPlayer:
    """Averages softmax action probabilities of multiple independent PPO models"""
    def __init__(self, models):
        self.models = models
        
    def act_eval(self, inf, legal_actions, rng=None):
        probs_list = []
        norm_inf = normalize_inf_state_v79(inf)
        t_inf = torch.tensor(norm_inf, dtype=torch.float32).unsqueeze(0)
        
        for m in self.models:
            m.eval()
            with torch.no_grad():
                logits, _ = m.forward(t_inf)
                mask = torch.ones(6, dtype=torch.bool)
                for a in range(6):
                    if a not in legal_actions:
                        mask[a] = False
                masked_logits = logits.clone()
                masked_logits[0, ~mask] = -1e9
                p = torch.softmax(masked_logits, dim=-1).cpu().numpy()[0]
                probs_list.append(p)
                
        avg_probs = np.mean(probs_list, axis=0)
        # Choose best action or sample
        return int(np.argmax(avg_probs))

def main():
    print("=================================================================")
    print("POKER AI V10.23: SEED VARIANCE FORENSICS & ENSEMBLE HYBRIDIZATION")
    print("=================================================================")

    # 1. Load models
    ctrl_models = []
    anti_models = []
    for r in range(1, 4):
        m_c = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_c.load_state_dict(torch.load(f"v1022/models/StrictControl_R{r}_100K.pt"))
        m_c.eval()
        ctrl_models.append(m_c)
        
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)

    ensemble_anti = EnsemblePPONeuralPlayer(anti_models)
    bp_agent = DeterministicTrueNFSP_Agent()

    # 2. Action distribution profile on Strict_Test_Limper
    game = get_5p_fullgame()
    eval_seeds = [98001, 98002, 98003] # Fast forensic profiling
    
    opp_factory = lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                             DeterministicUltraAggressiveBot(seed=s*19), LooseLimperBot(seed=s*23)]
    
    test_models = {
        "Control_R1 (Seed 101)": ctrl_models[0],
        "Control_R2 (Seed 202)": ctrl_models[1],
        "Control_R3 (Seed 303)": ctrl_models[2],
        "AntiLimper_R1 (Seed 101 - Failed)": anti_models[0],
        "AntiLimper_R2 (Seed 202 - Success)": anti_models[1],
        "AntiLimper_R3 (Seed 303 - Success)": anti_models[2],
        "AntiLimper_Ensemble (3 Seeds)": ensemble_anti,
        "Model_A (Equity Baseline)": ModelA(bp_agent, sims=200)
    }

    action_stats = {m_name: defaultdict(int) for m_name in test_models}
    total_decisions = {m_name: 0 for m_name in test_models}
    profits = {m_name: [] for m_name in test_models}

    print("\n[*] Profiling 400 hands per model under Strict Limper condition...")
    for s_base in eval_seeds:
        for h in range(400):
            seed_h = s_base + h
            seat = h % 5
            opps = opp_factory(seed_h)
            
            for m_name, model in test_models.items():
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
                        elif isinstance(model, EnsemblePPONeuralPlayer): act = model.act_eval(inf, l6)
                        else: act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                        action_stats[m_name][act] += 1
                        total_decisions[m_name] += 1
                        dec += 1
                    else:
                        opp_idx = cp if cp < seat else cp - 1
                        bot = opps[opp_idx]
                        if isinstance(bot, DeterministicTrueNFSP_Agent):
                            inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h*101 + cp*17))
                        else:
                            act = bot.select_action(st, cp, l6)
                            
                    if act == 0: f_set.add(cp)
                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                    
                ret_bb = st.returns()[seat] / 100.0
                profits[m_name].append(ret_bb)

    print("\n--- FORENSIC ACTION PROFILES & PROFIT RESULTS ---")
    results_summary = {}
    act_names = {0: "Fold", 1: "Check/Call", 2: "MinRaise", 3: "HalfPot", 4: "Pot", 5: "Allin"}

    for m_name in test_models:
        tot = total_decisions[m_name]
        mean_ev = np.mean(profits[m_name]) * 100.0
        fold_pct = (action_stats[m_name][0] / tot) * 100.0 if tot else 0
        call_pct = (action_stats[m_name][1] / tot) * 100.0 if tot else 0
        raise_pct = (sum(action_stats[m_name][a] for a in [2,3,4,5]) / tot) * 100.0 if tot else 0
        
        results_summary[m_name] = {
            "mean_bb100": float(mean_ev),
            "fold_pct": float(fold_pct),
            "call_pct": float(call_pct),
            "raise_pct": float(raise_pct),
            "action_counts": {act_names[k]: v for k, v in action_stats[m_name].items()}
        }
        print(f"[{m_name:34s}] EV = {mean_ev:+7.2f} bb/100 | Fold={fold_pct:4.1f}% | Call={call_pct:4.1f}% | Raise={raise_pct:4.1f}%")

    with open("v1023/data/v1023_forensic_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n[+] Forensic results saved to v1023/data/v1023_forensic_results.json")

if __name__ == "__main__":
    main()
