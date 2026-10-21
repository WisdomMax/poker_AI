import os
import sys
import json
import time
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v75.run_v75_true_6action_and_fold_recovery import (
    get_5p_fullgame, extract_player_spent, True6ActionAdapter,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v79.run_v79_controlled_causal_study import normalize_inf_state_v79
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1021.run_v1021_anti_limper_league import LooseLimperBot

def safe_float(val):
    if val is None or np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)

def advance_chance_nodes_persistent(state, rng):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = rng.choice(actions, p=probs)
        state.apply_action(action)

class RealisticAggressiveBot:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        
    def select_action(self, st, cp, l6):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        if tc > 0:
            r = [a for a in l6 if a in [2, 3]]
            if r and self.rng.random() < 0.70:
                return self.rng.choice(r)
            elif 5 in l6 and self.rng.random() < 0.05:
                return 5
            return 1 if 1 in l6 else (0 if 0 in l6 else 1)
        else:
            b = [a for a in l6 if a in [2, 3]]
            if b and self.rng.random() < 0.75:
                return self.rng.choice(b)
            return 1

class PPOAgentWrapper:
    def __init__(self, model, mode="argmax"):
        self.model = model
        self.mode = mode
        
    def act(self, inf, legal_actions, rng=None):
        self.model.eval()
        with torch.no_grad():
            norm_inf = normalize_inf_state_v79(inf)
            t_inf = torch.tensor(norm_inf, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model.forward(t_inf)
            mask = torch.ones(6, dtype=torch.bool)
            for a in range(6):
                if a not in legal_actions:
                    mask[a] = False
            masked_logits = logits.clone()
            masked_logits[0, ~mask] = -1e9
            return int(torch.argmax(masked_logits[0]).item())

class EnsembleAgentWrapper:
    def __init__(self, models):
        self.models = models
        
    def act(self, inf, legal_actions, rng=None):
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
                p = torch.softmax(masked_logits[0], dim=-1).cpu().numpy()
                probs_list.append(p)
                
        avg_probs = np.mean(probs_list, axis=0)
        return int(np.argmax(avg_probs))

# =====================================================================
# HYBRID AGENT: PPO Preflop Intuition + Model A Postflop Equity Guard
# =====================================================================
class HybridPPOModelAAgent:
    """Uses PPO (R2) for preflop decisions, and Model A (Equity Guard) for postflop decisions"""
    def __init__(self, preflop_agent, postflop_model_a):
        self.preflop_agent = preflop_agent
        self.postflop_model_a = postflop_model_a
        
    def act(self, st, seat, l6, f_set, seed_h, dec):
        rnd = get_round_from_state(st)
        if rnd == 0: # Preflop: use PPO
            inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
            return self.preflop_agent.act(inf, l6)
        else: # Postflop (Flop/Turn/River): use Model A
            return self.postflop_model_a.act(st, seat, l6, f_set, seed_h, dec)

def main():
    print("=================================================================")
    print("POKER AI V10.23.7: STATE-MATCHED POSTFLOP BENCHMARK & HYBRID TEST")
    print("=================================================================")

    # 1. Load models
    anti_models = []
    for r in range(1, 4):
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)
        
    # Clarification as per ChatGPT instruction: Untrained/Randomly Initialized Base Policy for Model A
    bp_agent = DeterministicTrueNFSP_Agent() # RandomlyInitializedNFSP_Base
    model_a = ModelA(bp_agent, sims=300)
    
    r1 = PPOAgentWrapper(anti_models[0], mode="argmax")
    r2 = PPOAgentWrapper(anti_models[1], mode="argmax")
    r3 = PPOAgentWrapper(anti_models[2], mode="argmax")
    ens = EnsembleAgentWrapper(anti_models)
    hybrid = HybridPPOModelAAgent(preflop_agent=r2, postflop_model_a=model_a)
    
    agents = {
        "R1_Argmax": r1,
        "R2_Argmax": r2,
        "R3_Argmax": r3,
        "Ensemble_Argmax": ens,
        "Model_A_MC300": model_a,
        "Hybrid_R2_ModelA": hybrid
    }

    game = get_5p_fullgame()
    
    # 2. PHASE 2: Collect 200 identical postflop states (Flop entry states)
    # Using Realistic League opponents (Balanced, TightPassive, RealisticAggressive, LooseLimper)
    print("\n[*] Phase 2: Generating 200 standardized, identical postflop flop states...")
    
    flop_states_data = [] # List of tuples: (hand_id, seed_h, seat, initial_state_at_flop, preflop_spent_at_flop, chance_rng_state)
    
    seed_candidate = 150000
    collected = 0
    target_flop_states = 200
    
    while collected < target_flop_states:
        seed_candidate += 1
        seat = collected % 5
        
        # Standard realistic preflop play
        opps = [
            DeterministicBalancedBot(rng_seed=seed_candidate * 13),
            DeterministicTightPassiveBot(seed=seed_candidate * 17),
            RealisticAggressiveBot(seed=seed_candidate * 19),
            LooseLimperBot(seed=seed_candidate * 23)
        ]
        
        chance_rng = np.random.RandomState(seed_candidate)
        st = game.new_initial_state()
        advance_chance_nodes_persistent(st, chance_rng)
        
        f_set = set()
        dec = 0
        hero_folded = False
        allin_happened = False
        
        while not st.is_terminal():
            rnd = get_round_from_state(st)
            if rnd > 0: # Flop reached!
                break
                
            if st.is_chance_node():
                advance_chance_nodes_persistent(st, chance_rng)
                continue
                
            cp = st.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st, cp)
            spent = extract_player_spent(st)
            
            if max(spent) >= 9900:
                allin_happened = True
                break
                
            if cp == seat:
                # Hero standard preflop line: call or raise standard to see flop
                # Call (1) or MinRaise (2)
                act = 1 if 1 in l6 else 0
                if act == 0:
                    hero_folded = True
                    break
            else:
                opp_idx = cp if cp < seat else cp - 1
                bot = opps[opp_idx]
                act = bot.select_action(st, cp, l6)
                if act == 5:
                    allin_happened = True
                    break
                    
            if act == 0:
                f_set.add(cp)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            st.apply_action(raw_act)
            
        # Check if legal flop state reached without all-in and hero active
        if not hero_folded and not allin_happened and not st.is_terminal():
            rnd = get_round_from_state(st)
            if rnd == 1: # Flop reached!
                # Clone the state and RNG
                spent_at_flop = extract_player_spent(st)[seat]
                flop_states_data.append({
                    "flop_id": collected,
                    "seed": seed_candidate,
                    "seat": seat,
                    "history": list(st.history()),
                    "spent_at_flop": spent_at_flop,
                    "pot_at_flop": sum(extract_player_spent(st))
                })
                collected += 1
                
    print(f"    [+] Successfully collected {len(flop_states_data)} standardized flop states.")
    print(f"    Average pot at flop: {np.mean([d['pot_at_flop'] for d in flop_states_data]):.1f} chips.")

    # 3. PHASE 3: Evaluate all agents on EXACT SAME 200 Flop States
    print("\n[*] Phase 3: Evaluating all 6 candidates on IDENTICAL 200 flop states...")
    
    postflop_delta_chips = {name: [] for name in agents} # Final returns minus preflop spent
    postflop_final_returns = {name: [] for name in agents}
    postflop_actions_count = {name: defaultdict(int) for name in agents}
    
    t0 = time.time()
    for item in flop_states_data:
        f_id = item["flop_id"]
        seed_h = item["seed"]
        seat = item["seat"]
        hist = item["history"]
        spent_preflop = item["spent_at_flop"]
        
        for name, agent in agents.items():
            # FRESH OPPONENTS
            opps = [
                DeterministicBalancedBot(rng_seed=seed_h * 13),
                DeterministicTightPassiveBot(seed=seed_h * 17),
                RealisticAggressiveBot(seed=seed_h * 19),
                LooseLimperBot(seed=seed_h * 23)
            ]
            
            # Reconstruct exact state at flop
            chance_rng = np.random.RandomState(seed_h)
            st = game.new_initial_state()
            for a in hist:
                st.apply_action(a)
                
            f_set = set()
            dec = 0
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                
                if cp == seat:
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    if isinstance(agent, ModelA) or isinstance(agent, HybridPPOModelAAgent):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    postflop_actions_count[name][act] += 1
                    dec += 1
                else:
                    opp_idx = cp if cp < seat else cp - 1
                    bot = opps[opp_idx]
                    act = bot.select_action(st, cp, l6)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_act)
                
            final_ret_chips = st.returns()[seat]
            delta_chips = final_ret_chips + spent_preflop # Net profit earned specifically in postflop
            
            postflop_final_returns[name].append(final_ret_chips / 100.0)
            postflop_delta_chips[name].append(delta_chips / 100.0)

    print(f"    Completed Phase 3 in {time.time()-t0:.1f}s.")

    # 4. Statistical Compilation & Paired Hypotheses Testing
    summary_results = {}
    action_names = {0: "Fold", 1: "CheckCall", 2: "HalfPot_0.5x", 3: "Pot_1.0x", 4: "DoublePot_2.0x", 5: "Allin"}
    
    for name in agents:
        delta_arr = np.array(postflop_delta_chips[name]) * 100.0 # in bb/100
        mean_delta = safe_float(np.mean(delta_arr))
        sem_delta = safe_float(stats.sem(delta_arr))
        
        tot_acts = sum(postflop_actions_count[name].values())
        act_pcts = {
            action_names[a]: safe_float(postflop_actions_count[name][a] / tot_acts * 100.0) if tot_acts else 0.0
            for a in range(6)
        }
        
        summary_results[name] = {
            "mean_postflop_delta_bb100": mean_delta,
            "sem_delta_bb100": sem_delta,
            "ci95_delta_bb100": [mean_delta - 1.96 * sem_delta, mean_delta + 1.96 * sem_delta],
            "total_postflop_decisions": tot_acts,
            "postflop_action_percentages": act_pcts
        }

    # Hypothesis Testing
    # H2: Does Model A outperform PPO on the exact same postflop states?
    delta_model_a = np.array(postflop_delta_chips["Model_A_MC300"]) * 100.0
    delta_r2 = np.array(postflop_delta_chips["R2_Argmax"]) * 100.0
    delta_ens = np.array(postflop_delta_chips["Ensemble_Argmax"]) * 100.0
    delta_hybrid = np.array(postflop_delta_chips["Hybrid_R2_ModelA"]) * 100.0
    
    diff_ma_r2 = delta_model_a - delta_r2
    t_h2, p_h2 = stats.ttest_rel(delta_model_a, delta_r2)
    
    diff_hyb_r2 = delta_hybrid - delta_r2
    t_h3, p_h3 = stats.ttest_rel(delta_hybrid, delta_r2)
    
    paired_hypotheses = {
        "H2: Model A vs R2 Postflop Delta (Same States)": {
            "mean_diff_bb100": safe_float(np.mean(diff_ma_r2)),
            "sem_diff": safe_float(stats.sem(diff_ma_r2)),
            "t_statistic": safe_float(t_h2),
            "p_value": safe_float(p_h2),
            "ci95_diff": [safe_float(np.mean(diff_ma_r2) - 1.96 * stats.sem(diff_ma_r2)), safe_float(np.mean(diff_ma_r2) + 1.96 * stats.sem(diff_ma_r2))]
        },
        "H3: Hybrid (R2+ModelA) vs R2 Alone Postflop Delta": {
            "mean_diff_bb100": safe_float(np.mean(diff_hyb_r2)),
            "sem_diff": safe_float(stats.sem(diff_hyb_r2)),
            "t_statistic": safe_float(t_h3),
            "p_value": safe_float(p_h3),
            "ci95_diff": [safe_float(np.mean(diff_hyb_r2) - 1.96 * stats.sem(diff_hyb_r2)), safe_float(np.mean(diff_hyb_r2) + 1.96 * stats.sem(diff_hyb_r2))]
        }
    }

    final_payload = {
        "target_states_evaluated": target_flop_states,
        "agents": summary_results,
        "paired_hypotheses": paired_hypotheses
    }

    out_path = "v1023/data/v1023_7_state_matched_results.json"
    with open(out_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] V10.23.7 State-Matched Results saved to {out_path}")

if __name__ == "__main__":
    main()
