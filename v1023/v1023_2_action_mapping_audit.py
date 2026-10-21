import os
import sys
import json
import time
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v75.run_v75_true_6action_and_fold_recovery import (
    get_5p_fullgame, advance_chance_nodes, extract_player_spent, True6ActionAdapter,
)
from v79.run_v79_controlled_causal_study import normalize_inf_state_v79
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1021.run_v1021_anti_limper_league import LooseLimperBot

# =====================================================================
# AGENT WRAPPERS WITH EXACT ACTION DEFINITIONS
# =====================================================================
# Action 0: Fold
# Action 1: Check/Call
# Action 2: HalfPot (0.50 Pot target)
# Action 3: Pot (1.00 Pot target)
# Action 4: DoublePot (2.00 Pot target - Oversized Overbet)
# Action 5: Allin (Stack 10,000)

EXACT_ACTION_NAMES = {
    0: "Fold",
    1: "CheckCall",
    2: "HalfPot_0.5x",
    3: "Pot_1.0x",
    4: "DoublePot_2.0x",
    5: "Allin"
}

class PPOAgentWrapper:
    def __init__(self, model, mode="sample"):
        self.model = model
        self.mode = mode
        
    def act(self, inf, legal_actions, rng):
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
            
            if self.mode == "argmax":
                return int(torch.argmax(masked_logits[0]).item())
            else:
                probs = torch.softmax(masked_logits[0], dim=-1).cpu().numpy()
                probs = probs / np.sum(probs)
                return int(rng.choice(len(probs), p=probs))

class EnsembleAgentWrapper:
    def __init__(self, models, mode="sample"):
        self.models = models
        self.mode = mode
        
    def act(self, inf, legal_actions, rng):
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
        avg_probs = avg_probs / np.sum(avg_probs)
        
        if self.mode == "argmax":
            return int(np.argmax(avg_probs))
        else:
            return int(rng.choice(len(avg_probs), p=avg_probs))

def main():
    print("=================================================================")
    print("POKER AI V10.23.2: ACTION MAPPING AUDIT & PAIRED COMPARISON")
    print("=================================================================")

    # 1. Load trained AntiLimper models
    anti_models = []
    for r in range(1, 4):
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)
        
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = ModelA(bp_agent, sims=300)
    
    agents = {
        "R1_Sample": PPOAgentWrapper(anti_models[0], mode="sample"),
        "R1_Argmax": PPOAgentWrapper(anti_models[0], mode="argmax"),
        "R2_Sample": PPOAgentWrapper(anti_models[1], mode="sample"),
        "R2_Argmax": PPOAgentWrapper(anti_models[1], mode="argmax"),
        "R3_Sample": PPOAgentWrapper(anti_models[2], mode="sample"),
        "R3_Argmax": PPOAgentWrapper(anti_models[2], mode="argmax"),
        "Ensemble_Sample": EnsembleAgentWrapper(anti_models, mode="sample"),
        "Ensemble_Argmax": EnsembleAgentWrapper(anti_models, mode="argmax"),
        "Model_A_MC300": model_a,
    }

    NUM_HANDS = 1000
    eval_seeds = [105000 + i for i in range(NUM_HANDS)]
    game = get_5p_fullgame()
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s*13),
        DeterministicTightPassiveBot(seed=s*17),
        DeterministicUltraAggressiveBot(seed=s*19),
        LooseLimperBot(seed=s*23)
    ]

    hand_returns = {name: [] for name in agents}
    action_counts = {name: defaultdict(int) for name in agents}
    engine_bet_amounts = {name: defaultdict(list) for name in agents} # actual chips committed by action
    
    # Check vs Call Opportunity Tracking
    check_call_audit = {
        name: {
            "opportunities_to_check": 0, # to_call == 0 when hero acts
            "chose_check": 0,           # chose action 1 when to_call == 0
            "chose_bet_when_can_check": 0, # chose action 2,3,4,5 when to_call == 0
            "chose_fold_when_can_check": 0,
            "opportunities_to_call": 0,  # to_call > 0 when hero acts
            "chose_call": 0,            # chose action 1 when to_call > 0
            "chose_raise": 0,           # chose action 2,3,4,5 when to_call > 0
            "chose_fold": 0             # chose action 0 when to_call > 0
        } for name in agents
    }

    traces_file = open("v1023/data/v1023_2_hand_traces.jsonl", "w", encoding="utf-8")
    
    print(f"[*] Executing 1,000 paired hands with comprehensive action mapping tracing...")
    t0 = time.time()

    for h_idx, seed_h in enumerate(eval_seeds):
        if (h_idx + 1) % 200 == 0:
            print(f"    Hand {h_idx+1}/{NUM_HANDS} ({time.time()-t0:.1f}s)")
            
        seat = h_idx % 5
        opps = opp_factory(seed_h)
        
        for name, agent in agents.items():
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            f_set = set()
            dec = 0
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            hand_trace = {
                "hand_id": h_idx,
                "seed": seed_h,
                "seat": seat,
                "agent": name,
                "decisions": []
            }
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                spent = extract_player_spent(st)
                to_call = max(spent) - spent[cp]
                
                if cp == seat:
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    if isinstance(agent, ModelA):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                        
                    raw_engine_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                    # Track exact Check vs Call opportunities
                    cca = check_call_audit[name]
                    if to_call == 0:
                        cca["opportunities_to_check"] += 1
                        if act == 1:
                            cca["chose_check"] += 1
                            detailed_name = "FreeCheck"
                        elif act in [2, 3, 4, 5]:
                            cca["chose_bet_when_can_check"] += 1
                            detailed_name = f"Bet_{EXACT_ACTION_NAMES[act]}"
                        else:
                            cca["chose_fold_when_can_check"] += 1
                            detailed_name = "Fold_When_Free"
                    else:
                        cca["opportunities_to_call"] += 1
                        if act == 1:
                            cca["chose_call"] += 1
                            detailed_name = "PaidCall"
                        elif act in [2, 3, 4, 5]:
                            cca["chose_raise"] += 1
                            detailed_name = f"Raise_{EXACT_ACTION_NAMES[act]}"
                        else:
                            cca["chose_fold"] += 1
                            detailed_name = "Fold_Facing_Bet"
                            
                    action_counts[name][detailed_name] += 1
                    action_counts[name][EXACT_ACTION_NAMES[act]] += 1
                    engine_bet_amounts[name][EXACT_ACTION_NAMES[act]].append(raw_engine_act)
                    
                    hand_trace["decisions"].append({
                        "dec": dec,
                        "abstract_action": act,
                        "exact_name": EXACT_ACTION_NAMES[act],
                        "to_call": to_call,
                        "engine_amount": raw_engine_act
                    })
                    dec += 1
                else:
                    opp_idx = cp if cp < seat else cp - 1
                    bot = opps[opp_idx]
                    if isinstance(bot, DeterministicTrueNFSP_Agent):
                        inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h*101 + cp*17))
                    else:
                        act = bot.select_action(st, cp, l6)
                    raw_engine_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_engine_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            hand_trace["return_bb"] = ret_bb
            
            # Write hand trace for R1, R3, Ensemble, Model_A (to avoid massive file)
            if name in ["R1_Argmax", "R3_Argmax", "Ensemble_Argmax", "Model_A_MC300"]:
                traces_file.write(json.dumps(hand_trace) + "\n")

    traces_file.close()
    print(f"[+] Traces written. Processing comprehensive paired stats...")

    # 2. Paired t-test and rigorous comparison
    # Compare each against baseline and compare Argmax vs Sample
    results_dict = {}
    
    for name in agents:
        ret_arr = np.array(hand_returns[name])
        mean_bb = float(np.mean(ret_arr) * 100.0)
        median_bb = float(np.median(ret_arr) * 100.0)
        std_bb = float(np.std(ret_arr) * 100.0)
        sem_bb = float(stats.sem(ret_arr) * 100.0)
        ci95_low = mean_bb - 1.96 * sem_bb
        ci95_high = mean_bb + 1.96 * sem_bb
        
        # Action breakdown
        tot_dec = sum(action_counts[name][EXACT_ACTION_NAMES[a]] for a in range(6))
        pcts = {
            EXACT_ACTION_NAMES[a]: (action_counts[name][EXACT_ACTION_NAMES[a]] / tot_dec * 100.0) if tot_dec else 0.0
            for a in range(6)
        }
        
        # Average engine action amounts
        avg_engine_amounts = {
            act_name: float(np.mean(amounts)) if amounts else 0.0
            for act_name, amounts in engine_bet_amounts[name].items()
        }
        
        results_dict[name] = {
            "mean_bb100": mean_bb,
            "median_bb100": median_bb,
            "std_bb100": std_bb,
            "sem_bb100": sem_bb,
            "ci95": [ci95_low, ci95_high],
            "action_percentages": pcts,
            "action_counts": dict(action_counts[name]),
            "check_call_audit": check_call_audit[name],
            "avg_engine_amounts": avg_engine_amounts
        }

    # 3. Direct Paired Comparisons
    paired_comparisons = {}
    
    # A. Argmax vs Sampling on same weights
    for r in ["R1", "R2", "R3", "Ensemble"]:
        sample_ret = np.array(hand_returns[f"{r}_Sample"])
        argmax_ret = np.array(hand_returns[f"{r}_Argmax"])
        diff = (argmax_ret - sample_ret) * 100.0
        t_stat, p_val = stats.ttest_rel(argmax_ret, sample_ret)
        paired_comparisons[f"{r}: Argmax vs Sample"] = {
            "mean_diff_bb100": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "ci95_diff": [float(np.mean(diff) - 1.96 * stats.sem(diff)), float(np.mean(diff) + 1.96 * stats.sem(diff))]
        }
        
    # B. Ensemble vs 3-Model Arithmetic Average
    # Sample
    mean_3ind_sample = (np.array(hand_returns["R1_Sample"]) + np.array(hand_returns["R2_Sample"]) + np.array(hand_returns["R3_Sample"])) / 3.0
    ens_sample = np.array(hand_returns["Ensemble_Sample"])
    diff_ens_sample = (ens_sample - mean_3ind_sample) * 100.0
    t_stat_es, p_val_es = stats.ttest_rel(ens_sample, mean_3ind_sample)
    paired_comparisons["Ensemble vs Mean(3 Ind) [Sample]"] = {
        "mean_diff_bb100": float(np.mean(diff_ens_sample)),
        "std_diff": float(np.std(diff_ens_sample)),
        "t_statistic": float(t_stat_es),
        "p_value": float(p_val_es),
        "ci95_diff": [float(np.mean(diff_ens_sample) - 1.96 * stats.sem(diff_ens_sample)), float(np.mean(diff_ens_sample) + 1.96 * stats.sem(diff_ens_sample))]
    }

    # Argmax
    mean_3ind_argmax = (np.array(hand_returns["R1_Argmax"]) + np.array(hand_returns["R2_Argmax"]) + np.array(hand_returns["R3_Argmax"])) / 3.0
    ens_argmax = np.array(hand_returns["Ensemble_Argmax"])
    diff_ens_argmax = (ens_argmax - mean_3ind_argmax) * 100.0
    t_stat_ea, p_val_ea = stats.ttest_rel(ens_argmax, mean_3ind_argmax)
    paired_comparisons["Ensemble vs Mean(3 Ind) [Argmax]"] = {
        "mean_diff_bb100": float(np.mean(diff_ens_argmax)),
        "std_diff": float(np.std(diff_ens_argmax)),
        "t_statistic": float(t_stat_ea),
        "p_value": float(p_val_ea),
        "ci95_diff": [float(np.mean(diff_ens_argmax) - 1.96 * stats.sem(diff_ens_argmax)), float(np.mean(diff_ens_argmax) + 1.96 * stats.sem(diff_ens_argmax))]
    }

    final_output = {
        "agents": results_dict,
        "paired_comparisons": paired_comparisons
    }

    with open("v1023/data/v1023_2_paired_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("[+] Saved v1023/data/v1023_2_paired_results.json successfully.")

if __name__ == "__main__":
    main()
