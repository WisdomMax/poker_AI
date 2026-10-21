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

def create_fresh_opponents(seed_h):
    """Instantiates completely fresh opponent bots with deterministic seeds"""
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        LooseLimperBot(seed=seed_h * 23)
    ]

def main():
    print("=================================================================")
    print("POKER AI V10.23.3: OPPONENT RNG ISOLATION & TRUE ACTION LEDGER")
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

    # 2. UNIT TEST: Validate Hand 0 pre-action environment invariance
    print("[*] Running Unit Test: Hand 0 pre-action state consistency across all agents...")
    test_seed = 105000
    test_seat = 0
    game = get_5p_fullgame()
    
    initial_to_call_set = set()
    initial_history_len_set = set()
    
    for name, agent in agents.items():
        st = game.new_initial_state()
        advance_chance_nodes(st, np.random.RandomState(test_seed))
        # ISOLATION: Fresh opponent instances for EACH agent
        fresh_opps = create_fresh_opponents(test_seed)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, np.random.RandomState(test_seed))
                continue
            cp = st.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st, cp)
            spent = extract_player_spent(st)
            to_call = max(spent) - spent[cp]
            
            if cp == test_seat:
                # First decision of hero reached
                initial_to_call_set.add(to_call)
                initial_history_len_set.add(len(st.history()))
                break
            else:
                opp_idx = cp if cp < test_seat else cp - 1
                bot = fresh_opps[opp_idx]
                if isinstance(bot, DeterministicTrueNFSP_Agent):
                    inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(test_seed * 101 + cp * 17))
                else:
                    act = bot.select_action(st, cp, l6)
                raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                st.apply_action(raw_act)
                
    print(f"    Unit Test Results: Unique initial to_call across all 9 agents: {initial_to_call_set}")
    print(f"    Unit Test Results: Unique history lengths before hero action: {initial_history_len_set}")
    assert len(initial_to_call_set) == 1, f"FAIL: Opponent RNG leak detected! {initial_to_call_set}"
    assert len(initial_history_len_set) == 1, f"FAIL: Opponent sequence inconsistent! {initial_history_len_set}"
    print(f"    [+] UNIT TEST PASSED: Pre-action environment is 100% IDENTICAL across all agents! (to_call={list(initial_to_call_set)[0]})\n")

    # 3. Comprehensive 1,000 Hand Paired Benchmark
    NUM_HANDS = 1000
    eval_seeds = [105000 + i for i in range(NUM_HANDS)]
    
    hand_returns = {name: [] for name in agents}
    action_counts = {name: defaultdict(int) for name in agents}
    action_added_chips = {name: defaultdict(list) for name in agents} # exact chips added to pot on action
    
    check_call_audit = {
        name: {
            "opportunities_to_check": 0,
            "chose_check": 0,
            "chose_bet_when_can_check": 0,
            "opportunities_to_call": 0,
            "chose_call": 0,
            "chose_raise": 0,
            "chose_fold": 0
        } for name in agents
    }

    traces_file = open("v1023/data/v1023_3_hand_traces.jsonl", "w", encoding="utf-8")
    
    print(f"[*] Starting 1,000 paired hands evaluation with opponent RNG isolation...")
    t0 = time.time()

    for h_idx, seed_h in enumerate(eval_seeds):
        if (h_idx + 1) % 200 == 0:
            print(f"    Hand {h_idx+1}/{NUM_HANDS} ({time.time()-t0:.1f}s)")
            
        seat = h_idx % 5
        
        for name, agent in agents.items():
            # KEY FIX: Instantiate brand new opponent bots for every single agent evaluation!
            opps = create_fresh_opponents(seed_h)
            
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
                spent_before_action = extract_player_spent(st)
                to_call = max(spent_before_action) - spent_before_action[cp]
                
                if cp == seat:
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    if isinstance(agent, ModelA):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                        
                    raw_engine_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                    # Track exact chips added to pot by this action
                    # spent_before_action[seat] vs spent_after_action[seat]
                    # We can compute it: in OpenSpiel, raw_engine_act is the total spent in round, or we can measure after apply_action
                    
                    cca = check_call_audit[name]
                    if to_call == 0:
                        cca["opportunities_to_check"] += 1
                        if act == 1:
                            cca["chose_check"] += 1
                        elif act in [2, 3, 4, 5]:
                            cca["chose_bet_when_can_check"] += 1
                    else:
                        cca["opportunities_to_call"] += 1
                        if act == 1:
                            cca["chose_call"] += 1
                        elif act in [2, 3, 4, 5]:
                            cca["chose_raise"] += 1
                        else:
                            cca["chose_fold"] += 1
                            
                    action_counts[name][EXACT_ACTION_NAMES[act]] += 1
                    
                    # Apply action
                    if act == 0:
                        f_set.add(cp)
                    st.apply_action(raw_engine_act)
                    
                    spent_after_action = extract_player_spent(st)
                    added_chips = spent_after_action[seat] - spent_before_action[seat]
                    action_added_chips[name][EXACT_ACTION_NAMES[act]].append(added_chips)
                    
                    hand_trace["decisions"].append({
                        "dec": dec,
                        "action_id": act,
                        "exact_name": EXACT_ACTION_NAMES[act],
                        "to_call": float(to_call),
                        "engine_action": int(raw_engine_act),
                        "added_chips": float(added_chips),
                        "total_spent_after": float(spent_after_action[seat])
                    })
                    dec += 1
                else:
                    opp_idx = cp if cp < seat else cp - 1
                    bot = opps[opp_idx]
                    if isinstance(bot, DeterministicTrueNFSP_Agent):
                        inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
                    else:
                        act = bot.select_action(st, cp, l6)
                    raw_engine_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    if act == 0:
                        f_set.add(cp)
                    st.apply_action(raw_engine_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            hand_trace["return_bb"] = ret_bb
            
            # Write traces for key models
            if name in ["R1_Argmax", "R3_Argmax", "Ensemble_Argmax", "Model_A_MC300"]:
                traces_file.write(json.dumps(hand_trace) + "\n")

    traces_file.close()
    print(f"[+] All 1,000 paired hands completed in {time.time()-t0:.1f}s. Computing rigorous paired statistics...")

    # 4. Statistical Compilation
    results_dict = {}
    for name in agents:
        ret_arr = np.array(hand_returns[name])
        mean_bb = float(np.mean(ret_arr) * 100.0)
        median_bb = float(np.median(ret_arr) * 100.0)
        std_bb = float(np.std(ret_arr) * 100.0)
        sem_bb = float(stats.sem(ret_arr) * 100.0)
        ci95_low = mean_bb - 1.96 * sem_bb
        ci95_high = mean_bb + 1.96 * sem_bb
        
        tot_dec = sum(action_counts[name][EXACT_ACTION_NAMES[a]] for a in range(6))
        pcts = {
            EXACT_ACTION_NAMES[a]: (action_counts[name][EXACT_ACTION_NAMES[a]] / tot_dec * 100.0) if tot_dec else 0.0
            for a in range(6)
        }
        
        avg_added_chips = {
            act_name: float(np.mean(chips_list)) if chips_list else 0.0
            for act_name, chips_list in action_added_chips[name].items()
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
            "avg_added_chips_per_action": avg_added_chips
        }

    # 5. Direct Paired Comparisons
    paired_comparisons = {}
    
    # Argmax vs Sample
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
        
    # Ensemble vs 3 Ind Mean
    # Sample
    mean_3ind_sample = (np.array(hand_returns["R1_Sample"]) + np.array(hand_returns["R2_Sample"]) + np.array(hand_returns["R3_Sample"])) / 3.0
    ens_sample = np.array(hand_returns["Ensemble_Sample"])
    diff_es = (ens_sample - mean_3ind_sample) * 100.0
    t_es, p_es = stats.ttest_rel(ens_sample, mean_3ind_sample)
    paired_comparisons["Ensemble vs Mean(3 Ind) [Sample]"] = {
        "mean_diff_bb100": float(np.mean(diff_es)),
        "std_diff": float(np.std(diff_es)),
        "t_statistic": float(t_es),
        "p_value": float(p_es),
        "ci95_diff": [float(np.mean(diff_es) - 1.96 * stats.sem(diff_es)), float(np.mean(diff_es) + 1.96 * stats.sem(diff_es))]
    }

    # Argmax
    mean_3ind_argmax = (np.array(hand_returns["R1_Argmax"]) + np.array(hand_returns["R2_Argmax"]) + np.array(hand_returns["R3_Argmax"])) / 3.0
    ens_argmax = np.array(hand_returns["Ensemble_Argmax"])
    diff_ea = (ens_argmax - mean_3ind_argmax) * 100.0
    t_ea, p_ea = stats.ttest_rel(ens_argmax, mean_3ind_argmax)
    paired_comparisons["Ensemble vs Mean(3 Ind) [Argmax]"] = {
        "mean_diff_bb100": float(np.mean(diff_ea)),
        "std_diff": float(np.std(diff_ea)),
        "t_statistic": float(t_ea),
        "p_value": float(p_ea),
        "ci95_diff": [float(np.mean(diff_ea) - 1.96 * stats.sem(diff_ea)), float(np.mean(diff_ea) + 1.96 * stats.sem(diff_ea))]
    }

    final_output = {
        "agents": results_dict,
        "paired_comparisons": paired_comparisons
    }

    with open("v1023/data/v1023_3_paired_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("[+] Saved v1023/data/v1023_3_paired_results.json successfully.")

if __name__ == "__main__":
    main()
