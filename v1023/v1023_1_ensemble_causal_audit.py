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
from v1021.run_v1021_anti_limper_league import LooseLimperBot

# =====================================================================
# AGENT WRAPPERS FOR RIGOROUS COMPARISON
# =====================================================================
class PPONeuralAgentWrapper:
    def __init__(self, model, mode="sample"):
        self.model = model
        self.mode = mode # "sample" or "argmax"
        
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

class EnsembleNeuralAgentWrapper:
    def __init__(self, models, mode="sample"):
        self.models = models
        self.mode = mode # "sample" or "argmax"
        
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

def get_to_call_amount(state, player):
    """Calculates the chips required to call"""
    spent = extract_player_spent(state)
    max_spent = max(spent)
    return max(0, max_spent - spent[player])

def main():
    print("=================================================================")
    print("POKER AI V10.23.1: ENSEMBLE CAUSAL AUDIT & POLICY DIAGNOSTICS")
    print("=================================================================")
    
    # 1. Load trained AntiLimper models
    anti_models = []
    for r in range(1, 4):
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)
        
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = ModelA(bp_agent, sims=300) # Full MC-300
    
    # 2. Setup 9 evaluation candidate agents
    agents = {
        "R1_Sample": PPONeuralAgentWrapper(anti_models[0], mode="sample"),
        "R1_Argmax": PPONeuralAgentWrapper(anti_models[0], mode="argmax"),
        "R2_Sample": PPONeuralAgentWrapper(anti_models[1], mode="sample"),
        "R2_Argmax": PPONeuralAgentWrapper(anti_models[1], mode="argmax"),
        "R3_Sample": PPONeuralAgentWrapper(anti_models[2], mode="sample"),
        "R3_Argmax": PPONeuralAgentWrapper(anti_models[2], mode="argmax"),
        "Ensemble_Sample": EnsembleNeuralAgentWrapper(anti_models, mode="sample"),
        "Ensemble_Argmax": EnsembleNeuralAgentWrapper(anti_models, mode="argmax"),
        "Model_A_MC300": model_a,
    }
    
    # 3. Non-overlapping 1,000 Unique Seeds (Evaluation set: 105,000 ~ 105,999)
    NUM_HANDS = 1000
    eval_seeds = [105000 + i for i in range(NUM_HANDS)]
    
    game = get_5p_fullgame()
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s*13),
        DeterministicTightPassiveBot(seed=s*17),
        DeterministicUltraAggressiveBot(seed=s*19),
        LooseLimperBot(seed=s*23)
    ]
    
    # Data structures for metrics
    hand_returns = {name: [] for name in agents}
    action_counts = {name: defaultdict(int) for name in agents}
    action_chips_spent = {name: defaultdict(float) for name in agents}
    
    # Detailed attribution: profit on hands where agent took action X
    # e.g., profit when agent PaidCall vs FreeCheck vs Raise
    hand_action_set = {name: [] for name in agents} # list of dict(actions_taken) per hand
    showdown_stats = {name: {"reached": 0, "won": 0, "net_chips": 0.0} for name in agents}
    
    print(f"[*] Starting rigorous evaluation over {NUM_HANDS} identical paired hands...")
    t0 = time.time()
    
    for h_idx, seed_h in enumerate(eval_seeds):
        if (h_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {h_idx + 1}/{NUM_HANDS} hands evaluated ({elapsed:.1f}s elapsed)")
            
        seat = h_idx % 5 # Rotate hero seat fairly
        opps = opp_factory(seed_h)
        
        for name, agent in agents.items():
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            f_set = set()
            dec = 0
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            actions_in_hand = defaultdict(int)
            spent_before = 0
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                
                if cp == seat:
                    to_call = get_to_call_amount(st, seat)
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    
                    if isinstance(agent, ModelA):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                        
                    # Classify action into detailed forensic categories
                    if act == 0:
                        detailed_act = "Fold"
                    elif act == 1:
                        if to_call == 0:
                            detailed_act = "FreeCheck"
                        else:
                            detailed_act = "PaidCall"
                            action_chips_spent[name]["PaidCall"] += to_call
                    elif act == 2:
                        detailed_act = "MinRaise"
                    elif act == 3:
                        detailed_act = "HalfPot"
                    elif act == 4:
                        detailed_act = "Pot"
                    elif act == 5:
                        detailed_act = "Allin"
                    else:
                        detailed_act = f"Action_{act}"
                        
                    action_counts[name][detailed_act] += 1
                    actions_in_hand[detailed_act] += 1
                    dec += 1
                else:
                    opp_idx = cp if cp < seat else cp - 1
                    bot = opps[opp_idx]
                    if isinstance(bot, DeterministicTrueNFSP_Agent):
                        inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h*101 + cp*17))
                    else:
                        act = bot.select_action(st, cp, l6)
                        
                if act == 0:
                    f_set.add(cp)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            hand_action_set[name].append(actions_in_hand)
            
            # Check showdown
            active_at_end = [p for p in range(5) if p not in f_set]
            if len(active_at_end) > 1 and seat in active_at_end:
                showdown_stats[name]["reached"] += 1
                if ret_bb > 0:
                    showdown_stats[name]["won"] += 1
                showdown_stats[name]["net_chips"] += ret_bb * 100.0

    total_time = time.time() - t0
    print(f"\n[+] Evaluation finished in {total_time:.1f}s.")
    
    # 4. Statistical Analysis & Attribution
    summary_report = {}
    print("\n" + "="*85)
    print(f"{'Agent':<18} | {'Mean bb/100':<12} | {'95% CI':<18} | {'Fold%':<6} | {'Check%':<6} | {'Call%':<6} | {'Raise%':<6}")
    print("="*85)
    
    for name in agents:
        returns = np.array(hand_returns[name])
        mean_bb100 = np.mean(returns) * 100.0
        sem = stats.sem(returns) * 100.0
        ci95 = 1.96 * sem
        
        tot_acts = sum(action_counts[name].values())
        fold_pct = (action_counts[name]["Fold"] / tot_acts) * 100.0 if tot_acts else 0
        check_pct = (action_counts[name]["FreeCheck"] / tot_acts) * 100.0 if tot_acts else 0
        call_pct = (action_counts[name]["PaidCall"] / tot_acts) * 100.0 if tot_acts else 0
        raise_acts = action_counts[name]["MinRaise"] + action_counts[name]["HalfPot"] + action_counts[name]["Pot"] + action_counts[name]["Allin"]
        raise_pct = (raise_acts / tot_acts) * 100.0 if tot_acts else 0
        
        # Hand attribution
        # Average return when PaidCall was executed at least once
        call_hands = [returns[i] * 100.0 for i in range(NUM_HANDS) if hand_action_set[name][i].get("PaidCall", 0) > 0]
        raise_hands = [returns[i] * 100.0 for i in range(NUM_HANDS) if any(hand_action_set[name][i].get(r, 0) > 0 for r in ["MinRaise", "HalfPot", "Pot", "Allin"])]
        
        mean_call_ret = np.mean(call_hands) if call_hands else 0.0
        mean_raise_ret = np.mean(raise_hands) if raise_hands else 0.0
        
        sd_reached = showdown_stats[name]["reached"]
        sd_win_rate = (showdown_stats[name]["won"] / sd_reached * 100.0) if sd_reached else 0.0
        sd_net_bb100 = (showdown_stats[name]["net_chips"] / 100.0) / (NUM_HANDS / 100.0)
        
        summary_report[name] = {
            "mean_bb100": float(mean_bb100),
            "sem_bb100": float(sem),
            "ci95_low": float(mean_bb100 - ci95),
            "ci95_high": float(mean_bb100 + ci95),
            "action_counts": dict(action_counts[name]),
            "fold_pct": float(fold_pct),
            "free_check_pct": float(check_pct),
            "paid_call_pct": float(call_pct),
            "raise_pct": float(raise_pct),
            "paid_call_hands_count": len(call_hands),
            "paid_call_hands_mean_bb100": float(mean_call_ret),
            "raise_hands_count": len(raise_hands),
            "raise_hands_mean_bb100": float(mean_raise_ret),
            "showdown_reached_count": sd_reached,
            "showdown_win_rate_pct": float(sd_win_rate),
            "showdown_net_bb100": float(sd_net_bb100)
        }
        
        print(f"{name:<18} | {mean_bb100:+8.2f}     | [{mean_bb100-ci95:+6.1f}, {mean_bb100+ci95:+6.1f}] | {fold_pct:5.1f}% | {check_pct:5.1f}% | {call_pct:5.1f}% | {raise_pct:5.1f}%")
        
    print("="*85)
    
    # Save output JSON
    os.makedirs("v1023/data", exist_ok=True)
    out_path = "v1023/data/v1023_1_causal_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[+] Rigorous Causal Audit Results saved to {out_path}")

if __name__ == "__main__":
    main()
