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
    get_5p_fullgame, advance_chance_nodes, extract_player_spent, True6ActionAdapter,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v79.run_v79_controlled_causal_study import normalize_inf_state_v79
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
# REALISTIC REASONABLE-SIZING AGGRESSIVE BOT (No blind all-in spam)
# =====================================================================
class RealisticAggressiveBot:
    """Aggressive player that uses realistic pot-sized bets instead of 80% blind all-ins"""
    def __init__(self, seed):
        self.rng = random.Random(seed)
        
    def select_action(self, st, cp, l6):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        if tc > 0:
            # 3-bet / raise with 0.5x or 1.0x Pot (Actions 2, 3), reserve All-in (5) for rare situations (5%)
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

# =====================================================================
# AGENT DEFINITIONS
# =====================================================================
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
    def __init__(self, models, mode="argmax"):
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

def evaluate_league(league_name, opp_factory, agents, eval_seeds, game):
    print(f"\n[*] Evaluating League: {league_name} ({len(eval_seeds)} hands)...")
    results = {}
    hand_records = {name: [] for name in agents}
    street_reach_counts = {name: defaultdict(int) for name in agents}
    allin_hand_flags = {name: [] for name in agents}
    
    t0 = time.time()
    for h_idx, seed_h in enumerate(eval_seeds):
        seat = h_idx % 5
        
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            f_set = set()
            dec = 0
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            is_allin_hand = False
            max_round_reached = 0
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                rnd = get_round_from_state(st)
                if rnd > max_round_reached:
                    max_round_reached = rnd
                    
                if cp == seat:
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    if isinstance(agent, ModelA):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    if act == 5 or raw_act == 10000:
                        is_allin_hand = True
                    dec += 1
                else:
                    opp_idx = cp if cp < seat else cp - 1
                    bot = opps[opp_idx]
                    if isinstance(bot, DeterministicTrueNFSP_Agent):
                        inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
                    else:
                        act = bot.select_action(st, cp, l6)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    if act == 5 or raw_act == 10000:
                        is_allin_hand = True
                        
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_records[name].append(ret_bb)
            allin_hand_flags[name].append(is_allin_hand)
            
            # Record max round reached (0: Preflop, 1: Flop, 2: Turn, 3: River)
            for r in range(max_round_reached + 1):
                street_reach_counts[name][r] += 1
                
    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f}s.")
    
    # Calculate detailed metrics
    for name in agents:
        returns = np.array(hand_records[name])
        allin_mask = np.array(allin_hand_flags[name], dtype=bool)
        
        all_bb100 = float(np.mean(returns) * 100.0)
        sem_all = float(stats.sem(returns) * 100.0)
        
        non_allin_returns = returns[~allin_mask]
        non_allin_bb100 = float(np.mean(non_allin_returns) * 100.0) if len(non_allin_returns) > 0 else 0.0
        sem_non = float(stats.sem(non_allin_returns) * 100.0) if len(non_allin_returns) > 0 else 0.0
        
        allin_returns = returns[allin_mask]
        allin_bb100 = float(np.mean(allin_returns) * 100.0) if len(allin_returns) > 0 else 0.0
        
        allin_hand_pct = float(np.mean(allin_mask) * 100.0)
        
        total_h = len(eval_seeds)
        results[name] = {
            "total_hands": total_h,
            "overall_mean_bb100": all_bb100,
            "overall_sem_bb100": sem_all,
            "overall_ci95": [all_bb100 - 1.96 * sem_all, all_bb100 + 1.96 * sem_all],
            "allin_hand_pct": allin_hand_pct,
            "non_allin_hands_count": int(len(non_allin_returns)),
            "non_allin_mean_bb100": non_allin_bb100,
            "non_allin_sem_bb100": sem_non,
            "non_allin_ci95": [non_allin_bb100 - 1.96 * sem_non, non_allin_bb100 + 1.96 * sem_non],
            "allin_hands_count": int(len(allin_returns)),
            "allin_mean_bb100": allin_bb100,
            "street_reach_pct": {
                "Preflop": 100.0,
                "Flop": float(street_reach_counts[name][1] / total_h * 100.0),
                "Turn": float(street_reach_counts[name][2] / total_h * 100.0),
                "River": float(street_reach_counts[name][3] / total_h * 100.0)
            }
        }
    return results

def main():
    print("=================================================================")
    print("POKER AI V10.23.4: REALISTIC POKER ENVIRONMENT & ROBUSTNESS AUDIT")
    print("=================================================================")

    # 1. Load models
    anti_models = []
    for r in range(1, 4):
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)
        
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = ModelA(bp_agent, sims=300)
    
    agents = {
        "R1_Argmax": PPOAgentWrapper(anti_models[0], mode="argmax"),
        "R2_Argmax": PPOAgentWrapper(anti_models[1], mode="argmax"),
        "R3_Argmax": PPOAgentWrapper(anti_models[2], mode="argmax"),
        "Ensemble_Argmax": EnsembleAgentWrapper(anti_models, mode="argmax"),
        "Model_A_MC300": model_a,
    }

    game = get_5p_fullgame()
    # Brand new unseen seeds: 120000 ~ 120399 (400 hands per league for high precision & speed)
    NUM_HANDS = 400
    eval_seeds = [120000 + i for i in range(NUM_HANDS)]

    # 2. Define 3 League Environments
    # League A: Classic Strict Limper (with 80% blind all-in UltraAggressiveBot)
    factory_a = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        DeterministicUltraAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]

    # League B: Realistic Reasonable-Sizing League (RealisticAggressiveBot with normal 0.5x/1.0x bets)
    factory_b = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]

    # League C: Unseen Diverse Strategy League (CheckRaiseTrap, Polarized, NFSP, Limper)
    factory_c = lambda s: [
        RigorousCheckRaiseTrapBot(seed=s * 13),
        RigorousPolarizedBluffValueBot(seed=s * 17),
        bp_agent,
        LooseLimperBot(seed=s * 23)
    ]

    leagues_data = {}
    leagues_data["League_A_Classic_StrictLimper"] = evaluate_league("League A (Classic Strict Limper)", factory_a, agents, eval_seeds, game)
    leagues_data["League_B_Realistic_Sizing"] = evaluate_league("League B (Realistic Reasonable Sizing)", factory_b, agents, eval_seeds, game)
    leagues_data["League_C_Unseen_Diverse"] = evaluate_league("League C (Unseen Diverse Strategies)", factory_c, agents, eval_seeds, game)

    out_path = "v1023/data/v1023_4_realistic_results.json"
    with open(out_path, "w") as f:
        json.dump(leagues_data, f, indent=2)
    print(f"\n[+] V10.23.4 Comprehensive Results saved to {out_path}")

if __name__ == "__main__":
    main()
