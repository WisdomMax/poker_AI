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
from v1020.run_v1020_experiment_integrity_audit import (
    RigorousCheckRaiseTrapBot, RigorousPolarizedBluffValueBot,
)
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

def evaluate_controlled_postflop(league_name, opp_factory, agents, eval_seeds, game):
    print(f"\n[*] Evaluating League: {league_name} ({len(eval_seeds)} hands)...")
    results = {}
    
    hand_returns = {name: [] for name in agents}
    hand_categories = {name: [] for name in agents}
    
    postflop_actions_taken = {name: defaultdict(int) for name in agents}
    postflop_hands_returns = {name: [] for name in agents}
    postflop_hands_count = {name: 0 for name in agents}
    
    t0 = time.time()
    for h_idx, seed_h in enumerate(eval_seeds):
        seat = h_idx % 5
        
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            
            # Precise all-in tracking
            someone_allin = False
            first_allin_street = None
            hero_allin_type = None # "Pushed", "Called", "Folded_To_Allin", "Already_Folded", "No_Allin"
            hero_folded = False
            hero_participated_postflop = False
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                rnd = get_round_from_state(st)
                spent_before = extract_player_spent(st)
                to_call = max(spent_before) - spent_before[cp]
                
                # Check if this player is facing an all-in
                is_facing_allin = (max(spent_before) >= 9900) and (to_call > 0)
                
                if cp == seat:
                    if rnd > 0:
                        hero_participated_postflop = True
                        
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    if isinstance(agent, ModelA):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        act = agent.act(inf, l6, agent_rng)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                    if rnd > 0:
                        postflop_actions_taken[name][act] += 1
                        
                    # Allin classification for Hero
                    if act == 5 or raw_act == 10000:
                        someone_allin = True
                        if first_allin_street is None:
                            first_allin_street = "Preflop" if rnd == 0 else "Postflop"
                        if is_facing_allin:
                            hero_allin_type = "Hero_Called_Allin"
                        else:
                            hero_allin_type = "Hero_Pushed_Allin"
                    elif act == 1 and is_facing_allin: # Called all-in via Call action
                        someone_allin = True
                        hero_allin_type = "Hero_Called_Allin"
                        if first_allin_street is None:
                            first_allin_street = "Preflop" if rnd == 0 else "Postflop"
                    elif act == 0:
                        hero_folded = True
                        if is_facing_allin and hero_allin_type is None:
                            hero_allin_type = "Hero_Folded_Facing_Allin"
                            
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
                        someone_allin = True
                        if first_allin_street is None:
                            first_allin_street = "Preflop" if rnd == 0 else "Postflop"
                            
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            
            # Finalize categorization
            if not someone_allin:
                final_cat = "No_Allin"
            elif hero_allin_type == "Hero_Pushed_Allin":
                final_cat = f"{first_allin_street}_Hero_Pushed"
            elif hero_allin_type == "Hero_Called_Allin":
                final_cat = f"{first_allin_street}_Hero_Called"
            elif hero_allin_type == "Hero_Folded_Facing_Allin":
                final_cat = f"{first_allin_street}_Hero_Folded_To_Allin"
            else: # Hero was already folded when opponents pushed allin
                final_cat = f"{first_allin_street}_Opponents_Allin_Hero_Folded_Prior"
                
            hand_categories[name].append(final_cat)
            
            if hero_participated_postflop:
                postflop_hands_returns[name].append(ret_bb)
                postflop_hands_count[name] += 1

    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f}s.")
    
    # Compile statistics
    action_names = {0: "Fold", 1: "CheckCall", 2: "HalfPot_0.5x", 3: "Pot_1.0x", 4: "DoublePot_2.0x", 5: "Allin"}
    
    for name in agents:
        returns = np.array(hand_returns[name])
        cats = np.array(hand_categories[name])
        
        all_bb100 = safe_float(np.mean(returns) * 100.0)
        sem_all = safe_float(stats.sem(returns) * 100.0)
        
        # Categorical statistics
        unique_cats = sorted(list(set(cats)))
        cat_stats = {}
        for c in unique_cats:
            mask = (cats == c)
            cnt = int(np.sum(mask))
            m_ret = safe_float(np.mean(returns[mask]) * 100.0) if cnt > 0 else 0.0
            s_ret = safe_float(stats.sem(returns[mask]) * 100.0) if cnt > 1 else 0.0
            cat_stats[c] = {
                "count": cnt,
                "pct": safe_float(cnt / len(returns) * 100.0),
                "mean_bb100": m_ret,
                "sem_bb100": s_ret
            }
            
        # Postflop performance
        pf_cnt = postflop_hands_count[name]
        pf_rets = np.array(postflop_hands_returns[name])
        pf_mean = safe_float(np.mean(pf_rets) * 100.0) if pf_cnt > 0 else 0.0
        pf_sem = safe_float(stats.sem(pf_rets) * 100.0) if pf_cnt > 1 else 0.0
        
        tot_pf_acts = sum(postflop_actions_taken[name].values())
        pf_act_pcts = {
            action_names[a]: safe_float(postflop_actions_taken[name][a] / tot_pf_acts * 100.0) if tot_pf_acts else 0.0
            for a in range(6)
        }
        
        results[name] = {
            "overall_mean_bb100": all_bb100,
            "overall_sem_bb100": sem_all,
            "overall_ci95": [all_bb100 - 1.96 * sem_all, all_bb100 + 1.96 * sem_all],
            "allin_breakdown": cat_stats,
            "controlled_postflop_benchmark": {
                "hands_participated_count": pf_cnt,
                "hands_participated_pct": safe_float(pf_cnt / len(returns) * 100.0),
                "mean_bb100_in_postflop_hands": pf_mean,
                "sem_bb100_in_postflop_hands": pf_sem,
                "ci95_in_postflop_hands": [pf_mean - 1.96 * pf_sem, pf_mean + 1.96 * pf_sem],
                "total_postflop_decisions": tot_pf_acts,
                "postflop_action_percentages": pf_act_pcts
            }
        }
        
    return results

def main():
    print("=================================================================")
    print("POKER AI V10.23.6: ALL-IN REPAIR & CONTROLLED POSTFLOP BENCHMARK")
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
    # Brand new unseen seeds: 140000 ~ 140499 (500 hands each)
    NUM_HANDS = 500
    eval_seeds = [140000 + i for i in range(NUM_HANDS)]

    # League B: Realistic Sizing League (Primary Focus)
    factory_b = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]

    # League C: Unseen Diverse Strategies (Robustness Check)
    factory_c = lambda s: [
        RigorousCheckRaiseTrapBot(seed=s * 13),
        RigorousPolarizedBluffValueBot(seed=s * 17),
        bp_agent,
        LooseLimperBot(seed=s * 23)
    ]

    leagues_data = {}
    leagues_data["League_B_Realistic"] = evaluate_controlled_postflop("League B (Realistic Sizing)", factory_b, agents, eval_seeds, game)
    leagues_data["League_C_Diverse"] = evaluate_controlled_postflop("League C (Unseen Diverse)", factory_c, agents, eval_seeds, game)

    out_path = "v1023/data/v1023_6_results.json"
    with open(out_path, "w") as f:
        json.dump(leagues_data, f, indent=2)
    print(f"\n[+] V10.23.6 Comprehensive Results saved to {out_path}")

if __name__ == "__main__":
    main()
