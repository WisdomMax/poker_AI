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
    def __init__(self, model):
        self.model = model
        
    def act(self, inf, legal_actions):
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
        
    def act(self, inf, legal_actions):
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
# FULL-GAME HYBRID AGENT (PPO Preflop + Model A Postflop)
# =====================================================================
class TrueFullgameHybridAgent:
    def __init__(self, preflop_agent, postflop_model_a):
        self.preflop_agent = preflop_agent
        self.postflop_model_a = postflop_model_a
        
    def act(self, st, seat, l6, f_set, seed_h, dec):
        rnd = get_round_from_state(st)
        if rnd == 0: # Preflop: PPO intuitive sizing and isolation
            inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
            return self.preflop_agent.act(inf, l6)
        else: # Postflop: Model A real-time equity search guard
            return self.postflop_model_a.act(st, seat, l6, f_set, seed_h, dec)

def run_experiment_b_fullgame(agents, eval_seeds, game):
    print(f"\n[*] EXPERIMENT B: True Full-Game Benchmark from Preflop ({len(eval_seeds)} hands)...")
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    hand_returns = {name: [] for name in agents}
    postflop_decisions = {name: [] for name in agents}
    allin_counts = {name: 0 for name in agents}
    
    t0 = time.time()
    for h_idx, seed_h in enumerate(eval_seeds):
        if (h_idx + 1) % 100 == 0:
            print(f"    Progress: {h_idx+1}/{len(eval_seeds)} ({time.time()-t0:.1f}s)")
            
        seat = h_idx % 5
        
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            agent_rng = np.random.RandomState(seed_h * 1000 + seat)
            
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            pf_decs = 0
            has_allin = False
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                rnd = get_round_from_state(st)
                
                if cp == seat:
                    if rnd > 0:
                        pf_decs += 1
                    if isinstance(agent, ModelA) or isinstance(agent, TrueFullgameHybridAgent):
                        act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    else:
                        inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                        act = agent.act(inf, l6)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    if act == 5 or raw_act == 10000:
                        has_allin = True
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
                        has_allin = True
                        
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            postflop_decisions[name].append(pf_decs)
            if has_allin:
                allin_counts[name] += 1
                
    elapsed = time.time() - t0
    print(f"    Completed Experiment B in {elapsed:.1f}s.")
    
    # Summary
    summary = {}
    for name in agents:
        returns = np.array(hand_returns[name])
        mean_bb = safe_float(np.mean(returns) * 100.0)
        sem_bb = safe_float(stats.sem(returns) * 100.0)
        pf_arr = np.array(postflop_decisions[name])
        
        summary[name] = {
            "overall_mean_bb100": mean_bb,
            "overall_sem_bb100": sem_bb,
            "overall_ci95": [mean_bb - 1.96 * sem_bb, mean_bb + 1.96 * sem_bb],
            "total_hands": len(eval_seeds),
            "allin_hands_count": allin_counts[name],
            "allin_pct": safe_float(allin_counts[name] / len(eval_seeds) * 100.0),
            "hands_with_postflop_decision": int(np.sum(pf_arr > 0)),
            "hands_with_postflop_decision_pct": safe_float(np.sum(pf_arr > 0) / len(eval_seeds) * 100.0),
            "total_postflop_decisions": int(np.sum(pf_arr)),
            "avg_postflop_decisions_per_hand": safe_float(np.mean(pf_arr))
        }
        
    # Paired comparisons
    paired = {}
    r2_ret = np.array(hand_returns["R2_Argmax"]) * 100.0
    ma_ret = np.array(hand_returns["Model_A_MC300"]) * 100.0
    hyb_ret = np.array(hand_returns["Hybrid_R2_ModelA"]) * 100.0
    hyb_ens_ret = np.array(hand_returns["Hybrid_Ensemble_ModelA"]) * 100.0
    
    # 1. Hybrid vs R2 Alone
    diff_h_r2 = hyb_ret - r2_ret
    t_hr2, p_hr2 = stats.ttest_rel(hyb_ret, r2_ret)
    paired["Hybrid_R2_ModelA vs R2_Argmax"] = {
        "mean_diff_bb100": safe_float(np.mean(diff_h_r2)),
        "sem_diff": safe_float(stats.sem(diff_h_r2)),
        "t_statistic": safe_float(t_hr2),
        "p_value": safe_float(p_hr2),
        "ci95_diff": [safe_float(np.mean(diff_h_r2) - 1.96 * stats.sem(diff_h_r2)), safe_float(np.mean(diff_h_r2) + 1.96 * stats.sem(diff_h_r2))]
    }
    
    # 2. Hybrid vs Model A Alone
    diff_h_ma = hyb_ret - ma_ret
    t_hma, p_hma = stats.ttest_rel(hyb_ret, ma_ret)
    paired["Hybrid_R2_ModelA vs Model_A_MC300"] = {
        "mean_diff_bb100": safe_float(np.mean(diff_h_ma)),
        "sem_diff": safe_float(stats.sem(diff_h_ma)),
        "t_statistic": safe_float(t_hma),
        "p_value": safe_float(p_hma),
        "ci95_diff": [safe_float(np.mean(diff_h_ma) - 1.96 * stats.sem(diff_h_ma)), safe_float(np.mean(diff_h_ma) + 1.96 * stats.sem(diff_h_ma))]
    }

    # 3. Hybrid Ensemble vs Ensemble Alone
    ens_ret = np.array(hand_returns["Ensemble_Argmax"]) * 100.0
    diff_he_ens = hyb_ens_ret - ens_ret
    t_he, p_he = stats.ttest_rel(hyb_ens_ret, ens_ret)
    paired["Hybrid_Ensemble_ModelA vs Ensemble_Argmax"] = {
        "mean_diff_bb100": safe_float(np.mean(diff_he_ens)),
        "sem_diff": safe_float(stats.sem(diff_he_ens)),
        "t_statistic": safe_float(t_he),
        "p_value": safe_float(p_he),
        "ci95_diff": [safe_float(np.mean(diff_he_ens) - 1.96 * stats.sem(diff_he_ens)), safe_float(np.mean(diff_he_ens) + 1.96 * stats.sem(diff_he_ens))]
    }

    return {"summary": summary, "paired_tests": paired}

def main():
    print("=================================================================")
    print("POKER AI V10.23.8: TRUE HYBRID BENCHMARK & MULTI-PHASE AUDIT")
    print("=================================================================")

    # 1. Load models
    anti_models = []
    for r in range(1, 4):
        m_a = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_a.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt"))
        m_a.eval()
        anti_models.append(m_a)
        
    bp_agent = DeterministicTrueNFSP_Agent() # RandomlyInitializedNFSP (documented baseline)
    model_a = ModelA(bp_agent, sims=300)
    
    r2 = PPOAgentWrapper(anti_models[1])
    ens = EnsembleAgentWrapper(anti_models)
    
    hybrid_r2 = TrueFullgameHybridAgent(preflop_agent=r2, postflop_model_a=model_a)
    hybrid_ens = TrueFullgameHybridAgent(preflop_agent=ens, postflop_model_a=model_a)
    
    agents = {
        "R2_Argmax": r2,
        "Ensemble_Argmax": ens,
        "Model_A_MC300": model_a,
        "Hybrid_R2_ModelA": hybrid_r2,
        "Hybrid_Ensemble_ModelA": hybrid_ens
    }

    game = get_5p_fullgame()
    # Brand new unseen seeds: 160000 ~ 160499 (500 hands)
    NUM_HANDS = 500
    eval_seeds = [160000 + i for i in range(NUM_HANDS)]

    exp_b_results = run_experiment_b_fullgame(agents, eval_seeds, game)

    final_payload = {
        "eval_seeds_range": [eval_seeds[0], eval_seeds[-1]],
        "num_hands": NUM_HANDS,
        "experiment_b_fullgame": exp_b_results
    }

    out_path = "v1023/data/v1023_8_hybrid_results.json"
    with open(out_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] V10.23.8 Final Results saved to {out_path}")

if __name__ == "__main__":
    main()
