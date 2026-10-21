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

class FlexibleHybridAgent:
    def __init__(self, preflop_agent, postflop_agent, preflop_type="ppo", postflop_type="model_a"):
        self.preflop_agent = preflop_agent
        self.postflop_agent = postflop_agent
        self.preflop_type = preflop_type
        self.postflop_type = postflop_type
        
    def act(self, st, seat, l6, f_set, seed_h, dec):
        rnd = get_round_from_state(st)
        if rnd == 0:
            if self.preflop_type == "model_a":
                return self.preflop_agent.act(st, seat, l6, f_set, seed_h, dec)
            else:
                inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                return self.preflop_agent.act(inf, l6)
        else:
            if self.postflop_type == "model_a":
                return self.postflop_agent.act(st, seat, l6, f_set, seed_h, dec)
            else:
                inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                return self.postflop_agent.act(inf, l6)

ACTION_NAMES = {0: "Fold", 1: "CheckCall", 2: "HalfPot", 3: "Pot", 4: "DoublePot", 5: "Allin"}

def run_v1023_9_experiment():
    print("=" * 70)
    print("POKER AI V10.23.9: HYBRID FORENSICS & 2x2 FACTORIAL AUDIT")
    print("=" * 70)
    
    # 1. Load PPO Models
    device = torch.device('cpu')
    anti_models = []
    ckpt_paths = [
        "v1022/models/StrictAntiLimper_R1_100K.pt",
        "v1022/models/StrictAntiLimper_R2_100K.pt",
        "v1022/models/StrictAntiLimper_R3_100K.pt",
    ]
    for p in ckpt_paths:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        anti_models.append(m)
        
    r2_model = anti_models[1]
    ppo_r2 = PPOAgentWrapper(r2_model)
    ppo_ens = EnsembleAgentWrapper(anti_models)
    
    # Baseline NFSP (Randomly initialized, documented deterministic baseline)
    bp_agent = DeterministicTrueNFSP_Agent()
    # Model A: Monte Carlo Equity Heuristic (sims=300)
    model_a = ModelA(bp_agent, sims=300)
    
    # Define Agents for 2x2 Factorial Study
    agents = {
        # Factorial 2x2 on R2
        "Policy_A_Pure_R2": FlexibleHybridAgent(ppo_r2, ppo_r2, "ppo", "ppo"),
        "Policy_B_Hybrid_R2_MA": FlexibleHybridAgent(ppo_r2, model_a, "ppo", "model_a"),
        "Policy_C_Inverted_MA_R2": FlexibleHybridAgent(model_a, ppo_r2, "model_a", "ppo"),
        "Policy_D_Pure_MA": FlexibleHybridAgent(model_a, model_a, "model_a", "model_a"),
        # Ensemble Factorial
        "Policy_E_Pure_Ensemble": FlexibleHybridAgent(ppo_ens, ppo_ens, "ppo", "ppo"),
        "Policy_F_Hybrid_Ens_MA": FlexibleHybridAgent(ppo_ens, model_a, "ppo", "model_a"),
        "Policy_G_Inverted_MA_Ens": FlexibleHybridAgent(model_a, ppo_ens, "model_a", "ppo"),
    }
    
    game = get_5p_fullgame()
    eval_seeds = list(range(160000, 160500)) # Same 500 hands as V10.23.8
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    print(f"\n[*] Running 7 Policies across {len(eval_seeds)} hands...")
    hand_returns = {name: [] for name in agents}
    hand_postflop_decs = {name: [] for name in agents}
    hand_action_traces = {name: [] for name in agents}
    
    t0 = time.time()
    for h_idx, seed_h in enumerate(eval_seeds):
        if (h_idx + 1) % 100 == 0:
            print(f"    Progress: {h_idx+1}/{len(eval_seeds)} ({time.time()-t0:.1f}s)")
            
        seat = h_idx % 5
        
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            pf_decs = 0
            actions_taken = []
            
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
                    act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    actions_taken.append({
                        "round": rnd,
                        "dec_idx": dec,
                        "act_id": act,
                        "act_name": ACTION_NAMES.get(act, str(act)),
                        "raw_amt": raw_act
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
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    
                if act == 0:
                    f_set.add(cp)
                st.apply_action(raw_act)
                
            ret_bb = st.returns()[seat] / 100.0
            hand_returns[name].append(ret_bb)
            hand_postflop_decs[name].append(pf_decs)
            hand_action_traces[name].append(actions_taken)
            
    print(f"[*] Benchmark finished in {time.time()-t0:.1f}s.")
    
    # -------------------------------------------------------------
    # PHASE 2: FORENSIC INVESTIGATION OF POSTFLOP HANDS
    # -------------------------------------------------------------
    print("\n[*] Phase 2: Forensic Reconciliation of Discrepant Hands...")
    ens_pure = np.array(hand_returns["Policy_E_Pure_Ensemble"])
    ens_hyb = np.array(hand_returns["Policy_F_Hybrid_Ens_MA"])
    ens_diff = ens_hyb - ens_pure
    
    discrepant_ens_indices = np.where(ens_diff != 0)[0]
    print(f"    Total hands where Hybrid Ensemble differs from Pure Ensemble: {len(discrepant_ens_indices)} hands!")
    
    discrepant_ens_details = []
    for idx in discrepant_ens_indices:
        seed = eval_seeds[idx]
        seat = idx % 5
        pure_ret = float(ens_pure[idx])
        hyb_ret = float(ens_hyb[idx])
        diff_bb = float(ens_diff[idx])
        pure_trace = hand_action_traces["Policy_E_Pure_Ensemble"][idx]
        hyb_trace = hand_action_traces["Policy_F_Hybrid_Ens_MA"][idx]
        discrepant_ens_details.append({
            "hand_index": int(idx),
            "seed": int(seed),
            "seat": int(seat),
            "pure_return_bb": pure_ret,
            "hybrid_return_bb": hyb_ret,
            "diff_bb": diff_bb,
            "pure_actions": pure_trace,
            "hybrid_actions": hyb_trace
        })
        print(f"      Hand {idx} (Seed {seed}): Pure={pure_ret:+.2f} BB, Hyb={hyb_ret:+.2f} BB, Diff={diff_bb:+.2f} BB")
        print(f"        Pure actions: {pure_trace}")
        print(f"        Hyb actions:  {hyb_trace}")

    # R2 Investigation
    r2_pure = np.array(hand_returns["Policy_A_Pure_R2"])
    r2_hyb = np.array(hand_returns["Policy_B_Hybrid_R2_MA"])
    r2_diff = r2_hyb - r2_pure
    discrepant_r2_indices = np.where(r2_diff != 0)[0]
    print(f"    Total hands where Hybrid R2 differs from Pure R2: {len(discrepant_r2_indices)} hands!")
    
    discrepant_r2_details = []
    for idx in discrepant_r2_indices:
        seed = eval_seeds[idx]
        seat = idx % 5
        discrepant_r2_details.append({
            "hand_index": int(idx),
            "seed": int(seed),
            "seat": int(seat),
            "pure_return_bb": float(r2_pure[idx]),
            "hybrid_return_bb": float(r2_hyb[idx]),
            "diff_bb": float(r2_diff[idx]),
            "pure_actions": hand_action_traces["Policy_A_Pure_R2"][idx],
            "hybrid_actions": hand_action_traces["Policy_B_Hybrid_R2_MA"][idx]
        })

    # -------------------------------------------------------------
    # PHASE 3: 2x2 FACTORIAL DECOMPOSITION
    # -------------------------------------------------------------
    summary = {}
    for name in agents:
        ret_arr = np.array(hand_returns[name])
        m_bb = safe_float(np.mean(ret_arr) * 100.0)
        s_bb = safe_float(stats.sem(ret_arr) * 100.0)
        pf_arr = np.array(hand_postflop_decs[name])
        summary[name] = {
            "mean_bb100": m_bb,
            "sem_bb100": s_bb,
            "ci95": [m_bb - 1.96 * s_bb, m_bb + 1.96 * s_bb],
            "total_hands": len(eval_seeds),
            "postflop_hands": int(np.sum(pf_arr > 0)),
            "total_postflop_decisions": int(np.sum(pf_arr))
        }

    mean_A = summary["Policy_A_Pure_R2"]["mean_bb100"]       # (R2, R2)
    mean_B = summary["Policy_B_Hybrid_R2_MA"]["mean_bb100"]   # (R2, MA)
    mean_C = summary["Policy_C_Inverted_MA_R2"]["mean_bb100"] # (MA, R2)
    mean_D = summary["Policy_D_Pure_MA"]["mean_bb100"]       # (MA, MA)
    
    preflop_main_effect = 0.5 * ((mean_A + mean_B) - (mean_C + mean_D))
    postflop_main_effect = 0.5 * ((mean_B + mean_D) - (mean_A + mean_C))
    interaction_effect = 0.5 * ((mean_B - mean_A) - (mean_D - mean_C))

    factorial_analysis = {
        "mean_A_Pure_R2": mean_A,
        "mean_B_Hybrid_R2_MA": mean_B,
        "mean_C_Inverted_MA_R2": mean_C,
        "mean_D_Pure_MA": mean_D,
        "preflop_main_effect_R2_over_MA": safe_float(preflop_main_effect),
        "postflop_main_effect_MA_over_R2": safe_float(postflop_main_effect),
        "interaction_effect": safe_float(interaction_effect)
    }

    # Paired Tests
    paired = {}
    def calc_paired(y1, y2, tag):
        diff = (np.array(y1) - np.array(y2)) * 100.0
        t_stat, p_val = stats.ttest_rel(y1, y2)
        sem = stats.sem(diff)
        m = np.mean(diff)
        paired[tag] = {
            "mean_diff_bb100": safe_float(m),
            "sem_diff": safe_float(sem),
            "t_statistic": safe_float(t_stat),
            "p_value": safe_float(p_val),
            "ci95_diff": [safe_float(m - 1.96 * sem), safe_float(m + 1.96 * sem)]
        }
        
    calc_paired(hand_returns["Policy_F_Hybrid_Ens_MA"], hand_returns["Policy_E_Pure_Ensemble"], "Hybrid_Ens_vs_Pure_Ens")
    calc_paired(hand_returns["Policy_B_Hybrid_R2_MA"], hand_returns["Policy_A_Pure_R2"], "Hybrid_R2_vs_Pure_R2")
    calc_paired(hand_returns["Policy_B_Hybrid_R2_MA"], hand_returns["Policy_C_Inverted_MA_R2"], "Hybrid_R2_vs_Inverted_R2")
    calc_paired(hand_returns["Policy_B_Hybrid_R2_MA"], hand_returns["Policy_D_Pure_MA"], "Hybrid_R2_vs_Pure_MA")
    calc_paired(hand_returns["Policy_F_Hybrid_Ens_MA"], hand_returns["Policy_D_Pure_MA"], "Hybrid_Ens_vs_Pure_MA")

    final_payload = {
        "model_hashes": {
            "StrictAntiLimper_R1_100K.pt": "1e29ff9262f2649a96074022a57489fe56ede464168aef92b2d3e221cc67c580",
            "StrictAntiLimper_R2_100K.pt": "42f0f6b9ff0fefaaab9e3214a0e97e22179b22560de844131aa8a3bba34b76f7",
            "StrictAntiLimper_R3_100K.pt": "961174c8f0778002a9ace26a86b944d93a5d303c3d97ffe32aa0d131f0c67a56"
        },
        "model_a_definition": "Monte Carlo Equity Heuristic (300 rollouts, deterministic NFSP fallback)",
        "eval_seeds_range": [160000, 160499],
        "summary": summary,
        "factorial_2x2": factorial_analysis,
        "paired_tests": paired,
        "phase2_forensics": {
            "ensemble_discrepant_count": len(discrepant_ens_indices),
            "ensemble_discrepant_hands": discrepant_ens_details,
            "r2_discrepant_count": len(discrepant_r2_indices),
            "r2_discrepant_hands": discrepant_r2_details[:10]
        }
    }

    out_path = "v1023/data/v1023_9_hybrid_forensics_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] Results successfully written to {out_path}")

if __name__ == "__main__":
    run_v1023_9_experiment()
