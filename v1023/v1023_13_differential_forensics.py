import os
import sys
import json
import time
import hashlib
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
    DeterministicTrueNFSP_Agent, ModelA, estimate_equity_det,
    DeterministicBalancedBot, DeterministicTightPassiveBot,
    RobustBayesianRangeModel
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1021.run_v1021_anti_limper_league import LooseLimperBot

ACTION_NAMES = {0: "Fold", 1: "CheckCall", 2: "HalfPot", 3: "Pot", 4: "DoublePot", 5: "Allin"}

def safe_float(val):
    if val is None or np.isnan(val) or np.isinf(val):
        return None
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

    def get_action_probs(self, inf, legal_actions):
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
            probs = torch.softmax(masked_logits[0], dim=-1).cpu().numpy()
            return probs.tolist()

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

class DedicatedInstrumentedModelA:
    def __init__(self, bp, tag, sims=300):
        self.bp = bp
        self.tag = tag
        self.sims = sims
        self.temp = 2.0
        self.stats = {
            "Flop": {"equity_heuristic": 0, "fallback_free_check": 0, "fallback_facing_bet_nofold": 0},
            "Turn": {"equity_heuristic": 0, "fallback_free_check": 0, "fallback_facing_bet_nofold": 0},
            "River": {"equity_heuristic": 0, "fallback_free_check": 0, "fallback_facing_bet_nofold": 0}
        }
        self.last_meta = {}
        
    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        pot = sum(sp)
        rnd = get_round_from_state(st)
        street_name = ["Preflop", "Flop", "Turn", "River"][min(rnd, 3)]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        
        if tc > 0 and 0 in l6:
            if street_name in self.stats:
                self.stats[street_name]["equity_heuristic"] += 1
            hc = list(np.where(inf[5:57] == 1.0)[0])
            bc = list(np.where(inf[57:109] == 1.0)[0])
            ao = sum(1 for p in range(5) if p != cp and (folded is None or p not in folded))
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hc, bc, mode='temperature', temperature=self.temp,
                opp_aggression_level=1.5, is_raised=(tc >= 100))
            po = tc / float(pot + tc)
            eq_rng = np.random.RandomState((hand_seed * 10007 + cp * 101 + dec_idx * 17) & 0xFFFFFFFF)
            eq = estimate_equity_det(hc, bc, combos, probs, ao, self.sims, eq_rng)
            
            self.last_meta = {
                "branch": "Equity_Heuristic", "street": street_name,
                "tc": tc, "pot": pot, "pot_odds": po, "mc_equity": eq, "active_opps": ao
            }
            if eq > po + 0.05:
                r = [a for a in l6 if a in [2, 3]]
                act = r[0] if (r and eq > 0.65) else (1 if 1 in l6 else l6[0])
            elif eq < po - 0.05:
                act = 0 # Fold
            else:
                act = 1 if 1 in l6 else l6[0]
            self.last_meta["act"] = act
            return act
        else:
            branch_sub = "fallback_free_check" if tc == 0 else "fallback_facing_bet_nofold"
            if street_name in self.stats:
                self.stats[street_name][branch_sub] += 1
            bp_rng = np.random.RandomState((hand_seed * 20011 + cp * 307 + dec_idx * 31) & 0xFFFFFFFF)
            act = self.bp.select_action_deterministic(inf, l6, bp_rng)
            self.last_meta = {
                "branch": branch_sub, "street": street_name,
                "tc": tc, "pot": pot, "act": act
            }
            return act

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

def trace_single_hand_deep(seed_h, ppo_r2, model_a, game):
    seat = seed_h % 5
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    # 1. Pure R2
    opps_pure = opp_factory(seed_h)
    chance_rng_pure = np.random.RandomState(seed_h)
    st_pure = game.new_initial_state()
    advance_chance_nodes_persistent(st_pure, chance_rng_pure)
    f_set_pure = set()
    dec = 0
    pure_log = []
    
    while not st_pure.is_terminal():
        if st_pure.is_chance_node():
            advance_chance_nodes_persistent(st_pure, chance_rng_pure)
            continue
        cp = st_pure.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_pure, cp)
        rnd = get_round_from_state(st_pure)
        sp = extract_player_spent(st_pure)
        pot = sum(sp)
        tc = max(sp) - sp[cp]
        
        if cp == seat:
            inf = np.array(st_pure.information_state_tensor(seat), dtype=np.float32)
            act = ppo_r2.act(inf, l6)
            probs = ppo_r2.get_action_probs(inf, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_pure, cp, act)
            pure_log.append({
                "round": rnd, "street": ["Preflop", "Flop", "Turn", "River"][min(rnd, 3)],
                "dec": dec, "pot": pot, "to_call": tc, "act": ACTION_NAMES[act], "raw_amt": raw_act,
                "ppo_probs": [round(x, 4) for x in probs]
            })
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opps_pure[opp_idx]
            if isinstance(bot, DeterministicTrueNFSP_Agent):
                inf_b = np.array(st_pure.information_state_tensor(cp), dtype=np.float32)
                act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
            else:
                act = bot.select_action(st_pure, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_pure, cp, act)
        if act == 0:
            f_set_pure.add(cp)
        st_pure.apply_action(raw_act)
    pure_ret = st_pure.returns()[seat] / 100.0
    
    # 2. Hybrid R2
    opps_hyb = opp_factory(seed_h)
    chance_rng_hyb = np.random.RandomState(seed_h)
    st_hyb = game.new_initial_state()
    advance_chance_nodes_persistent(st_hyb, chance_rng_hyb)
    f_set_hyb = set()
    dec = 0
    hyb_log = []
    
    agent_hyb = FlexibleHybridAgent(ppo_r2, model_a, "ppo", "model_a")
    while not st_hyb.is_terminal():
        if st_hyb.is_chance_node():
            advance_chance_nodes_persistent(st_hyb, chance_rng_hyb)
            continue
        cp = st_hyb.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_hyb, cp)
        rnd = get_round_from_state(st_hyb)
        sp = extract_player_spent(st_hyb)
        pot = sum(sp)
        tc = max(sp) - sp[cp]
        
        if cp == seat:
            act = agent_hyb.act(st_hyb, seat, l6, f_set_hyb, seed_h, dec)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_hyb, cp, act)
            meta = model_a.last_meta if rnd > 0 else {"branch": "PPO_Preflop"}
            hyb_log.append({
                "round": rnd, "street": ["Preflop", "Flop", "Turn", "River"][min(rnd, 3)],
                "dec": dec, "pot": pot, "to_call": tc, "act": ACTION_NAMES[act], "raw_amt": raw_act,
                "meta": meta
            })
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opps_hyb[opp_idx]
            if isinstance(bot, DeterministicTrueNFSP_Agent):
                inf_b = np.array(st_hyb.information_state_tensor(cp), dtype=np.float32)
                act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
            else:
                act = bot.select_action(st_hyb, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_hyb, cp, act)
        if act == 0:
            f_set_hyb.add(cp)
        st_hyb.apply_action(raw_act)
    hyb_ret = st_hyb.returns()[seat] / 100.0
    
    return {
        "seed": seed_h, "seat": seat,
        "pure_return_bb": pure_ret, "hybrid_return_bb": hyb_ret, "net_diff_bb": hyb_ret - pure_ret,
        "pure_log": pure_log, "hybrid_log": hyb_log
    }

def main():
    print("=" * 70)
    print("POKER AI V10.23.13: DIFFERENTIAL HAND FORENSICS & ACTIVATION AUDIT")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cpu')
    anti_models = []
    for r in range(1, 4):
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt", map_location=device))
        m.eval()
        anti_models.append(m)
        
    ppo_ens = EnsembleAgentWrapper(anti_models)
    ppo_r2 = PPOAgentWrapper(anti_models[1])
    
    # 2 Dedicated Model A instances to completely isolate stats
    bp_agent1 = DeterministicTrueNFSP_Agent()
    bp_agent2 = DeterministicTrueNFSP_Agent()
    model_a_ens = DedicatedInstrumentedModelA(bp_agent1, tag="Ensemble", sims=300)
    model_a_r2 = DedicatedInstrumentedModelA(bp_agent2, tag="R2", sims=300)
    
    game = get_5p_fullgame()
    eval_seeds = [180000 + i for i in range(500)]
    
    # 1. Forensic Deep Trace of Seed 180408 (Top Gainer +249 BB) and Seed 180158 (Worst Gainer -117 BB)
    print("\n[*] Phase 1: Deep Forensic Investigation of Hand 180408 (+249 BB)...")
    trace_180408 = trace_single_hand_deep(180408, ppo_r2, model_a_r2, game)
    print(f"    Hand 180408: Pure={trace_180408['pure_return_bb']:.2f} BB, Hyb={trace_180408['hybrid_return_bb']:.2f} BB, Net={trace_180408['net_diff_bb']:+.2f} BB")
    
    print("\n[*] Phase 1: Deep Forensic Investigation of Hand 180158 (-117 BB)...")
    trace_180158 = trace_single_hand_deep(180158, ppo_r2, model_a_r2, game)
    print(f"    Hand 180158: Pure={trace_180158['pure_return_bb']:.2f} BB, Hyb={trace_180158['hybrid_return_bb']:.2f} BB, Net={trace_180158['net_diff_bb']:+.2f} BB")
    
    # Reset stats before running full 500-hand holdout benchmark
    model_a_ens.stats = {k: {sub: 0 for sub in v} for k, v in model_a_ens.stats.items()}
    model_a_r2.stats = {k: {sub: 0 for sub in v} for k, v in model_a_r2.stats.items()}
    
    # 2. Run 500-hand benchmark with full policy activation tracking
    print(f"\n[*] Phase 2: Running 500-hand Policy Activation Audit ({eval_seeds[0]} ~ {eval_seeds[-1]})...")
    
    agents = {
        "Pure_Ensemble": FlexibleHybridAgent(ppo_ens, ppo_ens, "ppo", "ppo"),
        "Hybrid_Ensemble": FlexibleHybridAgent(ppo_ens, model_a_ens, "ppo", "model_a"),
        "Pure_R2": FlexibleHybridAgent(ppo_r2, ppo_r2, "ppo", "ppo"),
        "Hybrid_R2": FlexibleHybridAgent(ppo_r2, model_a_r2, "ppo", "model_a")
    }
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    hand_returns = {name: [] for name in agents}
    postflop_reached = {name: 0 for name in agents}
    action_sequences = {name: [] for name in agents}
    
    for h_idx, seed_h in enumerate(eval_seeds):
        seat = h_idx % 5
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            pf_active = False
            act_list = []
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                rnd = get_round_from_state(st)
                if rnd > 0:
                    pf_active = True
                if cp == seat:
                    act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    act_list.append((rnd, act))
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
            action_sequences[name].append(act_list)
            if pf_active:
                postflop_reached[name] += 1
                
    # 3. Activation Decomposition
    def analyze_pair(pure_name, hyb_name):
        p_ret = np.array(hand_returns[pure_name]) * 100.0
        h_ret = np.array(hand_returns[hyb_name]) * 100.0
        diff = h_ret - p_ret
        
        p_acts = action_sequences[pure_name]
        h_acts = action_sequences[hyb_name]
        
        action_diff_count = 0
        action_diff_same_profit = 0
        profit_improved = 0
        profit_worsened = 0
        
        for i in range(len(p_ret)):
            act_diff = (p_acts[i] != h_acts[i])
            prof_diff = diff[i]
            if act_diff:
                action_diff_count += 1
                if abs(prof_diff) < 1e-4:
                    action_diff_same_profit += 1
            if prof_diff > 1e-4:
                profit_improved += 1
            elif prof_diff < -1e-4:
                profit_worsened += 1
                
        # Sensitivity Analysis (Trimming)
        # Sort diffs to find top and bottom
        sorted_diff = np.sort(diff)
        mean_all = float(np.mean(diff))
        sem_all = float(stats.sem(diff))
        
        # Exclude max (Top 1)
        mean_no_max = float(np.mean(sorted_diff[:-1]))
        sem_no_max = float(stats.sem(sorted_diff[:-1]))
        
        # Exclude min (Worst 1)
        mean_no_min = float(np.mean(sorted_diff[1:]))
        sem_no_min = float(stats.sem(sorted_diff[1:]))
        
        # Trimmed (exclude both)
        mean_trimmed = float(np.mean(sorted_diff[1:-1]))
        sem_trimmed = float(stats.sem(sorted_diff[1:-1]))
        
        return {
            "postflop_reached_hands": postflop_reached[pure_name],
            "action_diverged_hands": action_diff_count,
            "action_diverged_same_profit_hands": action_diff_same_profit,
            "profit_improved_hands": profit_improved,
            "profit_worsened_hands": profit_worsened,
            "mean_diff_bb100_all": safe_float(mean_all),
            "sem_diff_all": safe_float(sem_all),
            "ci95_all": [safe_float(mean_all - 1.96 * sem_all), safe_float(mean_all + 1.96 * sem_all)],
            "sensitivity": {
                "exclude_top1_max_gainer": {
                    "top1_value_bb": float(sorted_diff[-1]),
                    "mean_bb100": safe_float(mean_no_max),
                    "sem_bb100": safe_float(sem_no_max)
                },
                "exclude_worst1_max_loser": {
                    "worst1_value_bb": float(sorted_diff[0]),
                    "mean_bb100": safe_float(mean_no_min),
                    "sem_bb100": safe_float(sem_no_min)
                },
                "trimmed_both_excluded": {
                    "mean_bb100": safe_float(mean_trimmed),
                    "sem_bb100": safe_float(sem_trimmed)
                }
            }
        }

    r2_analysis = analyze_pair("Pure_R2", "Hybrid_R2")
    ens_analysis = analyze_pair("Pure_Ensemble", "Hybrid_Ensemble")
    
    payload = {
        "audit_version": "POKER AI V10.23.13",
        "forensic_seed_180408_top_gainer": trace_180408,
        "forensic_seed_180158_worst_loser": trace_180158,
        "dedicated_model_a_stats": {
            "Hybrid_R2": model_a_r2.stats,
            "Hybrid_Ensemble": model_a_ens.stats
        },
        "activation_and_sensitivity": {
            "Hybrid_R2_vs_Pure_R2": r2_analysis,
            "Hybrid_Ensemble_vs_Pure_Ensemble": ens_analysis
        }
    }
    
    out_path = "v1023/data/v1023_13_differential_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] V10.23.13 Differential Results written to {out_path}")

if __name__ == "__main__":
    main()
