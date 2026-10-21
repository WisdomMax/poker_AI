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
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1021.run_v1021_anti_limper_league import LooseLimperBot

# -------------------------------------------------------------
# PHASE 1: DYNAMIC SHA-256 HASH VERIFICATION
# -------------------------------------------------------------
def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

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

    def get_action_probs(self, inf, legal_actions):
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
        return np.mean(probs_list, axis=0).tolist()

class InstrumentedModelA(ModelA):
    def __init__(self, bp_agent, sims=300):
        super().__init__(bp_agent, sims)
        self.equity_calls = 0
        self.nfsp_fallbacks = 0
        self.last_decision_meta = {}
        
    def act_with_meta(self, st, seat, l6, f_set, seed_h, dec):
        # We trace the internal decision branch of Model A
        sp = extract_player_spent(st)
        tc = max(sp) - sp[seat]
        pot = sum(sp)
        rnd = get_round_from_state(st)
        
        # Internal Model A logic
        act = super().act(st, seat, l6, f_set, seed_h, dec)
        
        # Determine whether it was an equity guard branch or fallback
        # In ModelA: if facing bet (tc > 0), it estimates equity vs pot odds
        # If tc == 0: it checks equity vs threshold for bet/check
        # If outside equity heuristics, it calls bp_agent
        was_fallback = (self.last_decision_meta.get("fallback", False))
        if was_fallback:
            self.nfsp_fallbacks += 1
        else:
            self.equity_calls += 1
            
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

ACTION_NAMES = {0: "Fold", 1: "CheckCall", 2: "HalfPot", 3: "Pot", 4: "DoublePot", 5: "Allin"}

def trace_hand_103_forensic(game, models, ppo_ens, model_a):
    print("\n" + "="*70)
    print("[*] PHASE 2: FORENSIC INVESTIGATION OF HAND 103 (Seed 160103)")
    print("="*70)
    
    seed_h = 160103
    seat = seed_h % 5 # Seat 3
    chance_rng = np.random.RandomState(seed_h)
    
    st = game.new_initial_state()
    advance_chance_nodes_persistent(st, chance_rng)
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    opps = opp_factory(seed_h)
    
    # State inspection
    hero_cards = st.information_state_string(seat)
    print(f"Hero Seat: {seat}, Seed: {seed_h}")
    print(f"Initial State Info: {hero_cards[:100]}...")
    
    # Trace Pure Ensemble
    st_pure = game.new_initial_state()
    advance_chance_nodes_persistent(st_pure, np.random.RandomState(seed_h))
    f_set = set()
    dec = 0
    pure_log = []
    while not st_pure.is_terminal():
        if st_pure.is_chance_node():
            advance_chance_nodes_persistent(st_pure, np.random.RandomState(seed_h))
            continue
        cp = st_pure.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_pure, cp)
        rnd = get_round_from_state(st_pure)
        sp = extract_player_spent(st_pure)
        pot = sum(sp)
        tc = max(sp) - sp[cp]
        
        if cp == seat:
            inf = np.array(st_pure.information_state_tensor(seat), dtype=np.float32)
            act = ppo_ens.act(inf, l6)
            probs = ppo_ens.get_action_probs(inf, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_pure, cp, act)
            pure_log.append({
                "round": rnd, "dec": dec, "pot": pot, "to_call": tc,
                "action": ACTION_NAMES[act], "raw_amt": raw_act, "probs": probs
            })
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opps[opp_idx]
            if isinstance(bot, DeterministicTrueNFSP_Agent):
                inf_b = np.array(st_pure.information_state_tensor(cp), dtype=np.float32)
                act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
            else:
                act = bot.select_action(st_pure, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_pure, cp, act)
        if act == 0:
            f_set.add(cp)
        st_pure.apply_action(raw_act)
    pure_ret = st_pure.returns()[seat] / 100.0
    
    # Trace Hybrid Ensemble
    st_hyb = game.new_initial_state()
    advance_chance_nodes_persistent(st_hyb, np.random.RandomState(seed_h))
    f_set = set()
    dec = 0
    hyb_log = []
    while not st_hyb.is_terminal():
        if st_hyb.is_chance_node():
            advance_chance_nodes_persistent(st_hyb, np.random.RandomState(seed_h))
            continue
        cp = st_hyb.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_hyb, cp)
        rnd = get_round_from_state(st_hyb)
        sp = extract_player_spent(st_hyb)
        pot = sum(sp)
        tc = max(sp) - sp[cp]
        
        if cp == seat:
            if rnd == 0:
                inf = np.array(st_hyb.information_state_tensor(seat), dtype=np.float32)
                act = ppo_ens.act(inf, l6)
                branch = "PPO_Preflop"
                eq = None
            else:
                act = model_a.act(st_hyb, seat, l6, f_set, seed_h, dec)
                branch = "ModelA_EquityHeuristic"
                eq = model_a._estimate_mc_equity(st_hyb, seat, f_set, 300)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_hyb, cp, act)
            hyb_log.append({
                "round": rnd, "dec": dec, "pot": pot, "to_call": tc,
                "action": ACTION_NAMES[act], "raw_amt": raw_act, "branch": branch, "mc_equity": eq
            })
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opps[opp_idx]
            if isinstance(bot, DeterministicTrueNFSP_Agent):
                inf_b = np.array(st_hyb.information_state_tensor(cp), dtype=np.float32)
                act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h * 101 + cp * 17))
            else:
                act = bot.select_action(st_hyb, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_hyb, cp, act)
        if act == 0:
            f_set.add(cp)
        st_hyb.apply_action(raw_act)
    hyb_ret = st_hyb.returns()[seat] / 100.0
    
    print("\n[+] Pure Ensemble Hand 103 Trace:")
    for l in pure_log:
        print(f"    Rnd {l['round']} | Pot: {l['pot']} | ToCall: {l['to_call']} | Action: {l['action']} ({l['raw_amt']}) | Probs: {[round(x, 3) for x in l['probs']]}")
    print(f"    Pure Return: {pure_ret:.2f} BB")
    
    print("\n[+] Hybrid Ensemble Hand 103 Trace:")
    for l in hyb_log:
        print(f"    Rnd {l['round']} | Pot: {l['pot']} | ToCall: {l['to_call']} | Action: {l['action']} ({l['raw_amt']}) | Branch: {l['branch']} | Equity: {l['mc_equity']}")
    print(f"    Hybrid Return: {hyb_ret:.2f} BB")
    print(f"    Net Saved by Model A: {hyb_ret - pure_ret:+.2f} BB")
    
    return {
        "hand_seed": seed_h,
        "hero_seat": seat,
        "pure_return": pure_ret,
        "hybrid_return": hyb_ret,
        "net_diff_bb": hyb_ret - pure_ret,
        "pure_log": pure_log,
        "hybrid_log": hyb_log
    }

def run_v1023_10_main():
    print("=" * 70)
    print("POKER AI V10.23.10: HYBRID REPRODUCIBILITY & INTERACTION VALIDATION")
    print("=" * 70)
    
    # 0. Global Seed Fix
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 1. Dynamic SHA-256 Hash Calculation
    ckpt_paths = [
        "v1022/models/StrictAntiLimper_R1_100K.pt",
        "v1022/models/StrictAntiLimper_R2_100K.pt",
        "v1022/models/StrictAntiLimper_R3_100K.pt",
    ]
    computed_hashes = {}
    for p in ckpt_paths:
        h = get_file_sha256(p)
        computed_hashes[os.path.basename(p)] = h
        print(f"[*] Verified SHA-256 for {p}: {h}")
        
    device = torch.device('cpu')
    anti_models = []
    for p in ckpt_paths:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        anti_models.append(m)
        
    ppo_r2 = PPOAgentWrapper(anti_models[1])
    ppo_ens = EnsembleAgentWrapper(anti_models)
    
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = ModelA(bp_agent, sims=300)
    
    game = get_5p_fullgame()
    
    # Phase 2: Hand 103 Trace
    hand_103_forensic = trace_hand_103_forensic(game, anti_models, ppo_ens, model_a)
    
    # Phase 3 & 4: 2x2 Factorial on NEW Independent Holdout Seeds (170,000 ~ 170,499)
    NUM_HANDS = 500
    new_eval_seeds = [170000 + i for i in range(NUM_HANDS)]
    print(f"\n[*] PHASE 3 & 4: 2x2 Factorial on NEW Holdout Seeds ({new_eval_seeds[0]} ~ {new_eval_seeds[-1]})...")
    
    agents = {
        "Policy_A_Pure_R2": FlexibleHybridAgent(ppo_r2, ppo_r2, "ppo", "ppo"),
        "Policy_B_Hybrid_R2_MA": FlexibleHybridAgent(ppo_r2, model_a, "ppo", "model_a"),
        "Policy_C_Inverted_MA_R2": FlexibleHybridAgent(model_a, ppo_r2, "model_a", "ppo"),
        "Policy_D_Pure_MA": FlexibleHybridAgent(model_a, model_a, "model_a", "model_a"),
        "Policy_E_Pure_Ensemble": FlexibleHybridAgent(ppo_ens, ppo_ens, "ppo", "ppo"),
        "Policy_F_Hybrid_Ens_MA": FlexibleHybridAgent(ppo_ens, model_a, "ppo", "model_a"),
    }
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    hand_returns = {name: [] for name in agents}
    hand_postflop_decs = {name: [] for name in agents}
    
    t0 = time.time()
    for h_idx, seed_h in enumerate(new_eval_seeds):
        if (h_idx + 1) % 100 == 0:
            print(f"    Progress: {h_idx+1}/{len(new_eval_seeds)} ({time.time()-t0:.1f}s)")
            
        seat = h_idx % 5
        
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            pf_decs = 0
            
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
            
    print(f"[*] Benchmark finished in {time.time()-t0:.1f}s.")
    
    # -------------------------------------------------------------
    # 2x2 RIGOROUS FACTORIAL HYPOTHESIS TESTING
    # -------------------------------------------------------------
    y_A = np.array(hand_returns["Policy_A_Pure_R2"]) * 100.0
    y_B = np.array(hand_returns["Policy_B_Hybrid_R2_MA"]) * 100.0
    y_C = np.array(hand_returns["Policy_C_Inverted_MA_R2"]) * 100.0
    y_D = np.array(hand_returns["Policy_D_Pure_MA"]) * 100.0
    y_E = np.array(hand_returns["Policy_E_Pure_Ensemble"]) * 100.0
    y_F = np.array(hand_returns["Policy_F_Hybrid_Ens_MA"]) * 100.0
    
    # 1. Simple Contrasts
    def paired_stat(diff_series):
        m = np.mean(diff_series)
        sem = stats.sem(diff_series)
        t_stat, p_val = stats.ttest_1samp(diff_series, 0.0)
        return {
            "mean_diff_bb100": safe_float(m),
            "sem_diff": safe_float(sem),
            "t_statistic": safe_float(t_stat),
            "p_value": safe_float(p_val),
            "ci95_diff": [safe_float(m - 1.96 * sem), safe_float(m + 1.96 * sem)]
        }
        
    contrasts = {
        "B_minus_A (Hybrid R2 vs Pure R2)": paired_stat(y_B - y_A),
        "D_minus_C (Pure MA vs Inverted R2)": paired_stat(y_D - y_C),
        "A_minus_C (Pure R2 vs Inverted R2)": paired_stat(y_A - y_C),
        "B_minus_D (Hybrid R2 vs Pure MA)": paired_stat(y_B - y_D),
        "F_minus_E (Hybrid Ens vs Pure Ens)": paired_stat(y_F - y_E),
        "F_minus_D (Hybrid Ens vs Pure MA)": paired_stat(y_F - y_D)
    }
    
    # 2. Main Effects and Interaction with Formal t-tests
    # Preflop Main Effect vector: 0.5 * ((B + A) - (D + C))
    preflop_vector = 0.5 * ((y_B + y_A) - (y_D + y_C))
    preflop_main_effect_test = paired_stat(preflop_vector)
    
    # Postflop Main Effect vector: 0.5 * ((B + D) - (A + C))
    postflop_vector = 0.5 * ((y_B + y_D) - (y_A + y_C))
    postflop_main_effect_test = paired_stat(postflop_vector)
    
    # Interaction vector: (B - A) - (D - C)
    interaction_vector = (y_B - y_A) - (y_D - y_C)
    interaction_test = paired_stat(interaction_vector)
    
    # Summary of individual policies
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
            "total_hands": NUM_HANDS,
            "postflop_hands": int(np.sum(pf_arr > 0)),
            "total_postflop_decisions": int(np.sum(pf_arr))
        }

    final_payload = {
        "dynamic_hashes": computed_hashes,
        "eval_seeds_holdout": [new_eval_seeds[0], new_eval_seeds[-1]],
        "hand_103_forensic": hand_103_forensic,
        "summary": summary,
        "simple_contrasts": contrasts,
        "factorial_2x2_formal_tests": {
            "preflop_main_effect_R2_over_MA": preflop_main_effect_test,
            "postflop_main_effect_MA_over_R2": postflop_main_effect_test,
            "interaction_effect_formal": interaction_test
        }
    }
    
    out_path = "v1023/data/v1023_10_interaction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] V10.23.10 Results successfully written to {out_path}")

if __name__ == "__main__":
    run_v1023_10_main()
