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
        return 0.0
    return float(val)

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def advance_chance_nodes_persistent(state, rng):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = rng.choice(actions, p=probs)
        state.apply_action(action)

# Hard Hash Verification
EXPECTED_HASHES = {
    "StrictAntiLimper_R1_100K.pt": "1e29ff9262f2649a96074022a57489fe56ede464168aef92b2d3e221cc67c580",
    "StrictAntiLimper_R2_100K.pt": "42f0f6b9ff0fefaaab9e3214a0e97e22179b22560de844131aa8a3bba34b76f7",
    "StrictAntiLimper_R3_100K.pt": "961174c8f0778002a9ace26a86b944d93a5d303c3d97ffe32aa0d131f0c67a56"
}

def verify_all_model_hashes():
    computed = {}
    for filename, exp_hash in EXPECTED_HASHES.items():
        path = os.path.join("v1022/models", filename)
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        actual_hash = get_file_sha256(path)
        assert actual_hash == exp_hash, f"Hash mismatch for {filename}! Exp: {exp_hash}, Act: {actual_hash}"
        computed[filename] = actual_hash
    print("[+] Phase 1: All SHA-256 model hashes verified with 100% hard assertions.")
    return computed

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

class FullyInstrumentedModelA:
    def __init__(self, bp, sims=300):
        self.bp = bp
        self.sims = sims
        self.temp = 2.0
        self.stats = {"equity_calls": 0, "fallback_calls": 0}
        self.last_meta = {}
        
    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        pot = sum(sp)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        
        if tc > 0 and 0 in l6:
            self.stats["equity_calls"] += 1
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
                "branch": "Equity_Heuristic",
                "tc": tc, "pot": pot, "pot_odds": po,
                "mc_equity": eq, "active_opps": ao
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
            self.stats["fallback_calls"] += 1
            bp_rng = np.random.RandomState((hand_seed * 20011 + cp * 307 + dec_idx * 31) & 0xFFFFFFFF)
            act = self.bp.select_action_deterministic(inf, l6, bp_rng)
            self.last_meta = {
                "branch": "NFSP_Fallback",
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

def repair_and_replay_hand_103(game, ppo_ens, model_a):
    print("\n" + "="*70)
    print("[*] PHASE 2: EXACT REPAIR & REPLAY OF HAND 103 (Seed 160103)")
    print("="*70)
    
    seed_h = 160103
    seat = seed_h % 5 # Seat 3
    
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    # 1. Replay Pure Ensemble with FRESH opps and SINGLE chance_rng
    opps_pure = opp_factory(seed_h)
    chance_rng_pure = np.random.RandomState(seed_h)
    st_pure = game.new_initial_state()
    advance_chance_nodes_persistent(st_pure, chance_rng_pure)
    f_set_pure = set()
    dec = 0
    pure_trace = []
    
    while not st_pure.is_terminal():
        if st_pure.is_chance_node():
            advance_chance_nodes_persistent(st_pure, chance_rng_pure)
            continue
        cp = st_pure.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_pure, cp)
        rnd = get_round_from_state(st_pure)
        
        if cp == seat:
            inf = np.array(st_pure.information_state_tensor(seat), dtype=np.float32)
            act = ppo_ens.act(inf, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_pure, cp, act)
            pure_trace.append({"round": rnd, "dec": dec, "act": ACTION_NAMES[act], "raw_amt": raw_act})
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

    # 2. Replay Hybrid Ensemble with FRESH opps and SINGLE chance_rng
    opps_hyb = opp_factory(seed_h)
    chance_rng_hyb = np.random.RandomState(seed_h)
    st_hyb = game.new_initial_state()
    advance_chance_nodes_persistent(st_hyb, chance_rng_hyb)
    f_set_hyb = set()
    dec = 0
    hyb_trace = []
    
    hybrid_agent = FlexibleHybridAgent(ppo_ens, model_a, "ppo", "model_a")
    while not st_hyb.is_terminal():
        if st_hyb.is_chance_node():
            advance_chance_nodes_persistent(st_hyb, chance_rng_hyb)
            continue
        cp = st_hyb.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st_hyb, cp)
        rnd = get_round_from_state(st_hyb)
        
        if cp == seat:
            act = hybrid_agent.act(st_hyb, seat, l6, f_set_hyb, seed_h, dec)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st_hyb, cp, act)
            meta = model_a.last_meta if rnd > 0 else {"branch": "PPO_Preflop"}
            hyb_trace.append({"round": rnd, "dec": dec, "act": ACTION_NAMES[act], "raw_amt": raw_act, "meta": meta})
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
    
    diff_bb = hyb_ret - pure_ret
    print(f"[+] Replay Pure Return:   {pure_ret:+.2f} BB | Trace: {pure_trace}")
    print(f"[+] Replay Hybrid Return: {hyb_ret:+.2f} BB | Trace: {hyb_trace}")
    print(f"[+] Replay Net Difference: {diff_bb:+.2f} BB")
    
    # Assert exact replication of V10.23.9 results
    assert abs(pure_ret - (-100.0)) < 1e-4, f"Pure return mismatch: {pure_ret}"
    assert abs(hyb_ret - (-30.25)) < 1e-4, f"Hybrid return mismatch: {hyb_ret}"
    assert abs(diff_bb - 69.75) < 1e-4, f"Diff mismatch: {diff_bb}"
    print("[✓] EXACT VERIFICATION: Hand 103 +69.75 BB save successfully replicated with bit-exact precision!")
    
    return {
        "pure_return": pure_ret,
        "hybrid_return": hyb_ret,
        "net_diff_bb": diff_bb,
        "pure_trace": pure_trace,
        "hybrid_trace": hyb_trace
    }

def run_confirmatory_benchmark(run_id, eval_seeds, game, ppo_ens, ppo_r2, model_a):
    print(f"\n[*] Running Confirmatory Benchmark [Run {run_id}] across {len(eval_seeds)} hands...")
    opp_factory = lambda s: [
        DeterministicBalancedBot(rng_seed=s * 13),
        DeterministicTightPassiveBot(seed=s * 17),
        RealisticAggressiveBot(seed=s * 19),
        LooseLimperBot(seed=s * 23)
    ]
    
    agents = {
        "Pure_Ensemble": FlexibleHybridAgent(ppo_ens, ppo_ens, "ppo", "ppo"),
        "Hybrid_Ensemble": FlexibleHybridAgent(ppo_ens, model_a, "ppo", "model_a"),
        "Pure_R2": FlexibleHybridAgent(ppo_r2, ppo_r2, "ppo", "ppo"),
        "Hybrid_R2": FlexibleHybridAgent(ppo_r2, model_a, "ppo", "model_a")
    }
    
    returns = {name: [] for name in agents}
    t0 = time.time()
    
    for h_idx, seed_h in enumerate(eval_seeds):
        seat = h_idx % 5
        for name, agent in agents.items():
            opps = opp_factory(seed_h)
            chance_rng = np.random.RandomState(seed_h)
            st = game.new_initial_state()
            advance_chance_nodes_persistent(st, chance_rng)
            f_set = set()
            dec = 0
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == seat:
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
            returns[name].append(ret_bb)
            
    print(f"    Run {run_id} completed in {time.time()-t0:.1f}s.")
    return returns

def main():
    print("=" * 70)
    print("POKER AI V10.23.11: REPLAY REPAIR & HYBRID CONFIRMATORY TEST")
    print("=" * 70)
    
    # Global Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Hard Hash Verification
    verified_hashes = verify_all_model_hashes()
    
    # Load Models
    device = torch.device('cpu')
    anti_models = []
    for r in range(1, 4):
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(f"v1022/models/StrictAntiLimper_R{r}_100K.pt", map_location=device))
        m.eval()
        anti_models.append(m)
        
    ppo_ens = EnsembleAgentWrapper(anti_models)
    ppo_r2 = PPOAgentWrapper(anti_models[1])
    
    # Deterministic NFSP fallback with fixed seed 42
    torch.manual_seed(42)
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = FullyInstrumentedModelA(bp_agent, sims=300)
    
    game = get_5p_fullgame()
    
    # Phase 2: Repair & Replay Hand 103
    h103_res = repair_and_replay_hand_103(game, ppo_ens, model_a)
    
    # Phase 3 & 4: Independent Confirmatory Benchmark on BRAND NEW Seeds (180,000 ~ 180,499)
    # Executing 2 Independent Runs to guarantee Bit-Exact Reproducibility
    eval_seeds = [180000 + i for i in range(500)]
    
    returns_run1 = run_confirmatory_benchmark(1, eval_seeds, game, ppo_ens, ppo_r2, model_a)
    returns_run2 = run_confirmatory_benchmark(2, eval_seeds, game, ppo_ens, ppo_r2, model_a)
    
    # Bit-exact check
    hash1 = hashlib.sha256(json.dumps(returns_run1).encode('utf-8')).hexdigest()
    hash2 = hashlib.sha256(json.dumps(returns_run2).encode('utf-8')).hexdigest()
    print(f"\n[*] Run 1 Returns Hash: {hash1}")
    print(f"[*] Run 2 Returns Hash: {hash2}")
    assert hash1 == hash2, "Reproducibility failure between Run 1 and Run 2!"
    print("[✓] EXACT REPRODUCIBILITY GUARANTEED: 100.0% Bit-exact match across independent runs!")
    
    # Statistical Hypothesis Testing
    # Primary Hypothesis: Hybrid Ensemble vs Pure Ensemble
    y_hyb_ens = np.array(returns_run1["Hybrid_Ensemble"]) * 100.0
    y_pure_ens = np.array(returns_run1["Pure_Ensemble"]) * 100.0
    diff_ens = y_hyb_ens - y_pure_ens
    t_ens, p_ens = stats.ttest_1samp(diff_ens, 0.0)
    sem_ens = stats.sem(diff_ens)
    m_ens = np.mean(diff_ens)
    
    # Secondary Hypothesis: Hybrid R2 vs Pure R2
    y_hyb_r2 = np.array(returns_run1["Hybrid_R2"]) * 100.0
    y_pure_r2 = np.array(returns_run1["Pure_R2"]) * 100.0
    diff_r2 = y_hyb_r2 - y_pure_r2
    t_r2, p_r2 = stats.ttest_1samp(diff_r2, 0.0)
    sem_r2 = stats.sem(diff_r2)
    m_r2 = np.mean(diff_r2)
    
    # Holm-Bonferroni correction (2 pre-specified tests)
    # p_values: [p_ens, p_r2]
    raw_p = [p_ens, p_r2]
    sorted_indices = np.argsort(raw_p)
    holm_p = np.zeros(2)
    holm_p[sorted_indices[0]] = min(1.0, raw_p[sorted_indices[0]] * 2)
    holm_p[sorted_indices[1]] = min(1.0, max(holm_p[sorted_indices[0]], raw_p[sorted_indices[1]] * 1))
    
    confirmatory_results = {
        "eval_seeds": [eval_seeds[0], eval_seeds[-1]],
        "run_hashes": {"run1": hash1, "run2": hash2, "bit_exact": (hash1 == hash2)},
        "model_a_instrumentation": model_a.stats,
        "hand_103_repair": h103_res,
        "summary": {
            "Pure_Ensemble": {
                "mean_bb100": safe_float(np.mean(y_pure_ens)),
                "sem_bb100": safe_float(stats.sem(y_pure_ens)),
                "ci95": [safe_float(np.mean(y_pure_ens) - 1.96 * stats.sem(y_pure_ens)), safe_float(np.mean(y_pure_ens) + 1.96 * stats.sem(y_pure_ens))]
            },
            "Hybrid_Ensemble": {
                "mean_bb100": safe_float(np.mean(y_hyb_ens)),
                "sem_bb100": safe_float(stats.sem(y_hyb_ens)),
                "ci95": [safe_float(np.mean(y_hyb_ens) - 1.96 * stats.sem(y_hyb_ens)), safe_float(np.mean(y_hyb_ens) + 1.96 * stats.sem(y_hyb_ens))]
            },
            "Pure_R2": {
                "mean_bb100": safe_float(np.mean(y_pure_r2)),
                "sem_bb100": safe_float(stats.sem(y_pure_r2)),
                "ci95": [safe_float(np.mean(y_pure_r2) - 1.96 * stats.sem(y_pure_r2)), safe_float(np.mean(y_pure_r2) + 1.96 * stats.sem(y_pure_r2))]
            },
            "Hybrid_R2": {
                "mean_bb100": safe_float(np.mean(y_hyb_r2)),
                "sem_bb100": safe_float(stats.sem(y_hyb_r2)),
                "ci95": [safe_float(np.mean(y_hyb_r2) - 1.96 * stats.sem(y_hyb_r2)), safe_float(np.mean(y_hyb_r2) + 1.96 * stats.sem(y_hyb_r2))]
            }
        },
        "hypothesis_tests": {
            "Primary_Hybrid_Ens_vs_Pure_Ens": {
                "mean_diff_bb100": safe_float(m_ens),
                "sem_diff": safe_float(sem_ens),
                "t_statistic": safe_float(t_ens),
                "nominal_p_value": safe_float(p_ens),
                "holm_bonferroni_p_value": safe_float(holm_p[0]),
                "ci95_diff": [safe_float(m_ens - 1.96 * sem_ens), safe_float(m_ens + 1.96 * sem_ens)]
            },
            "Secondary_Hybrid_R2_vs_Pure_R2": {
                "mean_diff_bb100": safe_float(m_r2),
                "sem_diff": safe_float(sem_r2),
                "t_statistic": safe_float(t_r2),
                "nominal_p_value": safe_float(p_r2),
                "holm_bonferroni_p_value": safe_float(holm_p[1]),
                "ci95_diff": [safe_float(m_r2 - 1.96 * sem_r2), safe_float(m_r2 + 1.96 * sem_r2)]
            }
        }
    }
    
    out_path = "v1023/data/v1023_11_confirmatory_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(confirmatory_results, f, indent=2)
    print(f"\n[+] V10.23.11 Final Confirmatory Results saved to {out_path}")

if __name__ == "__main__":
    main()
