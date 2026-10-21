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

class IsolatedInstrumentedModelA:
    def __init__(self, bp, sims=300):
        self.bp = bp
        self.sims = sims
        self.temp = 2.0
        # Isolated Street-level counters
        self.street_stats = {
            "Flop": {"equity_calls": 0, "fallback_calls": 0},
            "Turn": {"equity_calls": 0, "fallback_calls": 0},
            "River": {"equity_calls": 0, "fallback_calls": 0}
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
            if street_name in self.street_stats:
                self.street_stats[street_name]["equity_calls"] += 1
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
                "street": street_name,
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
            if street_name in self.street_stats:
                self.street_stats[street_name]["fallback_calls"] += 1
            bp_rng = np.random.RandomState((hand_seed * 20011 + cp * 307 + dec_idx * 31) & 0xFFFFFFFF)
            act = self.bp.select_action_deterministic(inf, l6, bp_rng)
            self.last_meta = {
                "branch": "NFSP_Fallback",
                "street": street_name,
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

def run_worker(output_json_path):
    # Set Seed
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
    
    bp_agent = DeterministicTrueNFSP_Agent()
    model_a = IsolatedInstrumentedModelA(bp_agent, sims=300)
    
    game = get_5p_fullgame()
    eval_seeds = [180000 + i for i in range(500)]
    
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
    traces = {name: [] for name in agents}
    
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
            act_list = []
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes_persistent(st, chance_rng)
                    continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == seat:
                    act = agent.act(st, seat, l6, f_set, seed_h, dec)
                    raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                    act_list.append(act)
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
            traces[name].append(act_list)
            
    payload = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "eval_seeds": [eval_seeds[0], eval_seeds[-1]],
        "elapsed_seconds": time.time() - t0,
        "street_stats": model_a.street_stats,
        "returns": returns,
        "action_traces_hash": hashlib.sha256(json.dumps(traces).encode('utf-8')).hexdigest()
    }
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Worker] Run completed. Output written to {output_json_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "v1023/data/worker_test.json"
    run_worker(out_path)
