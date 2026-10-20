import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
)
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1020.run_v1020_experiment_integrity_audit import (
    RigorousCheckRaiseTrapBot, RigorousPolarizedBluffValueBot,
)
from v1021.run_v1021_anti_limper_league import LooseLimperBot

def run_v1021_1_rigorous_benchmark():
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    eval_seeds = [96001, 96002, 96003, 96004, 96005]
    hands_per_cond = 400
    
    # 4 Strict Conditions (Exact 1-seat replacement: Seats 0,1,2 fixed, Seat 3 manipulated)
    conditions = [
        ("Strict_Seen_NFSP", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                        DeterministicUltraAggressiveBot(seed=s*19), bp_agent]),
        ("Strict_Test_Limper", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                          DeterministicUltraAggressiveBot(seed=s*19), LooseLimperBot(seed=s*23)]),
        ("True_Unseen_Trap", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                        DeterministicUltraAggressiveBot(seed=s*19), RigorousCheckRaiseTrapBot(seed=s*31)]),
        ("True_Unseen_Polarized", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                            DeterministicUltraAggressiveBot(seed=s*19), RigorousPolarizedBluffValueBot(seed=s*29)]),
    ]
    
    # Load 7 models
    models_dict = {"Model_A": ModelA(bp_agent, sims=300)}
    for r_idx in range(1, 4):
        c_m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        c_m.load_state_dict(torch.load(f"v1021/models/Control_R{r_idx}_100K.pt"))
        c_m.eval()
        models_dict[f"Control_R{r_idx}_100K"] = c_m
        
        a_m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        a_m.load_state_dict(torch.load(f"v1021/models/AntiLimper_R{r_idx}_100K.pt"))
        a_m.eval()
        models_dict[f"AntiLimper_R{r_idx}_100K"] = a_m

    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _ in conditions}
    
    print("\n=======================================================")
    print("[*] V10.21.1 RIGOROUS STRICT BENCHMARK EVALUATION")
    print(f"[*] 7 Models x 4 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print("=======================================================")
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_s = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen in conditions:
            for h in range(hands_per_cond):
                seed_h = s_base + h
                seat = h % 5
                opps = p_gen(seed_h)
                
                for m_name, model in models_dict.items():
                    st = game.new_initial_state()
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    f_set = set(); dec = 0
                    
                    while not st.is_terminal():
                        if st.is_chance_node(): advance_chance_nodes(st, np.random.RandomState(seed_h)); continue
                        cp = st.current_player()
                        l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                        
                        if cp == seat:
                            inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                            if isinstance(model, ModelA): act = model.act(st, seat, l6, f_set, seed_h, dec)
                            elif isinstance(model, PPONeuralPlayer): act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                            dec += 1
                        else:
                            opp_idx = cp if cp < seat else cp - 1
                            bot = opps[opp_idx]
                            if isinstance(bot, DeterministicTrueNFSP_Agent):
                                inf_b = np.array(st.information_state_tensor(cp), dtype=np.float32)
                                act = bot.select_action_deterministic(inf_b, l6, np.random.RandomState(seed_h*101 + cp*17))
                            else:
                                act = bot.select_action(st, cp, l6)
                                
                        if act == 0: f_set.add(cp)
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[c_name][m_name][s_base].append(ret_bb)
                    
        print(f"     [Seed {s_base} completed in {time.time()-t_s:.1f}s]", flush=True)
        
    summary = {}
    for c_name, _ in conditions:
        summary[c_name] = {}
        for m_name in models_dict:
            s_means = [np.mean(results[c_name][m_name][s]) * 100.0 for s in eval_seeds]
            m_val = float(np.mean(s_means))
            se_val = float(np.std(s_means, ddof=1) / np.sqrt(len(eval_seeds)))
            t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
            ci = [float(m_val - t_crit * se_val), float(m_val + t_crit * se_val)]
            summary[c_name][m_name] = {
                "mean": m_val, "se": se_val, "ci_95": ci, "seed_means": s_means
            }

    output = {
        "eval_seeds": eval_seeds,
        "hands_per_cond": hands_per_cond,
        "summary": summary
    }
    
    out_path = "v1021_1/data/v1021_1_strict_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] Strict results saved to {out_path}")

if __name__ == "__main__":
    run_v1021_1_rigorous_benchmark()
