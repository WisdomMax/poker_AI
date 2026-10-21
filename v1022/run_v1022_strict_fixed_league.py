import os
import sys
import json
import time
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1020.run_v1020_experiment_integrity_audit import (
    get_comprehensive_hand_strength,
    RigorousCheckRaiseTrapBot, RigorousPolarizedBluffValueBot,
)
from v1021.run_v1021_anti_limper_league import LooseLimperBot

# =====================================================================
# 1. STRICT 1-SEAT FIXED PPO TRAINING PIPELINE
# =====================================================================
def train_strict_fixed_model(run_name, opp_type, train_seed, total_hands=100000):
    print(f"\n[*] Starting Strict 1-Seat Training: {run_name} (Slot 3 = {opp_type}, Seed = {train_seed}, Hands = {total_hands})")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)
    
    model = PPONeuralPlayer(
        state_dim=1417, action_dim=6, hidden_dim=256,
        lr=0.0005, clip_eps=0.2, beta_entropy=0.01
    )
    
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    
    # STRICT 1-SEAT FIXED ASSIGNMENT (NO RANDOM CHOICE, NO REPLACEMENT)
    opp4 = bp_agent if opp_type == 'NFSP' else LooseLimperBot(seed=train_seed * 23)
    fixed_opps = [
        DeterministicBalancedBot(rng_seed=train_seed * 13),
        DeterministicTightPassiveBot(seed=train_seed * 17),
        DeterministicUltraAggressiveBot(seed=train_seed * 19),
        opp4
    ]
    
    t0 = time.time()
    for h in range(1, total_hands + 1):
        rng = np.random.RandomState(train_seed * 100000 + h)
        seat = h % 5
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        # Exact strict mapping to 4 opponent seats
        opps = fixed_opps
        hand_trajectory = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
                
            cp = st.current_player()
            legal = True6ActionAdapter.get_legal_6actions(st, cp)
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            
            if cp == seat:
                act, log_prob, val = model.act_train(inf, legal)
                norm_inf = normalize_inf_state_v79(inf)
                hand_trajectory.append((norm_inf, act, log_prob, val, legal))
            else:
                opp_idx = cp if cp < seat else cp - 1
                bot = opps[opp_idx]
                if isinstance(bot, DeterministicTrueNFSP_Agent):
                    act = bot.select_action_deterministic(inf, legal, rng)
                else:
                    act = bot.select_action(st, cp, legal)
                    
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            
        ret_chips = st.returns()[seat]
        ret_bb = ret_chips / 100.0
        r_norm = ret_bb / 100.0
        
        # PPO transition storage
        n_steps = len(hand_trajectory)
        for i in range(n_steps):
            s_t, a_t, lp_t, v_t, leg_t = hand_trajectory[i]
            R_t = r_norm * (0.99 ** (n_steps - 1 - i))
            adv_t = R_t - v_t
            mask_t = [1 if k in leg_t else 0 for k in range(6)]
            model.rollout_buffer.append((s_t, a_t, lp_t, R_t, adv_t, mask_t))
            
        # PPO Update every 200 hands
        if h % 200 == 0 and len(model.rollout_buffer) >= 64:
            buf = model.rollout_buffer
            states, actions, old_log_probs, returns, advantages, masks = zip(*buf)
            s_t = torch.tensor(np.array(states), dtype=torch.float32)
            a_t = torch.tensor(actions, dtype=torch.long)
            old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)
            ret_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
            adv_t = torch.tensor(advantages, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            mask_t = torch.tensor(np.array(masks), dtype=torch.bool)

            dataset_size = len(buf)
            indices = np.arange(dataset_size)
            batch_size = 64

            for _ in range(4):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    b_idx = indices[start:end]

                    b_s = s_t[b_idx]; b_a = a_t[b_idx]; b_old_lp = old_lp_t[b_idx]
                    b_ret = ret_t[b_idx]; b_adv = adv_t[b_idx]; b_mask = mask_t[b_idx]

                    model.train()
                    logits, v_pred = model.forward(b_s)
                    masked_logits = logits.clone()
                    masked_logits[~b_mask] = -1e9
                    probs = F.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_lp = dist.log_prob(b_a)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - b_old_lp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * b_adv
                    loss_actor = -torch.min(surr1, surr2).mean()
                    loss_critic = F.mse_loss(v_pred, b_ret)
                    loss_total = loss_actor + 0.5 * loss_critic - 0.01 * entropy

                    model.optimizer.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    model.optimizer.step()

            model.rollout_buffer.clear()
            
        if h % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    [{run_name}] Progress: {h}/{total_hands} hands ({elapsed:.1f}s)", flush=True)
            
    model_path = f"v1022/models/{run_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[+] Saved {run_name} to {model_path} (Total time: {time.time()-t0:.1f}s)")
    return model

# =====================================================================
# 2. STRICT 4-BENCHMARK EVALUATION HARNESS
# =====================================================================
def evaluate_strict_battery(models_dict, eval_seeds=[97001, 97002, 97003, 97004, 97005], hands_per_cond=400):
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    
    # 4 Strict Conditions
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
    
    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _ in conditions}
    
    print("\n=======================================================")
    print("[*] V10.22 RIGOROUS STRICT BENCHMARK EVALUATION")
    print(f"[*] {len(models_dict)} Models x {len(conditions)} Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
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

    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=100000)
    args = parser.parse_args()
    
    seeds = [101, 202, 303]
    
    # 1. Train Control PPO (Strict 1-Seat Fixed: Slot 3 = NFSP)
    print("\n=======================================================")
    print("PHASE 1: TRAINING 3 STRICT CONTROL PPO (Slot 3 = NFSP Fixed)")
    print("=======================================================")
    control_models = {}
    for r_idx, s in enumerate(seeds, 1):
        name = f"StrictControl_R{r_idx}_{args.hands//1000}K"
        m = train_strict_fixed_model(name, opp_type="NFSP", train_seed=s, total_hands=args.hands)
        control_models[name] = m
        
    # 2. Train Experimental PPO (Strict 1-Seat Fixed: Slot 3 = LooseLimper)
    print("\n=======================================================")
    print("PHASE 2: TRAINING 3 STRICT EXPERIMENTAL PPO (Slot 3 = LooseLimper Fixed)")
    print("=======================================================")
    anti_models = {}
    for r_idx, s in enumerate(seeds, 1):
        name = f"StrictAntiLimper_R{r_idx}_{args.hands//1000}K"
        m = train_strict_fixed_model(name, opp_type="LooseLimper", train_seed=s, total_hands=args.hands)
        anti_models[name] = m
        
    # 3. Comprehensive Benchmark Evaluation
    print("\n=======================================================")
    print("PHASE 3: STRICT BENCHMARK EVALUATION (Seen, Limper, Trap, Polarized)")
    print("=======================================================")
    bp = DeterministicTrueNFSP_Agent()
    eval_pool = {
        "Model_A": ModelA(bp, sims=300),
        **control_models,
        **anti_models
    }
    
    summary = evaluate_strict_battery(eval_pool, hands_per_cond=400)
    
    output = {
        "hands_per_model": args.hands,
        "seeds": seeds,
        "summary": summary
    }
    
    out_path = "v1022/data/v1022_strict_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] V10.22 Full results saved to {out_path}")

if __name__ == "__main__":
    main()
