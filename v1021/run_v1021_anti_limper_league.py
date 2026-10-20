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
from v1020.run_v1020_experiment_integrity_audit import get_comprehensive_hand_strength

# =====================================================================
# 1. LOOSE LIMPER BOT DEFINITION (The Causal Factor)
# =====================================================================
class LooseLimperBot:
    """Rigorous Loose-Limper: Calls 1 BB preflop, probes 0.5x pot on flop if checked to"""
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal_actions):
        sp = extract_player_spent(st)
        to_call = max(sp) - sp[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hs = get_comprehensive_hand_strength(inf)
        
        if to_call == 0:
            # Flop / Postflop probe: bet half pot if checked to
            if hs["is_good"] and 3 in legal_actions:
                return 3
            return 1 # Check
        else:
            if hs["is_monster"]:
                if 4 in legal_actions: return 4
                return 1
            if hs["is_good"] and to_call < 500: return 1
            # Limp factor: call 1 BB (to_call <= 100 chips)
            if to_call <= 100:
                return 1
            return 0 if 0 in legal_actions else 1

# =====================================================================
# 2. PPO TRAINING FUNCTION
# =====================================================================
def train_ppo_league_model(run_name, opp_type, train_seed, total_hands=50000):
    """
    opp_type: 'NFSP' (Control) or 'LooseLimper' (Experimental)
    """
    print(f"\n[*] Starting {run_name} (Opponent 4 = {opp_type}, Seed = {train_seed}, Hands = {total_hands})")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)
    
    model = PPONeuralPlayer(
        state_dim=1417, action_dim=6, hidden_dim=256,
        lr=0.0005, clip_eps=0.2, beta_entropy=0.01
    )
    
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    
    # 4 opponents: 3 fixed archetypes + 1 manipulated slot
    opp4 = bp_agent if opp_type == 'NFSP' else LooseLimperBot(seed=train_seed * 23)
    opp_pool = [
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
        
        opps = [random.choice(opp_pool) for _ in range(4)]
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
        
        # PPO transition storage into rollout_buffer
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

            for _ in range(4): # 4 epochs
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

                    # Ratio & Clipping
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
            
    # Save model checkpoint
    model_path = f"v1021/models/{run_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[+] Saved {run_name} to {model_path} (Total time: {time.time()-t0:.1f}s)")
    return model

# =====================================================================
# 3. COMPREHENSIVE EVALUATION BENCHMARK
# =====================================================================
def evaluate_model_battery(models_dict, eval_seeds=[95001, 95002, 95003, 95004, 95005], hands_per_cond=400):
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    
    # 3 Evaluation Conditions
    # 1. Seen_NFSP: Balanced, Tight, Ultra, NFSP (Retention check)
    # 2. Test_LooseLimper: Balanced, Tight, Ultra, LooseLimper (Exploitation check)
    # 3. Unseen_Adaptive: Balanced, Tight, Ultra, AdaptiveHistoryExploiter (Generalization check)
    from v1020.run_v1020_experiment_integrity_audit import RigorousAdaptiveHistoryExploiter
    
    eval_conditions = [
        ("Seen_NFSP", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                DeterministicUltraAggressiveBot(seed=s*19), bp_agent]),
        ("Test_LooseLimper", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                       DeterministicUltraAggressiveBot(seed=s*19), LooseLimperBot(seed=s*23)]),
        ("Unseen_Adaptive", lambda s: [DeterministicBalancedBot(rng_seed=s*13), DeterministicTightPassiveBot(seed=s*17),
                                      DeterministicUltraAggressiveBot(seed=s*19), RigorousAdaptiveHistoryExploiter(window_size=30)])
    ]
    
    results = {c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict} for c_name, _ in eval_conditions}
    
    print("\n=======================================================")
    print("[*] V10.21 EVALUATION BATTERY (Retention, Exploitation, Generalization)")
    print(f"[*] {len(models_dict)} Models x 3 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print("=======================================================")
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_s = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen in eval_conditions:
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
    for c_name, _ in eval_conditions:
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
    parser.add_argument("--hands", type=int, default=50000)
    args = parser.parse_args()
    
    seeds = [101, 202, 303]
    
    # 1. Train Control PPO (NFSP as opp 4)
    print("\n=======================================================")
    print("PHASE 1: TRAINING 3 CONTROL PPO MODELS (NFSP in Slot 4)")
    print("=======================================================")
    control_models = {}
    for r_idx, s in enumerate(seeds, 1):
        name = f"Control_R{r_idx}_{args.hands//1000}K"
        m = train_ppo_league_model(name, opp_type="NFSP", train_seed=s, total_hands=args.hands)
        control_models[name] = m
        
    # 2. Train Experimental PPO (LooseLimper as opp 4)
    print("\n=======================================================")
    print("PHASE 2: TRAINING 3 EXPERIMENTAL PPO MODELS (LooseLimper in Slot 4)")
    print("=======================================================")
    anti_models = {}
    for r_idx, s in enumerate(seeds, 1):
        name = f"AntiLimper_R{r_idx}_{args.hands//1000}K"
        m = train_ppo_league_model(name, opp_type="LooseLimper", train_seed=s, total_hands=args.hands)
        anti_models[name] = m
        
    # 3. Comprehensive Evaluation Battery
    print("\n=======================================================")
    print("PHASE 3: COMPREHENSIVE BENCHMARK EVALUATION")
    print("=======================================================")
    bp = DeterministicTrueNFSP_Agent()
    eval_pool = {
        "Model_A": ModelA(bp, sims=300),
        **control_models,
        **anti_models
    }
    
    summary = evaluate_model_battery(eval_pool, hands_per_cond=400)
    
    output = {
        "hands_per_model": args.hands,
        "seeds": seeds,
        "summary": summary
    }
    
    out_path = "v1021/data/v1021_league_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] Full experiment results saved to {out_path}")

if __name__ == "__main__":
    main()
