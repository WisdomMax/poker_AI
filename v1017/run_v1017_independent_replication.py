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
    get_pool_standard, get_pool_aggressive, get_pool_dynamic,
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

# =====================================================================
# V10.17 INDEPENDENT FROM-SCRATCH PPO TRAINING PIPELINE
# =====================================================================
def run_single_ppo_training(run_id, train_seed, total_hands=100000):
    print(f"\n=======================================================")
    print(f"[*] Starting Run {run_id}: From-Scratch PPO Training (Seed={train_seed}, Hands={total_hands})")
    print(f"=======================================================")
    
    # 1. Deterministic Initialization
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)
    
    model_e = PPONeuralPlayer(
        state_dim=1417, action_dim=6, hidden_dim=256,
        lr=0.0005, clip_eps=0.2, beta_entropy=0.01
    )
    
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    
    opp_pool_archetypes = [
        DeterministicBalancedBot(rng_seed=train_seed * 13),
        DeterministicTightPassiveBot(seed=train_seed * 17),
        DeterministicUltraAggressiveBot(seed=train_seed * 19),
        bp_agent
    ]
    
    action_counts = defaultdict(int)
    street_actions = defaultdict(lambda: defaultdict(int)) # street -> act -> count
    
    checkpoints = {}
    train_diagnostics = []
    
    kl_list = []
    clip_frac_list = []
    actor_loss_list = []
    critic_loss_list = []
    entropy_list = []
    
    t0 = time.time()
    
    for h in range(1, total_hands + 1):
        rng = np.random.RandomState(train_seed * 100000 + h)
        seat = h % 5
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        opps = [random.choice(opp_pool_archetypes) for _ in range(4)]
        hand_trajectory = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
            cp = st.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st, cp)
            
            if cp == seat:
                inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                act, log_prob, val = model_e.act_train(inf, l6)
                action_counts[act] += 1
                
                n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
                street = "Preflop" if n_b == 0 else ("Flop" if n_b == 3 else ("Turn" if n_b == 4 else "River"))
                street_actions[street][act] += 1
                
                hand_trajectory.append((normalize_inf_state_v79(inf), act, log_prob, val, l6))
            else:
                opp_idx = cp if cp < seat else cp - 1
                opp_bot = opps[opp_idx]
                if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                    inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(h*101 + cp*17))
                else:
                    act = opp_bot.select_action(st, cp, l6)
                    
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            
        ret = st.returns()[seat] / 100.0
        r_norm = ret / 100.0
        
        n_steps = len(hand_trajectory)
        for i in range(n_steps):
            s_t, a_t, lp_t, v_t, leg_t = hand_trajectory[i]
            R_t = r_norm * (0.99 ** (n_steps - 1 - i))
            adv_t = R_t - v_t
            mask_t = [1 if k in leg_t else 0 for k in range(6)]
            model_e.rollout_buffer.append((s_t, a_t, lp_t, R_t, adv_t, mask_t))
            
        # PPO Update every 200 hands
        if h % 200 == 0 and len(model_e.rollout_buffer) >= 64:
            # We track KL, clip fraction, entropy, losses
            buf = model_e.rollout_buffer
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
            
            epoch_kls = []
            epoch_clips = []
            epoch_act_losses = []
            epoch_crit_losses = []
            epoch_entropies = []

            for _ in range(4): # 4 epochs
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    b_idx = indices[start:end]

                    b_s = s_t[b_idx]; b_a = a_t[b_idx]; b_old_lp = old_lp_t[b_idx]
                    b_ret = ret_t[b_idx]; b_adv = adv_t[b_idx]; b_mask = mask_t[b_idx]

                    model_e.train()
                    logits, v_pred = model_e.forward(b_s)
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

                    model_e.optimizer.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(model_e.parameters(), max_norm=0.5)
                    model_e.optimizer.step()
                    
                    # Diagnostics
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                        clipped = ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()
                        epoch_kls.append(approx_kl)
                        epoch_clips.append(clipped)
                        epoch_act_losses.append(loss_actor.item())
                        epoch_crit_losses.append(loss_critic.item())
                        epoch_entropies.append(entropy.item())

            model_e.rollout_buffer = []
            kl_list.append(float(np.mean(epoch_kls)))
            clip_frac_list.append(float(np.mean(epoch_clips)))
            actor_loss_list.append(float(np.mean(epoch_act_losses)))
            critic_loss_list.append(float(np.mean(epoch_crit_losses)))
            entropy_list.append(float(np.mean(epoch_entropies)))

        # Save Checkpoints at 50K and 100K
        if h in [50000, 100000]:
            k_val = h // 1000
            save_path = f"v1017/models/model_e_r{run_id}_{k_val}k.pt"
            torch.save(model_e.state_dict(), save_path)
            checkpoints[f"{k_val}k"] = save_path
            
            tot_acts = sum(action_counts.values())
            print(f"  * [Run {run_id} | Checkpoint {k_val}K ({h} hands)]: Saved -> {save_path} (Time: {time.time()-t0:.1f}s)")
            print(f"    - Actions: Fold={action_counts[0]/tot_acts*100:.1f}%, Call={action_counts[1]/tot_acts*100:.1f}%, "
                  f"Min={action_counts[2]/tot_acts*100:.1f}%, Half={action_counts[3]/tot_acts*100:.1f}%, "
                  f"Pot={action_counts[4]/tot_acts*100:.1f}%, AllIn={action_counts[5]/tot_acts*100:.1f}%")
            print(f"    - Recent Diag (last 50 updates): Actor Loss={np.mean(actor_loss_list[-50:]):.4f}, "
                  f"Critic Loss={np.mean(critic_loss_list[-50:]):.4f}, Entropy={np.mean(entropy_list[-50:]):.4f} nats, "
                  f"KL={np.mean(kl_list[-50:]):.5f}, ClipFrac={np.mean(clip_frac_list[-50:]):.3f}")

    return {
        "run_id": run_id,
        "train_seed": train_seed,
        "checkpoints": checkpoints,
        "total_time_s": time.time() - t0,
        "action_counts": dict(action_counts),
        "street_actions": {k: dict(v) for k, v in street_actions.items()},
        "recent_kl": float(np.mean(kl_list[-50:])),
        "recent_clip_frac": float(np.mean(clip_frac_list[-50:])),
        "recent_entropy": float(np.mean(entropy_list[-50:])),
        "recent_actor_loss": float(np.mean(actor_loss_list[-50:])),
        "recent_critic_loss": float(np.mean(critic_loss_list[-50:])),
    }

# =====================================================================
# SYSTEMATIC REPLICATION EVALUATION HARNESS
# =====================================================================
def evaluate_all_models(models_dict, eval_seeds, hands_per_seed=1000):
    print(f"\n=======================================================")
    print(f"[*] Starting Systematic Replication Evaluation")
    print(f"[*] {len(models_dict)} Models x {len(eval_seeds)} Eval Seeds x {hands_per_seed} Hands")
    print(f"=======================================================")
    
    game = get_5p_fullgame()
    eval_pools = [
        ("Standard", get_pool_standard),
        ("Aggressive", get_pool_aggressive),
        ("Dynamic", get_pool_dynamic)
    ]
    
    # Store returns: model -> seed -> list of returns
    model_returns = {m_name: {s: [] for s in eval_seeds} for m_name in models_dict}
    
    t0 = time.time()
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        print(f"  -> Evaluating Seed {s_idx}/{len(eval_seeds)}: Seed Base {s_base}...")
        for h in range(hands_per_seed):
            seed_h = s_base + h
            seat = h % 5
            p_name, p_gen = eval_pools[h % 3]
            
            # Common deal & opponents for identical cards across all evaluated models
            opps = p_gen(seed_h)
            
            for m_name, model in models_dict.items():
                st = game.new_initial_state()
                advance_chance_nodes(st, np.random.RandomState(seed_h))
                
                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, np.random.RandomState(seed_h))
                        continue
                    cp = st.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                    
                    if cp == seat:
                        inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                        if isinstance(model, ModelA):
                            act = model.select_action(st, seat, l6)
                        elif isinstance(model, PPONeuralPlayer):
                            # Deterministic evaluation using seed_h
                            act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                        else:
                            act = model.select_action(st, cp, l6)
                    else:
                        opp_idx = cp if cp < seat else cp - 1
                        opp_bot = opps[opp_idx]
                        if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                            inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(seed_h*101 + cp*17))
                        else:
                            act = opp_bot.select_action(st, cp, l6)
                            
                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                    
                ret_bb = st.returns()[seat] / 100.0
                model_returns[m_name][s_base].append(ret_bb)

    # Compute summary statistics
    summary = {}
    for m_name in models_dict:
        seed_means = [np.mean(model_returns[m_name][s]) * 100.0 for s in eval_seeds]
        all_rets = [r for s in eval_seeds for r in model_returns[m_name][s]]
        
        overall_mean = float(np.mean(seed_means))
        seed_se = float(np.std(seed_means, ddof=1) / np.sqrt(len(eval_seeds)))
        t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
        ci_95 = [float(overall_mean - t_crit * seed_se), float(overall_mean + t_crit * seed_se)]
        
        summary[m_name] = {
            "overall_mean": overall_mean,
            "seed_se": seed_se,
            "ci_95": ci_95,
            "seed_means": seed_means,
            "total_hands": len(all_rets)
        }
        print(f"  * {m_name:12s}: Mean = {overall_mean:+7.2f} bb/100 | SE = {seed_se:5.2f} | 95% CI = [{ci_95[0]:+7.2f}, {ci_95[1]:+7.2f}] | Seeds = {[round(x,1) for x in seed_means]}")
        
    return summary, model_returns

# =====================================================================
# MAIN EXPERIMENT EXECUTION
# =====================================================================
def main():
    print("=======================================================================")
    print("POKER AI V10.17: INDEPENDENT PPO TRAINING REPLICATION")
    print("Rigorous 3-Run From-Scratch Replication & Scaling Audit")
    print("=======================================================================")
    
    train_seeds = [101, 202, 303]
    eval_seeds = [90001, 90002, 90003, 90004, 90005]
    
    # 1. Run 3 Independent 100K Trainings
    run_records = []
    checkpoint_models = {}
    
    for idx, t_seed in enumerate(train_seeds, 1):
        record = run_single_ppo_training(run_id=idx, train_seed=t_seed, total_hands=100000)
        run_records.append(record)
        
    print("\n[+] All 3 Independent 100K Trainings Completed Successfully!")
    
    # 2. Load Checkpoint Models
    eval_models = {}
    # Frozen Model A baseline
    eval_models["Model_A"] = ModelA()
    
    for r_idx in range(1, 4):
        # 50K
        m_50k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_50k.load_state_dict(torch.load(f"v1017/models/model_e_r{r_idx}_50k.pt"))
        m_50k.eval()
        eval_models[f"E_R{r_idx}_50K"] = m_50k
        
        # 100K
        m_100k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_100k.load_state_dict(torch.load(f"v1017/models/model_e_r{r_idx}_100k.pt"))
        m_100k.eval()
        eval_models[f"E_R{r_idx}_100K"] = m_100k
        
    # Also include V10.16 Model E10 (Champion from previous run) for direct lineage comparison
    if os.path.exists("v1016/models/model_e_100k.pt"):
        m_v1016 = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_v1016.load_state_dict(torch.load("v1016/models/model_e_100k.pt"))
        m_v1016.eval()
        eval_models["E_V1016_100K"] = m_v1016
        
    # 3. Evaluate All Models across 5 Eval Seeds
    eval_summary, raw_returns = evaluate_all_models(eval_models, eval_seeds, hands_per_seed=1000)
    
    # 4. Statistical Tests: 50K vs 100K paired t-tests per run
    paired_tests = {}
    for r_idx in range(1, 4):
        m50 = f"E_R{r_idx}_50K"
        m100 = f"E_R{r_idx}_100K"
        s50 = eval_summary[m50]["seed_means"]
        s100 = eval_summary[m100]["seed_means"]
        diffs = np.array(s100) - np.array(s50)
        mean_diff = float(np.mean(diffs))
        se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        t_stat, p_val = stats.ttest_rel(s100, s50)
        t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
        ci_diff = [float(mean_diff - t_crit * se_diff), float(mean_diff + t_crit * se_diff)]
        
        paired_tests[f"Run_{r_idx}"] = {
            "mean_50k": eval_summary[m50]["overall_mean"],
            "mean_100k": eval_summary[m100]["overall_mean"],
            "mean_diff": mean_diff,
            "se_diff": se_diff,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "ci_95": ci_diff,
            "seed_diffs": [float(x) for x in diffs]
        }
        print(f"\n[+] Run {r_idx} Scaling Test (E_100K - E_50K):")
        print(f"    - Difference: {mean_diff:+7.2f} ± {se_diff:5.2f} bb/100 (95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}])")
        print(f"    - t-statistic = {t_stat:.3f}, p-value = {p_val:.5f} (df=4)")
        
    # Meta-analysis across all 3 runs
    pooled_50k = [eval_summary[f"E_R{r}_50K"]["overall_mean"] for r in range(1, 4)]
    pooled_100k = [eval_summary[f"E_R{r}_100K"]["overall_mean"] for r in range(1, 4)]
    meta_diff = np.mean(pooled_100k) - np.mean(pooled_50k)
    print(f"\n[+] Meta-Analysis Across 3 Runs:")
    print(f"    - Mean 50K = {np.mean(pooled_50k):+7.2f} bb/100 (Runs: {[round(x,1) for x in pooled_50k]})")
    print(f"    - Mean 100K = {np.mean(pooled_100k):+7.2f} bb/100 (Runs: {[round(x,1) for x in pooled_100k]})")
    print(f"    - Meta Improvement = {meta_diff:+7.2f} bb/100")
    
    # Two-Way ANOVA / Variance Decomposition (Run vs Eval Seed)
    # Factor A: Training Run (3 levels), Factor B: Eval Seed (5 levels) for 100K models
    data_matrix = np.array([eval_summary[f"E_R{r}_100K"]["seed_means"] for r in range(1, 4)]) # shape (3, 5)
    grand_mean = np.mean(data_matrix)
    ss_total = np.sum((data_matrix - grand_mean)**2)
    ss_run = 5 * np.sum((np.mean(data_matrix, axis=1) - grand_mean)**2)
    ss_seed = 3 * np.sum((np.mean(data_matrix, axis=0) - grand_mean)**2)
    ss_error = ss_total - ss_run - ss_seed
    
    variance_decomp = {
        "ss_total": float(ss_total),
        "ss_run": float(ss_run),
        "ss_seed": float(ss_seed),
        "ss_error": float(ss_error),
        "run_variance_pct": float(ss_run / ss_total * 100.0) if ss_total > 0 else 0.0,
        "seed_variance_pct": float(ss_seed / ss_total * 100.0) if ss_total > 0 else 0.0,
        "residual_pct": float(ss_error / ss_total * 100.0) if ss_total > 0 else 0.0
    }
    print(f"\n[+] Variance Decomposition (100K Models):")
    print(f"    - Training Run Variance (Run-to-Run): {variance_decomp['run_variance_pct']:.2f}%")
    print(f"    - Eval Seed Variance (Card Distribution): {variance_decomp['seed_variance_pct']:.2f}%")
    print(f"    - Residual / Interaction: {variance_decomp['residual_pct']:.2f}%")
    
    # Save full experimental results
    output_data = {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "run_records": run_records,
        "eval_summary": eval_summary,
        "paired_tests": paired_tests,
        "meta_analysis": {
            "pooled_50k_mean": float(np.mean(pooled_50k)),
            "pooled_100k_mean": float(np.mean(pooled_100k)),
            "meta_diff": float(meta_diff),
            "runs_50k": [float(x) for x in pooled_50k],
            "runs_100k": [float(x) for x in pooled_100k]
        },
        "variance_decomposition": variance_decomp
    }
    
    with open("v1017/data/v1017_replication_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Saved full results to v1017/data/v1017_replication_results.json")

if __name__ == "__main__":
    main()
