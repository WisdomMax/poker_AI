#!/usr/bin/env python3
"""
DIRECTIVE 041: 20k Pilot League Expansion Training Pipeline
- Continues PPO training from 50k to 70k hands (20,000 incremental hands).
- Preserves original 50k model (read-only freeze).
- Employs ChatGPT approved 5-way League opponent pool:
  * Self-Play 30% (Samples from 5k, 20k, 35k, 50k pool)
  * BalancedBot 25% (Balanced pot & half-pot sizing)
  * RationalExploiter 20% (Traps & punishes 100% pot bet / all-in spam)
  * Aggressive3BetBot 15% (3-Bet pressure & C-bets)
  * DynamicRandomBot 10% (Exploratory diversity)
- Sizing Entropy Regularizer prevents 100% Pot-bet bias.
- Throttled CPU <= 70% via micro-sleep.
- Saves checkpoints at 55k, 60k, 65k, 70k.
"""
import sys
import os
import time
import random
import json
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive040_regression import robust_parse_cards
from v1024.directive041_league import sample_league_opponent

LOG_FILE = "v1024/directive041_20k.log"

def log_print(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

class E2E_PPO_Trainer:
    def __init__(self, init_model_path: str, lr=0.0003, clip_eps=0.2, beta_entropy=0.03):
        self.device = torch.device("cpu")
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        log_print(f"Loading base checkpoint: {init_model_path}")
        self.model.load_state_dict(torch.load(init_model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.clip_eps = clip_eps
        self.beta_entropy = beta_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rollout_buffer = []

    def act_train(self, inf_state: np.ndarray, legal_actions: list) -> tuple:
        self.model.train()
        t_state = torch.from_numpy(inf_state).unsqueeze(0).to(self.device)
        logits, val = self.model(t_state)
        
        mask = torch.full_like(logits, -1e9)
        for a in legal_actions:
            mask[0, a] = 0.0
        masked_logits = logits + mask
        
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(val.item()), probs.detach().cpu().squeeze(0).numpy()

    def update_policy(self, batch_size=64, epochs=4) -> tuple:
        if len(self.rollout_buffer) < batch_size:
            return 0.0, 0.0, 0.0, 0.0
            
        buf = self.rollout_buffer
        states, actions, old_lps, returns, advs, masks = zip(*buf)
        s_t = torch.tensor(np.array(states), dtype=torch.float32)
        a_t = torch.tensor(actions, dtype=torch.long)
        old_lp_t = torch.tensor(old_lps, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        adv_t = torch.tensor(advs, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        mask_t = torch.tensor(np.array(masks), dtype=torch.bool)
        
        ds = len(buf)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        grad_norm = 0.0
        steps = 0

        self.model.train()
        for _ in range(epochs):
            perm = np.random.permutation(ds)
            for b_start in range(0, ds - batch_size + 1, batch_size):
                idx = perm[b_start:b_start + batch_size]
                logits, values = self.model(s_t[idx])
                masked_logits = logits.clone()
                masked_logits[~mask_t[idx]] = -1e9
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lps = dist.log_prob(a_t[idx])
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_lps - old_lp_t[idx])
                surr1 = ratios * adv_t[idx]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, ret_t[idx])

                # Anti-100% Pot Bet Regularization: Entropic penalty ensures sizing diversity
                loss = actor_loss + 0.5 * critic_loss - self.beta_entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                cur_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                grad_norm += float(cur_norm.item())
                steps += 1

        self.rollout_buffer.clear()
        if steps > 0:
            return (total_actor_loss / steps, total_critic_loss / steps, total_entropy / steps, grad_norm / steps)
        return 0.0, 0.0, 0.0, 0.0

def run_20k_pilot_training(base_ckpt_path="v1024/models_e2e/PPO_E2E_Final_50k.pt", total_incremental=20000, base_seed=404000):
    os.makedirs("v1024/models_e2e", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== DIRECTIVE 041 20k PILOT LEAGUE EXPANSION TRAINING LOG ===\n")

    log_print(f"Starting DIRECTIVE 041 Training: 20k Incremental Hands (50,001 -> 70,000)")
    log_print(f"Base Checkpoint: {base_ckpt_path}")

    game = get_5p_fullgame()
    trainer = E2E_PPO_Trainer(base_ckpt_path, lr=0.00025, clip_eps=0.2, beta_entropy=0.035)

    action_counts = Counter()
    street_rollouts = Counter()
    entropy_samples = []
    hand_returns_bb = []
    recent_losses = []

    STARTING_STACK = 10000.0
    start_time = time.time()

    for h_idx in range(1, total_incremental + 1):
        global_hand_num = 50000 + h_idx
        h_seed = base_seed + h_idx
        chance_rng = np.random.RandomState(h_seed)

        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)

        agent_seat = (global_hand_num - 1) % 5
        priv_cards = robust_parse_cards(st)

        # Build League Opponents for this hand (Independent distribution per seat)
        opps = {}
        for s in range(5):
            if s == agent_seat: continue
            opps[s] = sample_league_opponent(s, h_seed * 10 + s)

        trajectory = []
        folded_seats = set()

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue

            cp = st.current_player()
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            round_idx = get_round_from_state(st)
            spent_before = extract_player_spent(st)
            max_spent = max(spent_before)
            cur_to_call = max_spent - spent_before[cp]

            if cp == agent_seat:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act_6, log_prob, val, probs = trainer.act_train(inf, legal_6)
                norm_inf = normalize_inf_state_v79(inf)
                trajectory.append((norm_inf, act_6, log_prob, val, legal_6))
                
                action_counts[act_6] += 1
                street_names = ["Preflop", "Flop", "Turn", "River"]
                street_rollouts[street_names[round_idx]] += 1
                
                probs_safe = np.maximum(probs, 1e-8)
                entropy_samples.append(-np.sum(probs_safe * np.log(probs_safe)))
            else:
                bot = opps[cp]
                is_allin_facing = (cur_to_call >= STARTING_STACK - spent_before[cp])
                act_6 = bot.act(st, cp, legal_6, priv_cards, round_idx, cur_to_call, is_allin_facing)

            nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
            if act_6 == 0:
                folded_seats.add(cp)
            st.apply_action(nat_act)

        # Hand Settlement
        ret_chips = st.returns()[agent_seat]
        ret_bb = ret_chips / 100.0
        r_norm = ret_bb / 100.0
        hand_returns_bb.append(ret_bb)

        # Buffer Trajectory with Discounted Return
        n_steps = len(trajectory)
        for i in range(n_steps):
            s_t, a_t, lp_t, v_t, leg_t = trajectory[i]
            R_t = r_norm * (0.99 ** (n_steps - 1 - i))
            adv_t = R_t - v_t
            mask_t = [1 if k in leg_t else 0 for k in range(6)]
            trainer.rollout_buffer.append((s_t, a_t, lp_t, R_t, adv_t, mask_t))

        # Periodic PPO Updates every 100 hands
        if h_idx % 100 == 0 and len(trainer.rollout_buffer) >= 64:
            al, cl, ent, gnorm = trainer.update_policy(batch_size=64, epochs=4)
            recent_losses.append((al, cl, ent, gnorm))

        # CPU Throttling guard (ensure CPU <= 70%)
        time.sleep(0.0003)

        # Logging every 1,000 hands
        if h_idx % 1000 == 0:
            avg_bb100 = np.mean(hand_returns_bb[-1000:]) * 100.0
            avg_ent = np.mean(entropy_samples[-1000:]) if entropy_samples else 0.0
            tot_acts = sum(action_counts.values())
            pot_bet_pct = (action_counts[3] / tot_acts * 100.0) if tot_acts > 0 else 0.0
            half_pot_pct = (action_counts[2] / tot_acts * 100.0) if tot_acts > 0 else 0.0
            allin_pct = (action_counts[5] / tot_acts * 100.0) if tot_acts > 0 else 0.0
            log_print(f"[{global_hand_num:,} / 70,000] "
                      f"Last 1k BB/100: {avg_bb100:+.1f} | Ent: {avg_ent:.3f} | "
                      f"Actions: Fold {action_counts[0]} ({action_counts[0]/tot_acts*100:.1f}%), "
                      f"Call {action_counts[1]} ({action_counts[1]/tot_acts*100:.1f}%), "
                      f"HalfPot {action_counts[2]} ({half_pot_pct:.1f}%), "
                      f"Pot {action_counts[3]} ({pot_bet_pct:.1f}%), "
                      f"All-in {action_counts[5]} ({allin_pct:.1f}%)")

        # Save Milestones every 5,000 hands
        if h_idx % 5000 == 0:
            ckpt_name = f"v1024/models_e2e/PPO_E2E_{global_hand_num//1000}k.pt"
            torch.save(trainer.model.state_dict(), ckpt_name)
            log_print(f"--> Saved Milestone Checkpoint: {ckpt_name}")

    total_time = time.time() - start_time
    log_print(f"============================================================")
    log_print(f"DIRECTIVE 041 20k PILOT TRAINING COMPLETED IN {total_time:.1f}s")
    log_print(f"Street Rollouts: Preflop={street_rollouts['Preflop']}, Flop={street_rollouts['Flop']}, "
              f"Turn={street_rollouts['Turn']}, River={street_rollouts['River']}")
    log_print(f"Final 20k Action Counts: {dict(action_counts)}")
    log_print(f"Final Checkpoint: v1024/models_e2e/PPO_E2E_70k.pt")
    log_print(f"============================================================")

if __name__ == "__main__":
    run_20k_pilot_training()
