#!/usr/bin/env python3
"""
DIRECTIVE 043: Universal Poker AlphaGo
PHASE C: 2,000-Hand Multi-Agent Self-Play Pilot & Baseline Benchmark Runner

Executes:
1. 2,000 Hands of 5-Player Simultaneous Self-Play (Agent A~E)
2. Saves independent checkpoints every 500 hands (500h, 1000h, 1500h, 2000h)
3. Records hand-by-hand raw JSONL with full 5-player ledger and zero-sum verification
4. Benchmarks top evolving agent against Frozen Baseline 50k model over 500 hands
5. Exports comprehensive results to v1024/directive043_selfplay_results.json
"""
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive040_regression import robust_parse_cards, parse_board_cards
from v1024.alphago_5p_selfplay import MultiAgentPokerAlphaGoEngine, AGENT_NAMES

BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
RAW_JSONL_PATH = "v1024/data/directive043_raw_selfplay_2000h.jsonl"
RESULTS_JSON_PATH = "v1024/directive043_selfplay_results.json"

def run_2000h_pilot_and_benchmark(num_hands: int = 2000, base_seed: int = 430000):
    print("=" * 80)
    print(f"DIRECTIVE 043: 5-PLAYER SELF-PLAY PILOT ({num_hands} hands)")
    print(f"Base Checkpoint: {BASE_50K_PATH}")
    print("=" * 80)

    engine = MultiAgentPokerAlphaGoEngine(
        base_ckpt_path=BASE_50K_PATH,
        log_file="v1024/alphago_pilot_2000h.log",
        base_seed=base_seed
    )

    os.makedirs(os.path.dirname(RAW_JSONL_PATH), exist_ok=True)
    raw_jsonl_file = open(RAW_JSONL_PATH, "w", encoding="utf-8")

    interval_stats = defaultdict(lambda: defaultdict(Counter))
    start_time = time.time()

    for h_idx in range(1, num_hands + 1):
        h_seed = base_seed + h_idx
        chance_rng = np.random.RandomState(h_seed)

        st = engine.game.new_initial_state()
        advance_chance_nodes(st, chance_rng)

        # Dynamic seat rotation
        seat_to_agent = {s: (s - (h_idx % 5)) % 5 for s in range(5)}
        agent_to_seat = {a_id: s for s, a_id in seat_to_agent.items()}

        priv_cards = robust_parse_cards(st)
        trajectories = {a_id: [] for a_id in range(5)}
        hand_action_log = []

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue

            cp = st.current_player()
            agent_id = seat_to_agent[cp]
            agent = engine.agents[agent_id]

            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            round_idx = get_round_from_state(st)

            raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            norm_inf = normalize_inf_state_v79(raw_inf)

            act_6, log_prob, val, probs = agent.select_action(norm_inf, legal_6, round_idx)
            trajectories[agent_id].append((norm_inf, act_6, log_prob, val, legal_6))
            hand_action_log.append({"seat": cp, "agent": agent.name, "round": round_idx, "action": act_6})

            # Track interval statistics (per 500 hands)
            interval_block = ((h_idx - 1) // 500) * 500 + 500
            interval_stats[interval_block][agent.name][act_6] += 1

            nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
            st.apply_action(nat_act)

        returns_chips = st.returns()
        board_cards = parse_board_cards(st)

        # Zero-Sum Verification
        tot_return = sum(returns_chips)
        assert abs(tot_return) < 1e-4, f"Zero-sum broken in hand {h_idx}: {tot_return}"

        hand_pnls = {}
        for a_id in range(5):
            seat = agent_to_seat[a_id]
            pnl_dollars = returns_chips[seat]
            pnl_bb = pnl_dollars / 100.0
            engine.agents[a_id].total_pnl_bb += pnl_bb
            hand_pnls[AGENT_NAMES[a_id]] = round(pnl_bb, 2)

            traj = trajectories[a_id]
            n_steps = len(traj)
            for i in range(n_steps):
                s_t, a_t, lp_t, v_t, leg_t = traj[i]
                R_t = pnl_bb * (0.99 ** (n_steps - 1 - i))
                engine.agents[a_id].buffer_transition(s_t, a_t, lp_t, R_t, v_t, leg_t)

        # Record Raw JSONL
        hand_record = {
            "hand_idx": h_idx,
            "seed": h_seed,
            "agent_seats": {AGENT_NAMES[a_id]: agent_to_seat[a_id] for a_id in range(5)},
            "cards": {AGENT_NAMES[seat_to_agent[s]]: priv_cards[s] for s in range(5)},
            "board": board_cards,
            "actions": hand_action_log,
            "pnls_bb": hand_pnls,
            "zero_sum_check": round(tot_return, 4)
        }
        raw_jsonl_file.write(json.dumps(hand_record, ensure_ascii=False) + "\n")

        # Updates every 50 hands
        if h_idx % 50 == 0:
            for a in engine.agents:
                a.update_policy(batch_size=64, epochs=4)
            time.sleep(0.0005) # Micro-sleep for CPU <= 70%

        # Logging every 200 hands
        if h_idx % 200 == 0:
            engine.log(f"--- Progress [{h_idx:,} / {num_hands:,}] ---")
            for a in engine.agents:
                tot = sum(a.action_counts.values())
                allin_rate = (a.action_counts[5] / tot * 100.0) if tot > 0 else 0.0
                pot_rate = (a.action_counts[3] / tot * 100.0) if tot > 0 else 0.0
                half_rate = (a.action_counts[2] / tot * 100.0) if tot > 0 else 0.0
                ent = np.mean(a.entropy_history[-200:]) if a.entropy_history else 0.0
                engine.log(f"  {a.name} (Upd: {a.gradient_update_count}): PnL {a.total_pnl_bb:+.1f} BB | "
                           f"Ent {ent:.3f} | HalfPot {half_rate:.1f}%, Pot {pot_rate:.1f}%, Allin {allin_rate:.1f}%")

        # Save Milestones every 500 hands
        if h_idx % 500 == 0:
            engine.log(f"--> [MILESTONE {h_idx}] Saving checkpoints for 5 agents...")
            for a in engine.agents:
                p = f"v1024/models_alphago/{a.name}_{h_idx}h.pt"
                torch.save(a.model.state_dict(), p)

    raw_jsonl_file.close()
    total_elapsed = time.time() - start_time
    engine.log(f"Self-play pilot {num_hands} hands completed in {total_elapsed:.1f}s")

    # Determine Top Evolving Agent
    best_agent = max(engine.agents, key=lambda a: a.total_pnl_bb)
    engine.log(f"Top Evolving Agent: {best_agent.name} (Total PnL: {best_agent.total_pnl_bb:+.1f} BB)")

    # -------------------------------------------------------------------------
    # Independent Evaluation League: Top Evolving Agent vs Frozen Baseline 50k
    # -------------------------------------------------------------------------
    engine.log(f"Starting Independent Evaluation: {best_agent.name} vs Frozen Baseline 50k (500 hands)...")
    eval_game = get_5p_fullgame()
    base_model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    base_model.load_state_dict(torch.load(BASE_50K_PATH, map_location="cpu", weights_only=False))
    base_model.eval()

    eval_pnls = []
    eval_allin_counts = 0
    eval_pot_counts = 0
    eval_halfpot_counts = 0
    eval_total_actions = 0

    for eval_idx in range(500):
        e_seed = 999000 + eval_idx
        c_rng = np.random.RandomState(e_seed)
        st = eval_game.new_initial_state()
        advance_chance_nodes(st, c_rng)

        hero_seat = eval_idx % 5

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, c_rng)
                continue

            cp = st.current_player()
            legal = True6ActionAdapter.get_legal_6actions(st, cp)
            raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            norm_inf = normalize_inf_state_v79(raw_inf)

            if cp == hero_seat:
                # Evolving Agent (best_agent)
                act, _, _, _ = best_agent.select_action(norm_inf, legal, round_idx=0)
                eval_total_actions += 1
                if act == 5: eval_allin_counts += 1
                elif act == 3: eval_pot_counts += 1
                elif act == 2: eval_halfpot_counts += 1
            else:
                # Frozen Baseline 50k
                with torch.no_grad():
                    t_inf = torch.from_numpy(norm_inf).unsqueeze(0)
                    logits, _ = base_model(t_inf)
                    mask = torch.full_like(logits, -1e9)
                    for a in legal: mask[0, a] = 0.0
                    probs = F.softmax(logits + mask, dim=-1).squeeze(0).numpy()
                    act = int(np.random.choice(range(6), p=probs))

            nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            st.apply_action(nat_act)

        ret_bb = st.returns()[hero_seat] / 100.0
        eval_pnls.append(ret_bb)

    mean_eval_bb100 = float(np.mean(eval_pnls) * 100.0)
    std_eval = float(np.std(eval_pnls, ddof=1))
    ci95_eval = 1.96 * (std_eval / np.sqrt(len(eval_pnls))) * 100.0
    t_stat, p_val = stats.ttest_1samp(eval_pnls, 0.0)

    engine.log(f"Evaluation Complete! {best_agent.name} vs Baseline 50k: "
               f"{mean_eval_bb100:+.2f} ± {ci95_eval:.2f} BB/100 (p={p_val:.4e})")

    # Build Comprehensive Output Results JSON
    results = {
        "directive": "DIRECTIVE 043: Universal Poker AlphaGo",
        "num_selfplay_hands": num_hands,
        "elapsed_sec": round(total_elapsed, 1),
        "base_checkpoint_sha256": "fc1961160e2edc9ac17a48b51dc5a0eb1e43bf819758192ba8989c11131e2b8d",
        "agents": {
            a.name: {
                "final_pnl_bb": round(a.total_pnl_bb, 2),
                "gradient_updates": a.gradient_update_count,
                "action_counts": dict(a.action_counts),
                "avg_entropy": round(float(np.mean(a.entropy_history)), 3) if a.entropy_history else 0.0,
                "final_loss": a.loss_history[-1] if a.loss_history else {}
            } for a in engine.agents
        },
        "interval_action_distributions": {
            str(interval): {
                agent_name: {
                    str(act): count for act, count in act_counter.items()
                } for agent_name, act_counter in agents_dict.items()
            } for interval, agents_dict in interval_stats.items()
        },
        "independent_benchmark_vs_50k": {
            "hero_agent": best_agent.name,
            "benchmark_hands": 500,
            "bb_per_100": round(mean_eval_bb100, 2),
            "ci95_bb": round(ci95_eval, 2),
            "ci_range": [round(mean_eval_bb100 - ci95_eval, 2), round(mean_eval_bb100 + ci95_eval, 2)],
            "p_value": float(p_val),
            "allin_rate_pct": round(eval_allin_counts / max(eval_total_actions, 1) * 100.0, 2),
            "pot_rate_pct": round(eval_pot_counts / max(eval_total_actions, 1) * 100.0, 2),
            "halfpot_rate_pct": round(eval_halfpot_counts / max(eval_total_actions, 1) * 100.0, 2)
        }
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    engine.log(f"Saved complete results to {RESULTS_JSON_PATH}")
    print("=" * 80)
    print("DIRECTIVE 043 PHASE C 2,000-HAND RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_2000h_pilot_and_benchmark()
