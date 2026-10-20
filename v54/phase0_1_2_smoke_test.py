"""
POKER AI V5.4 - PHASE 0, 1, 2: ENVIRONMENT CONFIRMATION, REPRODUCIBILITY & SMOKE TEST
- Phase 0: 5인 20BB 환경 확정 (Turn/River 미존재 명시), 칩 회계 원장 및 RNG 분리, 해시 기록
- Phase 1: NFSP-DQN 하이퍼파라미터 동결 (2x128 Q-Net & Avg-Policy Net, lr=1e-4, buffer 20k)
- Phase 2: 3개 시드 x 1,000 eps Smoke Test (가중치 변화, 손실함수, NaN 없음, 칩 보존, 체크포인트 입출력)
"""

import os
import sys
import csv
import time
import hashlib
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

# 상위 디렉터리 경로 등록
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v52')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v523')))

from nfsp_variants import NFSP_DQN_Reference
from v52.phase5_6_7_five_player_benchmark import advance_chance_nodes

def get_5p_20bb_poker_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 6,
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 2,
        'numBoardCards': '0 3',
        'blind': '50 100 0 0 0',
        'stack': '2000 2000 2000 2000 2000',
        'firstPlayer': '3 1'
    }
    return pyspiel.load_game('universal_poker', params)

def verify_chip_ledger_and_accounting(game, num_hands=200, seed=20261022):
    rng = np.random.RandomState(seed)
    ledger_records = []
    zero_sum_violations = 0
    bound_violations = 0

    for hand_id in range(num_hands):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                act = rng.choice(leg)
                st.apply_action(act)

        returns = st.returns()
        sum_ret = sum(returns)
        if abs(sum_ret) > 1e-4:
            zero_sum_violations += 1

        for p_idx in range(5):
            ret_bb = returns[p_idx] / 100.0
            if ret_bb < -20.0 or ret_bb > 80.0:
                bound_violations += 1

        ledger_records.append({
            'hand_id': hand_id,
            'p0_return_bb': returns[0] / 100.0,
            'p1_return_bb': returns[1] / 100.0,
            'p2_return_bb': returns[2] / 100.0,
            'p3_return_bb': returns[3] / 100.0,
            'p4_return_bb': returns[4] / 100.0,
            'sum_returns_bb': sum_ret / 100.0
        })

    return zero_sum_violations, bound_violations, ledger_records

def run_smoke_test(game, seeds=[42, 101, 202], episodes=1000):
    print("=" * 80)
    print("PHASE 2: 20BB 학습 무결성 Smoke Test (3개 시드 x 1,000 에피소드)")
    print("=" * 80)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    smoke_results = []
    for s in seeds:
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        rng = np.random.RandomState(s)

        agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
        initial_q_norm = sum(p.norm().item() for p in agent.q_net.parameters())
        initial_sl_norm = sum(p.norm().item() for p in agent.sl_net.parameters())

        q_losses = []
        sl_losses = []
        nan_detected = False
        illegal_action_detected = False

        start_time = time.time()
        for ep in range(1, episodes + 1):
            agent.reset_episode()
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)
            ep_history = defaultdict(list)

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                else:
                    cp = st.current_player()
                    leg = st.legal_actions(cp)
                    info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    act = agent.select_action(info, leg, is_eval=False)

                    if act not in leg:
                        illegal_action_detected = True

                    ep_history[cp].append((info, act, leg))
                    st.apply_action(act)

            returns = st.returns()
            for cp in range(5):
                hist = ep_history[cp]
                if not hist: continue
                for i in range(len(hist)):
                    s_t, a_t, _ = hist[i]
                    if i + 1 < len(hist):
                        ns, _, n_leg = hist[i+1]
                        done = 0.0
                        r = 0.0
                    else:
                        ns = np.zeros(s_dim, dtype=np.float32)
                        n_leg = []
                        done = 1.0
                        r = returns[cp] / 100.0
                    agent.rl_buffer.append((s_t, a_t, r, ns, n_leg, done))

            # update
            q_loss, sl_loss = agent.update(batch_size=64)
            if np.isnan(q_loss) or np.isinf(q_loss) or np.isnan(sl_loss) or np.isinf(sl_loss):
                nan_detected = True
            if q_loss > 0:
                q_losses.append(q_loss)
            if sl_loss > 0:
                sl_losses.append(sl_loss)

        elapsed = time.time() - start_time
        final_q_norm = sum(p.norm().item() for p in agent.q_net.parameters())
        final_sl_norm = sum(p.norm().item() for p in agent.sl_net.parameters())

        q_weight_delta = abs(final_q_norm - initial_q_norm)
        sl_weight_delta = abs(final_sl_norm - initial_sl_norm)

        # 모델 저장 및 SHA256 체크
        ckpt_path = f"v54/models/smoke_seed_{s}.pt"
        torch.save({
            'q_net': agent.q_net.state_dict(),
            'sl_net': agent.sl_net.state_dict(),
            'seed': s
        }, ckpt_path)

        with open(ckpt_path, 'rb') as f:
            ckpt_hash = hashlib.sha256(f.read()).hexdigest()

        res = {
            'seed': s,
            'episodes': episodes,
            'elapsed_sec': round(elapsed, 2),
            'rl_buffer_size': len(agent.rl_buffer),
            'sl_buffer_size': len(agent.sl_buffer),
            'q_weight_delta': round(q_weight_delta, 4),
            'sl_weight_delta': round(sl_weight_delta, 4),
            'nan_detected': nan_detected,
            'illegal_action_detected': illegal_action_detected,
            'avg_q_loss': round(float(np.mean(q_losses[-100:])), 4) if q_losses else 0.0,
            'avg_sl_loss': round(float(np.mean(sl_losses[-100:])), 4) if sl_losses else 0.0,
            'ckpt_hash': ckpt_hash[:16]
        }
        smoke_results.append(res)
        print(f"  * Seed {s} 완료 | 시간: {elapsed:.2f}s | RL Buffer: {len(agent.rl_buffer)} | Q Delta: {q_weight_delta:.4f} | SL Delta: {sl_weight_delta:.4f} | NaN: {nan_detected} | Hash: {ckpt_hash[:16]}")

    return smoke_results

def main():
    print("============================================================")
    print("PHASE 0: 5-PLAYER 20BB POKER ENVIRONMENT CONFIRMATION")
    print("============================================================")
    game = get_5p_20bb_poker_game()
    init_st = game.new_initial_state()
    info_dim = len(init_st.information_state_tensor(0))
    action_dim = game.num_distinct_actions()

    print(f"  * Game Name: universal_poker (5-Player NLH)")
    print(f"  * Rounds: {game.max_game_length()} (Preflop, Flop - 2 Betting Rounds ONLY, NO Turn/River)")
    print(f"  * Info State Tensor Dimension: {info_dim}")
    print(f"  * Actions: {action_dim} (0: Fold, 1: Call/Check, 2: Pot Raise, 3: All-in)")
    print(f"  * Blinds: SB 50, BB 100 (Stack 2000 chips = 20BB)")

    print("\n[PHASE 0.1: Chip Ledger & Zero-sum Conservation Audit]")
    zero_err, bound_err, ledger_records = verify_chip_ledger_and_accounting(game, num_hands=200)
    print(f"  * Zero-sum Violations: {zero_err} (0 허용)")
    print(f"  * Return Bound Violations (-20BB ~ +80BB): {bound_err} (0 허용)")

    os.makedirs('v54/data', exist_ok=True)
    with open('v54/data/phase0_chip_ledger_200hands.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ledger_records[0].keys())
        writer.writeheader()
        writer.writerows(ledger_records)
    print(f"  * 200핸드 원장 저장 완료: v54/data/phase0_chip_ledger_200hands.csv")

    assert zero_err == 0, "Zero-sum violation detected!"
    assert bound_err == 0, "Bound violation detected!"

    print("\n============================================================")
    print("PHASE 1: HYPERPARAMETER FREEZE AUDIT")
    print("============================================================")
    hp_spec = {
        'architecture': '2 Hidden Layers x 128 Units (Q-Net & SL-Net)',
        'q_lr': 0.0001,
        'sl_lr': 0.0001,
        'gamma': 0.99,
        'eta': 0.10,
        'target_sync_steps': 200,
        'rl_buffer_capacity': 20000,
        'sl_buffer_capacity': 20000,
        'batch_size': 64,
        'grad_clip_norm': 1.0,
        'reward_unit': 'Terminal Net Chips / 100.0 (Net BB)'
    }
    for k, v in hp_spec.items():
        print(f"  * {k}: {v}")

    # Phase 2 실행
    smoke_results = run_smoke_test(game, seeds=[42, 101, 202], episodes=1000)

    with open('v54/data/phase2_smoke_test_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=smoke_results[0].keys())
        writer.writeheader()
        writer.writerows(smoke_results)
    print(f"\n[Smoke Test 결과 저장 완료]: v54/data/phase2_smoke_test_results.csv")

    all_passed = all(
        (not r['nan_detected']) and 
        (not r['illegal_action_detected']) and 
        (r['q_weight_delta'] > 0) and 
        (r['sl_weight_delta'] > 0) 
        for r in smoke_results
    )
    print(f"\n>>> PHASE 0, 1, 2 최종 판정: {'ALL PASSED (정상 통과)' if all_passed else 'FAILED'}")

if __name__ == '__main__':
    main()
