"""
POKER AI V10.24: DIRECTIVE 015 - EVALUATION ENGINE REPAIR & FINAL CHECKPOINT QUALIFICATION
==========================================================================================
Strict Implementation of ChatGPT DIRECTIVE 015 Protocol:
Phase 1: Evaluation Engine Repair & RNG Isolation Audit
Phase 2: 20,000-Hand Independent Unseen Benchmark League (Seeds 400,000 ~ 419,999)
Phase 3: Deep Statistical Qualification, Paired Difference CIs, and Head-to-Head Showdown
"""

import os
import sys
import json
import time
import hashlib
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1021.run_v1021_anti_limper_league import LooseLimperBot
from v1023.v1023_13_differential_forensics import PPOAgentWrapper, RealisticAggressiveBot

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_opponents_for_pool(pool_name, base_seed):
    """
    상대 봇 객체를 완전히 독립적으로 새로 생성하여 RNG 상태 오염/공유를 원천 차단.
    """
    if pool_name == "Standard":
        return [
            DeterministicBalancedBot(rng_seed=base_seed + 13),
            DeterministicTightPassiveBot(seed=base_seed + 17),
            DeterministicUltraAggressiveBot(seed=base_seed + 19),
            LooseLimperBot(seed=base_seed + 23)
        ], ["Balanced", "TightPassive", "UltraAggressive", "LooseLimper"]
    elif pool_name == "Unseen":
        return [
            RealisticAggressiveBot(seed=base_seed + 31),
            DeterministicBalancedBot(rng_seed=base_seed + 37),
            DeterministicTightPassiveBot(seed=base_seed + 41),
            DeterministicUltraAggressiveBot(seed=base_seed + 43)
        ], ["RealisticAggressive", "Balanced_Alt", "TightPassive_Alt", "UltraAggressive_Alt"]
    else:
        raise ValueError(f"Unknown pool name: {pool_name}")

def simulate_single_hand(game, ppo_agent, postflop_guard, seat, seed, pool_name):
    """
    단일 핸드 시뮬레이션:
    - rng_card: 덱 분배용 독립 난수
    - opp_seed_base: 상대 봇용 독립 시드
    - sim_seed: MC 시뮬레이션용 독립 시드
    """
    chance_rng = np.random.RandomState(seed)
    opp_seed_base = int(hashlib.sha256(f"opp_{seed}".encode()).hexdigest()[:8], 16)
    sim_seed = int(hashlib.sha256(f"sim_{seed}".encode()).hexdigest()[:8], 16)
    
    opponents, opp_names = get_opponents_for_pool(pool_name, opp_seed_base)
    
    st = game.new_initial_state()
    advance_chance_nodes(st, chance_rng)
    f_set = set()
    dec = 0
    actions_taken = []
    mc_interventions = 0
    preflop_actions = []
    postflop_actions = []
    
    while not st.is_terminal():
        if st.is_chance_node():
            advance_chance_nodes(st, chance_rng)
            continue
        cp = st.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st, cp)
        rnd = get_round_from_state(st)
        
        if cp == seat:
            if rnd == 0 or postflop_guard is None:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = ppo_agent.act(inf, l6)
            else:
                act = postflop_guard.act(st, seat, l6, f_set, sim_seed, dec)
                mc_interventions += 1
                
            actions_taken.append(act)
            if rnd == 0:
                preflop_actions.append(act)
            else:
                postflop_actions.append(act)
                
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opponents[opp_idx]
            act = bot.select_action(st, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            
        if act == 0:
            f_set.add(cp)
        st.apply_action(raw_act)
        
    returns_chips = st.returns()
    assert abs(sum(returns_chips)) < 1e-4, f"Chip conservation violated! Sum: {sum(returns_chips)}"
    
    ret_bb = returns_chips[seat] / 100.0  # 1BB = 100칩
    
    return {
        "ret_chips": float(returns_chips[seat]),
        "ret_bb": float(ret_bb),
        "actions": actions_taken,
        "preflop_actions": preflop_actions,
        "postflop_actions": postflop_actions,
        "mc_interventions": mc_interventions,
        "all_returns": [float(r) for r in returns_chips]
    }

def verify_reproducibility(game, ppo_c4, ppo_c10, guard):
    """
    Phase 1: 순서 불변성 및 난수 완전 격리 재현성 검증
    """
    print("\n" + "=" * 70, flush=True)
    print("[Phase 1] Auditing RNG Isolation & Evaluation Order Invariance...", flush=True)
    print("=" * 70, flush=True)
    
    test_seeds = [399990 + i for i in range(10)]
    
    # 순서 1: C4 -> C10+MC
    order1_c4 = []
    order1_c10 = []
    for s in test_seeds:
        res_c4 = simulate_single_hand(game, ppo_c4, None, s % 5, s, "Standard")
        res_c10 = simulate_single_hand(game, ppo_c10, guard, s % 5, s, "Standard")
        order1_c4.append(res_c4)
        order1_c10.append(res_c10)
        
    # 순서 2: C10+MC -> C4 (순서 역전)
    order2_c4 = []
    order2_c10 = []
    for s in test_seeds:
        res_c10 = simulate_single_hand(game, ppo_c10, guard, s % 5, s, "Standard")
        res_c4 = simulate_single_hand(game, ppo_c4, None, s % 5, s, "Standard")
        order2_c4.append(res_c4)
        order2_c10.append(res_c10)
        
    # 순서 3: C4 -> C10+MC -> C4_dup (동일 정책 중복 실행)
    order3_c4_dup = []
    for s in test_seeds:
        _ = simulate_single_hand(game, ppo_c4, None, s % 5, s, "Standard")
        _ = simulate_single_hand(game, ppo_c10, guard, s % 5, s, "Standard")
        res_c4_dup = simulate_single_hand(game, ppo_c4, None, s % 5, s, "Standard")
        order3_c4_dup.append(res_c4_dup)
        
    for i in range(len(test_seeds)):
        assert order1_c4[i]["ret_chips"] == order2_c4[i]["ret_chips"] == order3_c4_dup[i]["ret_chips"], \
            f"Seed {test_seeds[i]}: C4 return mismatch between orders!"
        assert order1_c4[i]["actions"] == order2_c4[i]["actions"] == order3_c4_dup[i]["actions"], \
            f"Seed {test_seeds[i]}: C4 action sequence mismatch!"
        assert order1_c10[i]["ret_chips"] == order2_c10[i]["ret_chips"], \
            f"Seed {test_seeds[i]}: C10 return mismatch between orders!"
        assert order1_c10[i]["actions"] == order2_c10[i]["actions"], \
            f"Seed {test_seeds[i]}: C10 action sequence mismatch!"
            
    print("  [+] Order Invariance Check: PASS (100% bit-exact across all permutation orders)", flush=True)
    print("  [+] Opponent Bot RNG Isolation Check: PASS (Zero inter-policy RNG leakage)", flush=True)
    print("  [+] Action Sequence Determinism Check: PASS (All actions exactly identical)", flush=True)
    print("=" * 70 + "\n", flush=True)
    return True

def main():
    print("=" * 75, flush=True)
    print("POKER AI V10.24: DIRECTIVE 015 FINAL CHECKPOINT QUALIFICATION", flush=True)
    print("Under ChatGPT Official Directive & Comprehensive Audit Standards", flush=True)
    print("=" * 75, flush=True)
    
    # 1. 모델 경로 및 해시 동결
    ckpts = {
        "Baseline": "v1022/models/StrictAntiLimper_R2_100K.pt",
        "Cycle_2_725k": "v1024/models/V1024_PPO_Cycle_2_725353hands.pt",
        "Cycle_4_1211k": "v1024/models/V1024_PPO_Cycle_4_1211670hands.pt",
        "Cycle_10_2289k": "v1024/models/V1024_PPO_Cycle_10_2289023hands.pt",
        "Cycle_13_2890k": "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt"
    }
    
    hashes = {}
    models = {}
    print("\n[*] 1. Freezing Checkpoints & Verifying SHA-256 Hashes:", flush=True)
    for name, path in ckpts.items():
        sha = compute_sha256(path)
        hashes[name] = {"path": path, "sha256": sha}
        print(f"  - [{name:15s}] {path} -> {sha[:16]}...", flush=True)
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        m.eval()
        models[name] = m

    # 2. 게임 환경 및 어댑티브 가드 준비
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    adaptive_guard = ModelA(bp_agent, sims=300, adaptive=True, adaptive_sims=3000, margin_threshold=0.025)
    
    # 5개 정책 구성
    policies = {
        "Cycle_4_PurePPO": (PPOAgentWrapper(models["Cycle_4_1211k"]), None, "Primary Contender"),
        "Cycle_10_AdaptiveMC": (PPOAgentWrapper(models["Cycle_10_2289k"]), adaptive_guard, "Primary Contender"),
        "Cycle_2_AdaptiveMC": (PPOAgentWrapper(models["Cycle_2_725k"]), adaptive_guard, "Secondary Candidate"),
        "Baseline_PurePPO": (PPOAgentWrapper(models["Baseline"]), None, "Pre-trained V10.22 Baseline Policy"),
        "Cycle_13_PurePPO": (PPOAgentWrapper(models["Cycle_13_2890k"]), None, "Forensic Subject")
    }

    # Phase 1: 재현성 및 난수 격리 감사
    repro_passed = verify_reproducibility(
        game, 
        policies["Cycle_4_PurePPO"][0], 
        policies["Cycle_10_AdaptiveMC"][0], 
        adaptive_guard
    )

    # Phase 2: 20,000핸드 미공개 독립 평가
    # 400,000 ~ 409,999: Standard Opponent Pool (10,000핸드)
    # 410,000 ~ 419,999: Unseen Opponent Pool (10,000핸드)
    total_hands = 20000
    seeds = [400000 + i for i in range(total_hands)]
    print(f"[*] 2. Starting 20,000 Independent Benchmark Hands (Seeds {seeds[0]} ~ {seeds[-1]})...", flush=True)
    print(f"  - Standard Pool: Seeds 400,000 ~ 409,999 (10,000 hands)", flush=True)
    print(f"  - Unseen Pool  : Seeds 410,000 ~ 419,999 (10,000 hands)", flush=True)
    print(f"  - Seat Rotation: Complete 5-seat balance (4,000 hands per seat)\n", flush=True)
    
    raw_log_path = "v1024/data/directive015_raw_hands.jsonl"
    os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)
    raw_log_file = open(raw_log_path, "w", encoding="utf-8")
    
    # 데이터 수집용 컨테이너
    hand_rets = {k: [] for k in policies}
    pool_rets = {k: {"Standard": [], "Unseen": []} for k in policies}
    seat_rets = {k: defaultdict(list) for k in policies}
    act_counts_pre = {k: defaultdict(int) for k in policies}
    act_counts_post = {k: defaultdict(int) for k in policies}
    mc_counts = {k: 0 for k in policies}
    
    start_time = time.time()
    
    for idx, seed in enumerate(seeds):
        seat = idx % 5
        pool_name = "Standard" if seed < 410000 else "Unseen"
        
        hand_record = {
            "hand_idx": idx,
            "seed": seed,
            "seat": seat,
            "pool": pool_name,
            "policies": {}
        }
        
        for p_name, (agent, guard, role) in policies.items():
            res = simulate_single_hand(game, agent, guard, seat, seed, pool_name)
            
            hand_rets[p_name].append(res["ret_bb"])
            pool_rets[p_name][pool_name].append(res["ret_bb"])
            seat_rets[p_name][seat].append(res["ret_bb"])
            mc_counts[p_name] += res["mc_interventions"]
            
            for a in res["preflop_actions"]:
                act_counts_pre[p_name][a] += 1
            for a in res["postflop_actions"]:
                act_counts_post[p_name][a] += 1
                
            hand_record["policies"][p_name] = {
                "ret_chips": res["ret_chips"],
                "ret_bb": res["ret_bb"],
                "mc_interventions": res["mc_interventions"]
            }
            
        raw_log_file.write(json.dumps(hand_record) + "\n")
        
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            rem = (total_hands - (idx + 1)) / rate
            print(f"  [Progress] {idx + 1:5d} / {total_hands} hands ({elapsed:.1f}s, {rate:.1f} hands/s, ETA: {rem/60:.1f} min)", flush=True)

    raw_log_file.close()
    print(f"\n[*] Raw hand logs saved to {raw_log_path}", flush=True)

    # Phase 3: 심층 통계 분석 및 최종 후보 대조 검정
    print("\n[*] 3. Computing Deep Statistical Metrics and Paired Tests...", flush=True)
    
    base_arr = np.array(hand_rets["Baseline_PurePPO"]) * 100.0  # bb/100
    c4_arr = np.array(hand_rets["Cycle_4_PurePPO"]) * 100.0
    c10_arr = np.array(hand_rets["Cycle_10_AdaptiveMC"]) * 100.0
    
    summary_overall = {}
    for p_name in policies:
        arr = np.array(hand_rets[p_name]) * 100.0
        diff_base = arr - base_arr
        sem = float(np.std(arr) / np.sqrt(len(arr)))
        diff_sem = float(np.std(diff_base) / np.sqrt(len(diff_base)))
        t_stat, p_val = stats.ttest_rel(arr, base_arr) if p_name != "Baseline_PurePPO" else (0.0, 1.0)
        
        # 행동 분포
        tot_pre = sum(act_counts_pre[p_name].values())
        tot_post = sum(act_counts_post[p_name].values())
        
        pre_fold = act_counts_pre[p_name][0] / tot_pre * 100.0 if tot_pre > 0 else 0.0
        pre_call = act_counts_pre[p_name][1] / tot_pre * 100.0 if tot_pre > 0 else 0.0
        pre_raise = sum(act_counts_pre[p_name][a] for a in range(2, 6)) / tot_pre * 100.0 if tot_pre > 0 else 0.0
        
        post_fold = act_counts_post[p_name][0] / tot_post * 100.0 if tot_post > 0 else 0.0
        post_call = act_counts_post[p_name][1] / tot_post * 100.0 if tot_post > 0 else 0.0
        post_raise = sum(act_counts_post[p_name][a] for a in range(2, 6)) / tot_post * 100.0 if tot_post > 0 else 0.0
        
        summary_overall[p_name] = {
            "mean_bb100": float(np.mean(arr)),
            "std_bb100": float(np.std(arr)),
            "sem": sem,
            "ci_95": [float(np.mean(arr) - 1.96 * sem), float(np.mean(arr) + 1.96 * sem)],
            "positive_return_rate_pct": float(np.mean(arr > 0) * 100.0),
            "diff_vs_baseline_mean": float(np.mean(diff_base)),
            "diff_vs_baseline_sem": diff_sem,
            "diff_vs_baseline_ci95": [float(np.mean(diff_base) - 1.96 * diff_sem), float(np.mean(diff_base) + 1.96 * diff_sem)],
            "t_stat_vs_baseline": float(t_stat),
            "p_val_vs_baseline": float(p_val) if not np.isnan(p_val) else 1.0,
            "max_loss_single_hand_bb": float(np.min(arr) / 100.0),
            "var_95_bb": float(np.percentile(arr / 100.0, 5)),
            "quartiles_bb100": {
                "q25": float(np.percentile(arr, 25)),
                "q50": float(np.percentile(arr, 50)),
                "q75": float(np.percentile(arr, 75))
            },
            "preflop_action_pct": {"fold": pre_fold, "call": pre_call, "raise": pre_raise},
            "postflop_action_pct": {"fold": post_fold, "call": post_call, "raise": post_raise},
            "mc_interventions_total": mc_counts[p_name]
        }

    # Head-to-Head 직접 맞대결: Cycle 4 Pure PPO vs Cycle 10 + Adaptive MC
    diff_c10_c4 = c10_arr - c4_arr
    diff_c10_c4_sem = float(np.std(diff_c10_c4) / np.sqrt(len(diff_c10_c4)))
    t_c10_c4, p_c10_c4 = stats.ttest_rel(c10_arr, c4_arr)
    
    head_to_head = {
        "description": "Cycle_10_AdaptiveMC minus Cycle_4_PurePPO",
        "diff_mean_bb100": float(np.mean(diff_c10_c4)),
        "diff_sem": diff_c10_c4_sem,
        "ci_95": [float(np.mean(diff_c10_c4) - 1.96 * diff_c10_c4_sem), float(np.mean(diff_c10_c4) + 1.96 * diff_c10_c4_sem)],
        "t_stat": float(t_c10_c4),
        "p_value": float(p_c10_c4),
        "c10_win_frequency_pct": float(np.mean(c10_arr > c4_arr) * 100.0),
        "tie_frequency_pct": float(np.mean(c10_arr == c4_arr) * 100.0),
        "c4_win_frequency_pct": float(np.mean(c4_arr > c10_arr) * 100.0)
    }

    # Pool별 요약 (Standard vs Unseen)
    pool_summary = {}
    for pool in ["Standard", "Unseen"]:
        pool_summary[pool] = {}
        for p_name in policies:
            p_arr = np.array(pool_rets[p_name][pool]) * 100.0
            p_sem = float(np.std(p_arr) / np.sqrt(len(p_arr)))
            pool_summary[pool][p_name] = {
                "mean_bb100": float(np.mean(p_arr)),
                "sem": p_sem,
                "ci_95": [float(np.mean(p_arr) - 1.96 * p_sem), float(np.mean(p_arr) + 1.96 * p_sem)],
                "positive_return_rate_pct": float(np.mean(p_arr > 0) * 100.0)
            }

    # Seat별 요약
    seat_summary = {}
    for seat in range(5):
        seat_summary[f"Seat_{seat}"] = {}
        for p_name in policies:
            s_arr = np.array(seat_rets[p_name][seat]) * 100.0
            seat_summary[f"Seat_{seat}"][p_name] = {
                "mean_bb100": float(np.mean(s_arr)),
                "sem": float(np.std(s_arr) / np.sqrt(len(s_arr)))
            }

    final_report_data = {
        "meta": {
            "title": "POKER AI V10.24 DIRECTIVE 015 Final Checkpoint Qualification",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_hands": total_hands,
            "seed_range": [seeds[0], seeds[-1]],
            "baseline_identity": {
                "name": "StrictAntiLimper_R2_100K.pt",
                "origin": "Pre-trained V10.22 Anti-Limper Baseline Policy (100,000 hands trained in V10.22)",
                "note": "Confirmed pre-trained policy baseline, NOT random untrained weights"
            },
            "sha256_hashes": hashes,
            "reproducibility_audit": {
                "status": "PASS",
                "order_invariance": True,
                "bit_exact": True,
                "details": "Verified 100% bit-exact across all permutation orders and duplicate runs"
            }
        },
        "overall_summary": summary_overall,
        "head_to_head_c10_vs_c4": head_to_head,
        "pool_summary": pool_summary,
        "seat_summary": seat_summary
    }

    result_json_path = "v1024/data/v1024_directive015_results.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Qualification Complete! Result JSON written to {result_json_path}", flush=True)
    
    # 간이 콘솔 요약 출력
    print("\n" + "=" * 90)
    print(f"{'Policy':<25} | {'bb/100':<10} | {'95% CI':<24} | {'vs Base (bb)':<14} | {'p-val':<8} | {'PosRet%':<8}")
    print("-" * 90)
    for p_name, s in summary_overall.items():
        ci_str = f"[{s['ci_95'][0]:.1f}, {s['ci_95'][1]:.1f}]"
        diff_str = f"{s['diff_vs_baseline_mean']:+.1f}" if p_name != "Baseline_PurePPO" else "0.0"
        p_str = f"{s['p_val_vs_baseline']:.4f}" if p_name != "Baseline_PurePPO" else "1.0000"
        print(f"{p_name:<25} | {s['mean_bb100']:+9.2f} | {ci_str:<24} | {diff_str:<14} | {p_str:<8} | {s['positive_return_rate_pct']:5.1f}%")
    print("=" * 90)
    
    print("\n[Head-to-Head: Cycle 10 + Adaptive MC vs Cycle 4 Pure PPO]")
    print(f"  - Paired Diff Mean: {head_to_head['diff_mean_bb100']:+.2f} bb/100 (SEM: {head_to_head['diff_sem']:.2f})")
    print(f"  - 95% Paired CI   : [{head_to_head['ci_95'][0]:.2f}, {head_to_head['ci_95'][1]:.2f}]")
    print(f"  - Paired t-test   : t = {head_to_head['t_stat']:.3f}, p = {head_to_head['p_value']:.4f}")
    print(f"  - Hand Win Rates  : C10+MC Wins {head_to_head['c10_win_frequency_pct']:.1f}%, Ties {head_to_head['tie_frequency_pct']:.1f}%, C4 Wins {head_to_head['c4_win_frequency_pct']:.1f}%\n")

if __name__ == "__main__":
    main()
