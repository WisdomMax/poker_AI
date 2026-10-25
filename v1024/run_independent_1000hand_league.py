"""
POKER AI V10.24: INDEPENDENT 1,000-HAND HOLDOUT BENCHMARK LEAGUE
================================================================
ChatGPT 공식 검증 지시사항을 100% 반영한 엄격 독립 교차 평가 엔진:
1. 5대 마일스톤 체크포인트(Cycle 2, 4, 7, 10, 13) + Baseline 동결 및 SHA-256 해시 기록
2. 완전 미공개 독립 시드 (Seeds 300,000 ~ 300,999, 1,000핸드)
3. 5인 좌석 순환 (Seat Rotation 0~4 각 200핸드씩 균등 배분)
4. 상대 봇 4종(Balanced, TightPassive, UltraAggressive, LooseLimper) 고정 출전
5. 난수 스트림 분리 (카드 분배, 상대 행동, 쇼다운 시뮬레이션 격리)
6. 12개 정책(6개 모델 x [단독, Adaptive MC]) 정밀 대조 평가
"""

import os
import sys
import json
import time
import subprocess
import hashlib
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA, estimate_equity_det,
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

def main():
    print("=" * 75, flush=True)
    print("POKER AI V10.24: INDEPENDENT 1,000-HAND BENCHMARK LEAGUE START", flush=True)
    print("Under Strict ChatGPT Directive & Seat Rotation Protocol", flush=True)
    print("=" * 75, flush=True)
    
    # 1. 체크포인트 정의 및 SHA-256 해시 동결
    ckpts = {
        "Baseline": "v1022/models/StrictAntiLimper_R2_100K.pt",
        "Cycle_2_725k": "v1024/models/V1024_PPO_Cycle_2_725353hands.pt",
        "Cycle_4_1211k": "v1024/models/V1024_PPO_Cycle_4_1211670hands.pt",
        "Cycle_7_1771k": "v1024/models/V1024_PPO_Cycle_7_1771164hands.pt",
        "Cycle_10_2289k": "v1024/models/V1024_PPO_Cycle_10_2289023hands.pt",
        "Cycle_13_2890k": "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt"
    }
    
    hashes = {}
    models = {}
    print("\n[*] 1. Freezing Checkpoints & Computing SHA-256 Hashes:", flush=True)
    for name, path in ckpts.items():
        sha = compute_sha256(path)
        hashes[name] = {"path": path, "sha256": sha}
        print(f"  - [{name:15s}] {path} -> {sha[:16]}...", flush=True)
        
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        m.eval()
        models[name] = m

    # 2. 게임 환경 및 상대 봇 준비
    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    adaptive_guard = ModelA(bp_agent, sims=300, adaptive=True, adaptive_sims=3000, margin_threshold=0.025)
    
    # 12개 정책 구성 (6개 모델 x [PPO 단독, Adaptive MC])
    policies = {}
    for name, m in models.items():
        policies[f"{name}_PurePPO"] = (PPOAgentWrapper(m), None)
        policies[f"{name}_AdaptiveMC"] = (PPOAgentWrapper(m), adaptive_guard)
        
    # 3. 미공개 독립 시드 1,000개 (300,000 ~ 300,999)
    league_seeds = [300000 + i for i in range(1000)]
    print(f"\n[*] 2. Initializing 1,000 Independent Unseen Seeds: {league_seeds[0]} ~ {league_seeds[-1]}", flush=True)
    print(f"[*] 3. Evaluating 12 Policies with 5-Player Complete Seat Rotation...", flush=True)
    
    results = {k: [] for k in policies}
    opp_results = {k: defaultdict(list) for k in policies}
    action_counts = {k: defaultdict(int) for k in policies}
    adaptive_interventions = {k: 0 for k in policies}
    
    start_eval_time = time.time()
    
    for idx, seed in enumerate(league_seeds):
        # 5인 좌석 순환 (Seat 0~4 균등 배분)
        seat = idx % 5
        
        # 난수 분리
        rng_card = np.random.RandomState(seed)
        opp_seed_base = (seed * 1000 + 47) & 0xFFFFFF
        
        opponents = [
            DeterministicBalancedBot(rng_seed=opp_seed_base + 13),
            DeterministicTightPassiveBot(seed=opp_seed_base + 17),
            DeterministicUltraAggressiveBot(seed=opp_seed_base + 19),
            LooseLimperBot(seed=opp_seed_base + 23)
        ]
        opp_names = ["Balanced", "TightPassive", "UltraAggressive", "LooseLimper"]
        
        for p_name, (ppo_agent, postflop_guard) in policies.items():
            # 카드 분배용 동일 복제 난수
            chance_rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            advance_chance_nodes(st, chance_rng)
            f_set = set()
            dec = 0
            
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
                        act = postflop_guard.act(st, seat, l6, f_set, seed, dec)
                        adaptive_interventions[p_name] += 1
                        
                    action_counts[p_name][act] += 1
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
                
            # 칩 보존 법칙 및 회계 검증: 총 칩 합계는 정확히 100,000칩 (초기 20,000 * 5)
            returns_chips = st.returns()
            assert abs(sum(returns_chips)) < 1e-4, f"Chip conservation violated! Sum: {sum(returns_chips)}"
            
            ret_bb = returns_chips[seat] / 100.0  # 1BB = 100칩
            results[p_name].append(ret_bb)
            
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_eval_time
            print(f"  [Progress] Evaluated {idx + 1} / 1000 hands ({elapsed:.1f}s, {(idx+1)/elapsed:.1f} hands/s)", flush=True)

    # 4. 통계 지표 및 기준선 대조 Paired t-test 분석
    print("\n[*] 4. Computing Full Statistical Metrics & Paired Tests...", flush=True)
    base_arr = np.array(results["Baseline_PurePPO"]) * 100.0
    
    summary = {}
    for p_name, rets in results.items():
        arr = np.array(rets) * 100.0  # bb/100 단위
        diff = arr - base_arr
        sem = float(np.std(arr) / np.sqrt(len(arr)))
        t_stat, p_val = stats.ttest_rel(arr, base_arr) if p_name != "Baseline_PurePPO" else (0.0, 1.0)
        
        tot_acts = sum(action_counts[p_name].values())
        fold_pct = action_counts[p_name][0] / tot_acts * 100.0 if tot_acts > 0 else 0.0
        call_pct = action_counts[p_name][1] / tot_acts * 100.0 if tot_acts > 0 else 0.0
        raise_pct = sum([action_counts[p_name][k] for k in range(2, 6)]) / tot_acts * 100.0 if tot_acts > 0 else 0.0
        
        summary[p_name] = {
            "mean_bb100": float(np.mean(arr)),
            "std_bb100": float(np.std(arr)),
            "sem": sem,
            "ci_95": [float(np.mean(arr) - 1.96 * sem), float(np.mean(arr) + 1.96 * sem)],
            "improvement_vs_base": float(np.mean(diff)),
            "t_stat": float(t_stat),
            "p_value_vs_base": float(p_val) if not np.isnan(p_val) else 1.0,
            "win_rate_pct": float(np.mean(arr > 0) * 100.0),
            "actions_pct": {"fold": fold_pct, "call": call_pct, "raise": raise_pct},
            "adaptive_interventions": adaptive_interventions[p_name]
        }
        
    out_data = {
        "meta": {
            "title": "POKER AI V10.24 Independent 1,000-Hand Holdout Benchmark League",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed_range": [league_seeds[0], league_seeds[-1]],
            "total_hands": 1000,
            "seat_rotation": "0~4 equal 200 hands each",
            "chip_conservation_verified": True,
            "model_hashes": hashes
        },
        "summary": summary
    }
    
    out_json_path = "v1024/data/v1024_independent_1000hand_league.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n[*] Full League Data Saved to {out_json_path}", flush=True)
    
    # 5. 콘솔 리포트 테이블 출력
    print("\n" + "=" * 100, flush=True)
    print(f"{'Policy Name':<25} | {'Mean bb/100':>12} | {'SEM':>8} | {'95% CI':>22} | {'vs Base':>10} | {'p-value':>8} | {'WinRate':>7}", flush=True)
    print("-" * 100, flush=True)
    for p_name, s in summary.items():
        ci_str = f"[{s['ci_95'][0]:+7.1f}, {s['ci_95'][1]:+7.1f}]"
        print(f"{p_name:<25} | {s['mean_bb100']:+12.2f} | {s['sem']:8.2f} | {ci_str:>22} | {s['improvement_vs_base']:+10.2f} | {s['p_value_vs_base']:8.4f} | {s['win_rate_pct']:6.1f}%", flush=True)
    print("=" * 100, flush=True)

if __name__ == "__main__":
    main()
