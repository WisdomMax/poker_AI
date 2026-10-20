"""
POKER AI V4.4 - PHASE 2: OpenSpiel Leduc Poker CFR & CFR+ 기준 모델 고도화
- 100, 250, 500, 1,000, 2,000 Iterations 단계별 수렴도 측정
- Standard CFR vs CFR+ 비교
- NashConv, Exploitability, 학습 소요 시간, 메모리 사용량 측정
- 가장 정밀한 공식 기준 모델을 저장하여 이후 NFSP 및 Bayesian 평가 벤치마크로 확립
"""

import sys
import os
import time
import pickle
import psutil
from typing import List, Dict, Tuple
import pyspiel
from open_spiel.python.algorithms import exploitability, cfr

def benchmark_cfr_algorithm(game, solver_class, name: str, iterations_list: List[int]) -> List[Dict]:
    process = psutil.Process()
    print(f"\n▶ [{name} 벤치마크 실행 중 (목표: {max(iterations_list)} iterations)]...")
    solver = solver_class(game)
    records = []
    t0 = time.time()
    
    current_iter = 0
    for target_it in iterations_list:
        needed = target_it - current_iter
        for _ in range(needed):
            solver.evaluate_and_update_policy()
        current_iter = target_it
        
        elapsed = time.time() - t0
        avg_policy = solver.average_policy()
        nc = float(exploitability.nash_conv(game, avg_policy))
        exp = nc / 2.0
        mem = process.memory_info().rss / (1024 * 1024)
        
        records.append({
            "name": name,
            "iteration": target_it,
            "nash_conv": nc,
            "exploitability": exp,
            "time": elapsed,
            "memory_mb": mem
        })
        print(f"  * {name} Iteration {target_it:>4}: NashConv = {nc:.6f} (Exp: {exp*100:>5.2f} bb/100) | 소요: {elapsed:>5.1f}s | 메모리: {mem:.1f} MB")
        
    return records, solver

def run_phase2_cfr():
    print("=" * 80)
    print("🔬 [V4.4 PHASE 2] Leduc Poker CFR 및 CFR+ 공식 기준 모델 고도화")
    print("=" * 80)
    
    game = pyspiel.load_game("leduc_poker")
    iters = [100, 250, 500, 1000, 2000]
    
    # 1. Standard CFR
    cfr_recs, cfr_solver = benchmark_cfr_algorithm(game, cfr.CFRSolver, "Standard CFR", iters)
    
    # 2. CFR+ (CFRPlus)
    cfr_plus_recs, plus_solver = benchmark_cfr_algorithm(game, cfr.CFRPlusSolver, "CFR+", iters)
    
    # 3. 종합 비교표 출력
    print("\n" + "=" * 80)
    print("📊 [Leduc Poker CFR vs CFR+ 반복 횟수별 종합 비교표]")
    print("=" * 80)
    print(f"{'Iterations':<12} | {'Standard CFR NashConv':<22} | {'CFR+ NashConv':<22} | {'CFR+ 개선폭':<12}")
    print("-" * 80)
    
    for r_c, r_p in zip(cfr_recs, cfr_plus_recs):
        it = r_c["iteration"]
        nc_c = r_c["nash_conv"]
        nc_p = r_p["nash_conv"]
        diff = nc_c - nc_p
        print(f"{it:<12} | {nc_c:<22.6f} | {nc_p:<22.6f} | {diff:>+10.6f}")
        
    print("=" * 80)
    
    # 최적 모델 저장
    best_plus_nc = cfr_plus_recs[-1]["nash_conv"]
    print(f"🎯 최종 확정된 공식 기준 모델: CFR+ 2,000 Iterations (NashConv: {best_plus_nc:.6f}, Exploit: {best_plus_nc*50:.2f} bb/100)")
    
    # Policy 직렬화 저장
    models_dir = "v44/models"
    os.makedirs(models_dir, exist_ok=True)
    # OpenSpiel tabular policy 딕셔너리 추출
    best_policy = plus_solver.average_policy()
    policy_dict = {}
    
    # 936개 정보집합에 대한 정책 추출
    def extract_tabular(st):
        if st.is_terminal(): return
        if st.is_chance_node():
            for a, _ in st.chance_outcomes():
                nxt = st.clone(); nxt.apply_action(a); extract_tabular(nxt)
            return
        cp = st.current_player()
        info_str = st.information_state_string(cp)
        if info_str not in policy_dict:
            probs = best_policy.action_probabilities(st)
            policy_dict[info_str] = probs
        for a in st.legal_actions(cp):
            nxt = st.clone(); nxt.apply_action(a); extract_tabular(nxt)
            
    extract_tabular(game.new_initial_state())
    
    with open(f"{models_dir}/leduc_cfr_plus_2000.pkl", "wb") as f:
        pickle.dump(policy_dict, f)
        
    print(f"  💾 총 {len(policy_dict)}개 정보집합 정책 추출 완료 -> {models_dir}/leduc_cfr_plus_2000.pkl 저장 완료!")

if __name__ == "__main__":
    run_phase2_cfr()
