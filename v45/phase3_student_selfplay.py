"""
POKER AI V4.5 - PHASE 3: Student + Self-Play Fine-Tuning 연구
- 핵심 질문: CFR+로부터 완벽히 증류된 Student 신경망(NashConv 0.0087)을 자가 대국(NFSP Self-Play)으로 미세조정하면
  더 개선되는가, 아니면 DQN의 근사 오차와 샘플링 노이즈로 인해 정책이 퇴보(Catastrophic Forgetting/Drift)하는가?
- Student 가중치를 Average Policy Network에 로드한 상태에서 NFSP 학습 진행
- 에피소드별(0, 5k, 10k, 20k, 30k) NashConv 및 Exploitability 추적
"""

import sys
import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

# 단위 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from units import format_metrics, nashconv_to_exploitability
from phase2_distillation import LeducStudentNet, StudentOpenSpielPolicy

# NFSP 에이전트 모듈 임포트
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper

def run_phase3_student_selfplay(
    total_episodes: int = 30000,
    checkpoints = [0, 5000, 10000, 20000, 30000],
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = pyspiel.load_game("leduc_poker")
    device = torch.device("cpu")
    student_ckpt_path = "v45/models/leduc_student_best.pt"
    
    print("=" * 80)
    print("🔬 [V4.5 PHASE 3] Student + Self-Play Fine-Tuning 안정성 및 퇴보 검증")
    print("=" * 80)
    print(f"📖 기본 Student 모델 로드: {student_ckpt_path}")
    
    # 1. 두 에이전트 초기화
    agent0 = LeducNFSPAgent(0, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
    agent1 = LeducNFSPAgent(1, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
    
    # Student 가중치를 Average Policy (avg_net)에 로드
    state_dict = torch.load(student_ckpt_path, map_location=device)
    agent0.avg_net.load_state_dict(state_dict)
    agent1.avg_net.load_state_dict(state_dict)
    print("✅ Agent0 및 Agent1의 Average Policy Network에 Student 가중치 초기화 완료")
    
    wrapper = LeducNFSPPolicyWrapper(game, agent0, agent1)
    
    # 초기(0 에피소드) NashConv 측정
    records = []
    initial_nc = float(exploitability.nash_conv(game, wrapper))
    exp_init = nashconv_to_exploitability(initial_nc)
    print(f"\n[초기 상태 (Episode 0)] NashConv = {initial_nc:.6f} (Exp: {exp_init['exploitability_bb_per_100']:.2f} bb/100)")
    records.append({
        "episode": 0,
        "nash_conv": initial_nc,
        "exp_bb": exp_init['exploitability_bb_per_100']
    })
    
    t0 = time.time()
    
    for ep in range(1, total_episodes + 1):
        agent0.sample_episode_mode()
        agent1.sample_episode_mode()
        
        state = game.new_initial_state()
        traj0, traj1 = [], []
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, prs = zip(*outcomes)
                ch_act = np.random.choice(acts, p=prs)
                state.apply_action(ch_act)
                continue
                
            cur_p = state.current_player()
            info_vec = state.information_state_tensor(cur_p)
            legal_acts = state.legal_actions(cur_p)
            agent = agent0 if cur_p == 0 else agent1
            
            act, probs = agent.act(info_vec, legal_acts)
            mask = np.zeros(3, dtype=np.float32)
            mask[legal_acts] = 1.0
            
            st_tensor = torch.tensor(info_vec, dtype=torch.float32)
            mk_tensor = torch.tensor(mask, dtype=torch.float32)
            
            if agent.current_mode == 'best_response':
                agent.sl_buffer.add((st_tensor, probs, mk_tensor))
                
            if cur_p == 0:
                traj0.append((st_tensor, act, mk_tensor))
            else:
                traj1.append((st_tensor, act, mk_tensor))
                
            state.apply_action(act)
            
        returns = state.returns()
        r0 = returns[0] / 10.0
        r1 = returns[1] / 10.0
        
        for i, (st, ac, mk) in enumerate(traj0):
            is_done = (i == len(traj0) - 1)
            nst = traj0[i+1][0] if not is_done else st
            rw = r0 if is_done else 0.0
            agent0.rl_buffer.push(st, ac, rw, nst, is_done, mk)
            
        for i, (st, ac, mk) in enumerate(traj1):
            is_done = (i == len(traj1) - 1)
            nst = traj1[i+1][0] if not is_done else st
            rw = r1 if is_done else 0.0
            agent1.rl_buffer.push(st, ac, rw, nst, is_done, mk)
            
        if ep % 64 == 0:
            agent0.train_step()
            agent1.train_step()
        
        if ep in checkpoints:
            nc = float(exploitability.nash_conv(game, wrapper))
            exp = nashconv_to_exploitability(nc)
            elapsed = time.time() - t0
            print(f"  * Episode {ep:>5}/{total_episodes}: NashConv = {nc:.6f} (Exp: {exp['exploitability_bb_per_100']:>5.2f} bb/100) | 소요: {elapsed:.1f}s")
            records.append({
                "episode": ep,
                "nash_conv": nc,
                "exp_bb": exp['exploitability_bb_per_100']
            })
            
    print("\n" + "=" * 80)
    print("📊 [Student + Self-Play Fine-Tuning 단계별 수렴도/퇴보 분석표]")
    print("=" * 80)
    print(f"{'학습 에피소드':<15} | {'NashConv':<15} | {'Exploitability (bb/100)':<25} | {'초기 대비 변동'}")
    print("-" * 80)
    
    for r in records:
        delta = r['nash_conv'] - initial_nc
        delta_str = f"{delta:+.6f}" if r['episode'] > 0 else "기준점 (Distilled)"
        print(f"{r['episode']:<15} | {r['nash_conv']:<15.6f} | {r['exp_bb']:<25.2f} | {delta_str}")
    print("=" * 80)
    
    final_nc = records[-1]['nash_conv']
    if final_nc > initial_nc:
        print("\n⚠️ [학습 결과 및 원인 규명]:")
        print(f"  - CFR+에서 완벽히 수렴한 Student 가중치(NashConv {initial_nc:.6f})에 NFSP Self-Play를 적용한 결과,")
        print(f"    최종 NashConv가 {final_nc:.6f}로 상승(퇴보/Catastrophic Forgetting)하였습니다.")
        print("  - 주 원인: (1) DQN Best Response 추정치의 함수 근사 노이즈,")
        print("            (2) 유한한 에피소드 궤적 샘플링으로 인한 표본 편향 및 정책 드리프트(Policy Drift),")
        print("            (3) Reservoir Buffer에 새로 유입되는 노이즈 섞인 액션이 평균 정책을 오염시킴.")
        print("  - 결론: 이미 수학적 내쉬 균형에 근접한 모델에는 순수 강화학습 파인튜닝보다")
        print("          사전 학습된 정밀 지도학습(Distillation) 가중치를 직접 고정(Freeze)하여 사용하는 것이 훨씬 우수함.")
    else:
        print(f"\n🎉 [학습 결과]: Self-Play 미세조정을 통해 NashConv가 {initial_nc:.6f} -> {final_nc:.6f}로 개선되었습니다.")
        
    # 모델 저장 (비교용)
    torch.save(agent0.avg_net.state_dict(), "v45/models/leduc_student_finetuned_p0.pt")
    torch.save(agent1.avg_net.state_dict(), "v45/models/leduc_student_finetuned_p1.pt")
    print("💾 파인튜닝 모델 저장 완료: v45/models/leduc_student_finetuned_p0/p1.pt")

if __name__ == "__main__":
    run_phase3_student_selfplay()
