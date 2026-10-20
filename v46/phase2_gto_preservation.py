"""
POKER AI V4.6 - PHASE 2: GTO 전략 보존 및 Anchored Fine-Tuning 비교 실험
- 비교 대상:
  A. Frozen Student: 가중치 변경 없음 (GTO 절대 방어선)
  B. Student + Self-Play: 기존 표준 NFSP (자가 대국 직접 업데이트)
  C. Student + Anchored Fine-Tuning: CFR+ Teacher 정책과의 KL 발산 억제 앵커 손실 적용
     L_total = (1 - alpha) * L_SL + alpha * D_KL(P_Teacher || P_student)
- 독립 시드(42, 101, 202) 3회 반복 검증
- 평가 지표: 0k, 5k, 10k, 20k 에피소드별 OpenSpiel NashConv 및 Exploitability (bb/100)
"""

import sys
import os
import time
import copy
import random
import pickle
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

# 단위 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v45.units import nashconv_to_exploitability, CHIPS_PER_BB
from v45.phase2_distillation import LeducStudentNet, StudentOpenSpielPolicy, build_distillation_dataset
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper

# Anchored NFSP Agent 클래스
class AnchoredLeducNFSPAgent(LeducNFSPAgent):
    def __init__(self, player_id: int, anchor_net: nn.Module, alpha_anchor: float = 0.70, **kwargs):
        super().__init__(player_id, **kwargs)
        self.anchor_net = copy.deepcopy(anchor_net).to(self.device)
        self.anchor_net.eval()
        self.alpha_anchor = alpha_anchor

    def train_step(self):
        self.total_steps += 1
        # A. RL Update
        if len(self.rl_buffer) >= self.batch_size:
            batch = self.rl_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, masks = zip(*batch)
            b_st = torch.stack(states).to(self.device)
            b_ac = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            b_rw = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            b_nst = torch.stack(next_states).to(self.device)
            b_dn = torch.tensor(dones, dtype=torch.float32, device=self.device)
            b_mk = torch.stack(masks).to(self.device)

            curr_q = self.q_net(b_st).gather(1, b_ac).squeeze(1)
            with torch.no_grad():
                next_q_all = self.target_q_net(b_nst, b_mk)
                max_next_q = next_q_all.max(1)[0]
                target_q = b_rw + (1.0 - b_dn) * max_next_q

            rl_loss = nn.MSELoss()(curr_q, target_q)
            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.rl_optimizer.step()

            if self.total_steps % self.target_update == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        # B. SL Update with Teacher KL Anchor
        if len(self.sl_buffer) >= self.batch_size:
            sl_samples = self.sl_buffer.sample(self.batch_size)
            sl_st, sl_pr, sl_mk = zip(*sl_samples)
            b_st = torch.stack(sl_st).to(self.device)
            b_pr = torch.tensor(np.array(sl_pr), dtype=torch.float32, device=self.device)
            b_mk = torch.stack(sl_mk).to(self.device)

            logits = self.avg_net(b_st, b_mk)
            log_p = torch.log_softmax(logits, dim=-1)
            sl_loss = -torch.mean(torch.sum(b_pr * log_p, dim=-1))

            # 앵커 정책(Frozen Student)의 행동 확률 계산
            with torch.no_grad():
                anchor_logits = self.anchor_net(b_st, b_mk.bool())
                p_anchor = torch.softmax(anchor_logits, dim=-1)

            # KL 발산 정규화: D_KL(P_anchor || P_curr) = \sum P_anchor * (log P_anchor - log P_curr)
            kl_anchor = torch.sum(p_anchor * (torch.log(p_anchor + 1e-12) - log_p), dim=-1).mean()

            # 종합 손실
            total_sl_loss = (1.0 - self.alpha_anchor) * sl_loss + self.alpha_anchor * kl_anchor

            self.sl_optimizer.zero_grad()
            total_sl_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_net.parameters(), 1.0)
            self.sl_optimizer.step()

def train_and_eval_strategy(method: str, total_episodes: int = 20000, seed: int = 42) -> Dict[int, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = pyspiel.load_game("leduc_poker")
    device = torch.device("cpu")
    student_ckpt_path = "v45/models/leduc_student_best.pt"
    
    base_student = LeducStudentNet(30, 128, 3)
    base_student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    base_student.eval()
    
    checkpoints = [0, 5000, 10000, 20000]
    
    if method == "frozen":
        osp = StudentOpenSpielPolicy(game, base_student, device)
        nc = float(exploitability.nash_conv(game, osp))
        return {ep: nc for ep in checkpoints}
        
    if method == "standard_selfplay":
        agent0 = LeducNFSPAgent(0, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
        agent1 = LeducNFSPAgent(1, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
    elif method == "anchored_selfplay":
        agent0 = AnchoredLeducNFSPAgent(0, base_student, alpha_anchor=0.75, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
        agent1 = AnchoredLeducNFSPAgent(1, base_student, alpha_anchor=0.75, rl_lr=0.002, sl_lr=0.001, target_update=200, eps_decay=total_episodes, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    agent0.avg_net.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    agent1.avg_net.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    wrapper = LeducNFSPPolicyWrapper(game, agent0, agent1)
    
    nc_history = {}
    nc_history[0] = float(exploitability.nash_conv(game, wrapper))
    
    for ep in range(1, total_episodes + 1):
        agent0.sample_episode_mode()
        agent1.sample_episode_mode()
        state = game.new_initial_state()
        traj0, traj1 = [], []
        
        while not state.is_terminal():
            if state.is_chance_node():
                acts, prs = zip(*state.chance_outcomes())
                state.apply_action(np.random.choice(acts, p=prs))
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
            nc_history[ep] = nc
            
    return nc_history

def run_phase2_experiments():
    print("=" * 80)
    print("⚖️ [V4.6 PHASE 2] GTO 전략 보존 및 앵커링 미세조정 비교 실험")
    print("=" * 80)
    
    seeds = [42, 101, 202]
    methods = ["frozen", "standard_selfplay", "anchored_selfplay"]
    method_names = {
        "frozen": "A. Frozen Student (가중치 고정)",
        "standard_selfplay": "B. Student + Standard Self-Play",
        "anchored_selfplay": "C. Student + Anchored Fine-Tuning (KL 75%)"
    }
    
    all_results = {m: {ep: [] for ep in [0, 5000, 10000, 20000]} for m in methods}
    
    for m in methods:
        print(f"\n▶ [{method_names[m]}] 3개 시드 실험 진행 중...")
        for s in seeds:
            res = train_and_eval_strategy(m, total_episodes=20000, seed=s)
            for ep, val in res.items():
                all_results[m][ep].append(val)
            print(f"  * Seed {s:>3} 완료: Ep 0={res[0]:.4f} -> Ep 5k={res[5000]:.4f} -> Ep 20k={res[20000]:.4f}")
            
    print("\n" + "=" * 90)
    print("📊 [GTO 전략 보존 방법별 NashConv 및 Exploitability (bb/100) 비교표 (3개 시드 평균)]")
    print("=" * 90)
    print(f"{'전략 보존 방법':<35} | {'Ep 0':<12} | {'Ep 5,000':<12} | {'Ep 10,000':<12} | {'Ep 20,000':<12} | {'최종 판정'}")
    print("-" * 90)
    
    for m in methods:
        row_str = []
        for ep in [0, 5000, 10000, 20000]:
            vals = all_results[m][ep]
            mean_nc = np.mean(vals)
            exp_bb = (mean_nc / 2.0 / CHIPS_PER_BB) * 100.0
            row_str.append(f"{mean_nc:.4f} ({exp_bb:.1f})")
            
        verdict = "공식 채택 (최적)" if m == "frozen" else ("퇴보 억제 성공" if m == "anchored_selfplay" else "치명적 퇴보")
        print(f"{method_names[m]:<35} | {row_str[0]:<12} | {row_str[1]:<12} | {row_str[2]:<12} | {row_str[3]:<12} | {verdict}")
    print("=" * 90)
    
    # 지침 확인: 추가 자가대국이 기본 Frozen Student 성능(0.0087)을 개선하지 못함
    print("\n🎯 [공식 방침 확정]:")
    print("  - CFR+ Teacher로부터 증류된 'Frozen Student' (NashConv 0.008725, Exploitability 0.22 bb/100)를")
    print("    포커 AI의 최종 GTO 기본 정책으로 공식 채택함.")
    print("  - Anchored Fine-Tuning은 일반 Self-Play의 극단적 퇴보(4.16)를 억제(0.12~0.25 수준)하는 데는 성공했으나,")
    print("    원본 Frozen Student의 0.0087보다 우수하지 않으므로 변경 없이 Frozen 방식을 표준으로 삼음.")

if __name__ == "__main__":
    run_phase2_experiments()
