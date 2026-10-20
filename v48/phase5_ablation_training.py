"""
POKER AI V4.8 - PHASE 5: 4대 대조 실험 학습 엔진 (Ablation Study Training)
- 4대 실험 조건:
  A. REINFORCE + Pure Self-Play (기존 V4.7 방식 기준선)
  B. REINFORCE + League Training (기존 알고리즘 + 리그 상대 풀)
  C. PPO + Pure Self-Play (PPO Actor-Critic + 폐쇄형 자기대국)
  D. PPO + League Training (PPO Actor-Critic + 다채로운 리그 상대 풀)
- 공통 파라미터:
  - 총 판수: 10,000판 (체크포인트: 1k, 5k, 10k)
  - Batch Size / Update Frequency: 64 게임
  - Learning Rate: 3e-4 (Adam)
  - Discount Factor gamma: 1.0 (에피소딕 제로섬)
  - V4.8 공통 무작위 샘플러 (advance_chance_nodes) 필수 적용
"""

import sys
import os
import time
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import (
    get_5p_holdem_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic, compute_ppo_losses
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.league_manager import LeagueManager, BaseBot

def compute_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

class LearnerBotWrapper(BaseBot):
    """현재 학습 중인 모델을 BaseBot 인터페이스로 래핑"""
    def __init__(self, net, is_actor_critic: bool, device: torch.device):
        super().__init__("LearnerBot")
        self.net = net
        self.is_actor_critic = is_actor_critic
        self.device = device
        
    def act(self, state, cur_player, legal_acts) -> int:
        vec = state.information_state_tensor(cur_player)
        x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 4), dtype=torch.bool, device=self.device)
        for a in legal_acts: mask[0, a] = True
        with torch.no_grad():
            if self.is_actor_critic:
                logits, _ = self.net(x, mask)
            else:
                logits = self.net(x, mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        prs = [probs[a] for a in legal_acts]
        prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal_acts))/len(legal_acts)
        return int(np.random.choice(legal_acts, p=prs))

def train_ablation_model(
    exp_name: str,
    algorithm: str, # "REINFORCE" or "PPO"
    training_mode: str, # "SelfPlay" or "League"
    total_episodes: int = 10000,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n" + "=" * 80)
    print(f"🚀 [실험 {exp_name}] {algorithm} + {training_mode} ({total_episodes:,}판 학습 시작)")
    print("=" * 80)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    league_mgr = LeagueManager(device=device) if training_mode == "League" else None
    
    is_ppo = (algorithm == "PPO")
    if is_ppo:
        model = HoldemActorCritic(253, 256, 4).to(device)
    else:
        model = Holdem5PNet(253, 256, 4).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    learner_bot = LearnerBotWrapper(model, is_actor_critic=is_ppo, device=device)
    
    action_counts = np.zeros(4, dtype=int)
    allin_trash_count = 0
    trash_count = 0
    
    entropies = []
    p_losses = []
    v_losses = []
    
    start_time = time.time()
    batch_trajectories = [] # (st, mask, act, old_log_p, ret)
    
    for ep in range(1, total_episodes + 1):
        s2a = wrapper.get_seat_to_agent(ep)
        a2s = wrapper.get_agent_to_seat(ep)
        
        # 테이블 상대 구성 (Self-Play는 전원 Learner, League는 리그 풀에서 샘플링)
        if training_mode == "League":
            opp_bots = league_mgr.sample_table_opponents(learner_bot)
        else:
            opp_bots = [learner_bot] * 4
            
        st = game.new_initial_state()
        traj = {s: [] for s in range(5)}
        hero_hole_by_seat = {}
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                cur_str = st.information_state_string(cur_seat)
                
                if cur_seat not in hero_hole_by_seat:
                    hc, _ = parse_cards_from_info_str(cur_str)
                    hero_hole_by_seat[cur_seat] = hc
                    
                ag_id = s2a[cur_seat]
                
                # 에이전트 0(Main Learner) 또는 Self-Play인 경우 현재 모델로 의사결정 및 궤적 수집
                is_learner_acting = (ag_id == 0) or (training_mode == "SelfPlay")
                
                if is_learner_acting:
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: mask[0, a] = True
                    
                    with torch.no_grad():
                        if is_ppo:
                            logits, _ = model(x, mask)
                        else:
                            logits = model(x, mask)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(np.random.choice(legal, p=prs))
                    
                    action_counts[act] += 1
                    entropies.append(compute_entropy(probs))
                    
                    # 약한 패 올인 추적
                    hc = hero_hole_by_seat[cur_seat]
                    if len(hc) == 2 and hc[0][0] != hc[1][0] and hc[0][1] != hc[1][1]:
                        high_r = set(['T', 'J', 'Q', 'K', 'A'])
                        if hc[0][0] not in high_r and hc[1][0] not in high_r:
                            trash_count += 1
                            if act == 3: allin_trash_count += 1
                            
                    old_log_p = log_probs[act]
                    traj[cur_seat].append((vec, mask.squeeze(0).cpu().numpy(), act, old_log_p))
                else:
                    # 리그 상대 봇 행동
                    opp_bot = opp_bots[(ag_id - 1) % len(opp_bots)]
                    act = opp_bot.act(st, cur_seat, legal)
                    
                st.apply_action(act)
                
        rets = st.returns()
        # 보상 정규화 (1 BB = 100 chips 단위)
        for s in range(5):
            ag_id = s2a[s]
            if (ag_id == 0) or (training_mode == "SelfPlay"):
                r_bb = rets[s] / 100.0
                for v, m, a, olp in traj[s]:
                    batch_trajectories.append((v, m, a, olp, r_bb))
                    
        # 64 에피소드마다 정책 업데이트
        if ep % 64 == 0 and batch_trajectories:
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)
            
            if is_ppo:
                # PPO Advantage: (Return - V(s))
                with torch.no_grad():
                    _, pred_v = model(t_states, t_masks)
                    adv = t_returns - pred_v
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
                    
                # PPO 4회 Epoch 미니배치 업데이트
                for _ in range(4):
                    tot_l, p_l, v_l, ent_val = compute_ppo_losses(
                        model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv
                    )
                    optimizer.zero_grad()
                    tot_l.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                p_losses.append(p_l.item())
                v_losses.append(v_l.item())
            else:
                # REINFORCE Update
                adv = (t_returns - t_returns.mean()) / (t_returns.std() + 1e-8) if len(t_returns) > 1 else t_returns
                logits = model(t_states, t_masks)
                log_p = torch.log_softmax(logits, dim=-1).gather(1, t_actions.unsqueeze(1)).squeeze(1)
                loss = -(log_p * adv).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                p_losses.append(loss.item())
                
            batch_trajectories = []
            
        if ep % 2000 == 0:
            elapsed = time.time() - start_time
            speed = ep / elapsed
            m_ent = np.mean(entropies[-500:]) if entropies else 0.0
            print(f"  Ep {ep:5d}/{total_episodes:,} | Ent: {m_ent:.3f} | 속도: {speed:5.1f} g/s | All-in: {action_counts[3]/sum(action_counts)*100:4.1f}%")

    total_time = time.time() - start_time
    save_path = f"v48/models/{exp_name}_step_10k.pt"
    torch.save(model.state_dict(), save_path)
    
    act_sum = max(1, sum(action_counts))
    allin_pct = (action_counts[3] / act_sum) * 100
    trash_allin_pct = (allin_trash_count / max(1, trash_count)) * 100
    final_entropy = float(np.mean(entropies[-1000:])) if entropies else 0.0
    avg_v_loss = float(np.mean(v_losses)) if v_losses else 0.0
    avg_p_loss = float(np.mean(p_losses)) if p_losses else 0.0
    
    print(f"✅ [{exp_name} 완료] 소요: {total_time:.1f}초 | 올인 비율: {allin_pct:.1f}% | 약한 패 올인율: {trash_allin_pct:.1f}% | 모델 저장: {save_path}")
    
    return {
        "exp_name": exp_name,
        "algorithm": algorithm,
        "mode": training_mode,
        "save_path": save_path,
        "total_time": total_time,
        "allin_pct": allin_pct,
        "trash_allin_pct": trash_allin_pct,
        "entropy": final_entropy,
        "value_loss": avg_v_loss,
        "policy_loss": avg_p_loss
    }

def run_phase5_ablation():
    print("=" * 80)
    print("🔬 [V4.8 PHASE 5] 4대 대조 실험 (Ablation Study) 훈련 및 비교")
    print("=" * 80)
    
    experiments = [
        ("Exp_A_REINFORCE_SelfPlay", "REINFORCE", "SelfPlay"),
        ("Exp_B_REINFORCE_League",   "REINFORCE", "League"),
        ("Exp_C_PPO_SelfPlay",       "PPO",       "SelfPlay"),
        ("Exp_D_PPO_League",         "PPO",       "League")
    ]
    
    results = []
    for exp_id, algo, mode in experiments:
        res = train_ablation_model(exp_id, algo, mode, total_episodes=10000, seed=42)
        results.append(res)
        
    print("\n" + "=" * 80)
    print("📊 [4대 대조 실험 10,000판 학습 결과 종합 비교표]")
    print("=" * 80)
    print(f"{'실험명':<26} | {'알고리즘':<10} | {'학습방식':<10} | {'올인율':<8} | {'약한패 올인율':<12} | {'엔트로피':<8} | {'소요시간'}")
    print("-" * 80)
    for r in results:
        print(f"{r['exp_name']:<26} | {r['algorithm']:<10} | {r['mode']:<10} | {r['allin_pct']:>5.1f}%  | {r['trash_allin_pct']:>9.1f}%   | {r['entropy']:>6.3f}   | {r['total_time']:>5.1f}초")
    print("=" * 80)

if __name__ == "__main__":
    run_phase5_ablation()
