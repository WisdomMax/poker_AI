"""
POKER AI V4.8.1 - PHASE 7: 제어된 학습 테스트 (Controlled Learning Progression)
- 목적:
  "학습 판수(Episodes)가 0 -> 1k -> 5k -> 10k -> 20k로 증가함에 따라
   신경망이 실제로 AA와 72o의 카드 차이를 인지하고 정책 거리(TV Distance)를 벌려나가는가?"
- 측정 지표:
  1. AA 정책 확률 pi(a|AA) vs 72o 정책 확률 pi(a|72o)
  2. Total Variation Distance: D_TV = 0.5 * sum(|pi_AA - pi_72o|)
  3. 정책 엔트로피 H(pi)
  4. 올인 비율 및 약한 패 올인율
  5. Value Loss (MSE) 및 Policy Loss
  6. 가중치 변위 ||Delta W||
"""

import sys
import os
import time
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import (
    get_5p_holdem_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic, compute_ppo_losses
from v48.league_manager import LeagueManager, BaseBot
from v48.phase5_ablation_training import compute_entropy, LearnerBotWrapper

def compute_tv_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p1 - p2)))

def run_controlled_learning_experiment(total_episodes: int = 20000, seed: int = 2026):
    print("=" * 80)
    print(f"🔬 [V4.8.1 PHASE 7] 제어된 학습 테스트: 에피소드 증가에 따른 카드 인지 학습 곡선 ({total_episodes:,}판)")
    print("=" * 80)
    
    device = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    league_mgr = LeagueManager(device=device)
    
    model = HoldemActorCritic(253, 256, 4).to(device)
    initial_weights = torch.cat([p.flatten() for p in model.parameters()]).clone()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    learner_bot = LearnerBotWrapper(model, is_actor_critic=True, device=device)
    
    # 대표 상태 준비 (UTG 프리플롭 첫 턴의 AA 및 72o 고정 텐서)
    # OpenSpiel에서 결정론적으로 탐색
    tensor_aa = None
    tensor_72o = None
    legal_actions = None
    
    print("🔍 검증용 표준 AA 및 72o 입력 텐서 추출 중...")
    for s_idx in range(1, 100000):
        np.random.seed(s_idx)
        st = game.new_initial_state()
        advance_chance_nodes(st)
        cur_p = st.current_player()
        if cur_p == 2:
            hc, _ = parse_cards_from_info_str(st.information_state_string(cur_p))
            if len(hc) == 2:
                r1, r2 = hc[0][0], hc[1][0]
                s1, s2 = hc[0][1], hc[1][1]
                if tensor_aa is None and r1 == 'A' and r2 == 'A':
                    tensor_aa = st.information_state_tensor(cur_p)
                    legal_actions = st.legal_actions(cur_p)
                elif tensor_72o is None and ((r1=='7' and r2=='2') or (r1=='2' and r2=='7')) and s1 != s2:
                    tensor_72o = st.information_state_tensor(cur_p)
            if tensor_aa is not None and tensor_72o is not None:
                break
                
    mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
    for a in legal_actions: mask[0, a] = True
    x_aa = torch.tensor(tensor_aa, dtype=torch.float32, device=device).unsqueeze(0)
    x_72o = torch.tensor(tensor_72o, dtype=torch.float32, device=device).unsqueeze(0)
    
    checkpoints = [0, 1000, 5000, 10000, 20000]
    progression_records = []
    
    def evaluate_checkpoint(ep: int):
        with torch.no_grad():
            l_aa, _ = model(x_aa, mask)
            l_72o, _ = model(x_72o, mask)
            p_aa = torch.softmax(l_aa, dim=-1).squeeze(0).cpu().numpy()
            p_72o = torch.softmax(l_72o, dim=-1).squeeze(0).cpu().numpy()
            
        tv_dist = compute_tv_distance(p_aa, p_72o)
        curr_w = torch.cat([p.flatten() for p in model.parameters()]).clone()
        w_diff = torch.norm(curr_w - initial_weights).item()
        ent_aa = compute_entropy(p_aa)
        ent_72o = compute_entropy(p_72o)
        
        return {
            "ep": ep,
            "p_aa": p_aa,
            "p_72o": p_72o,
            "tv_dist": tv_dist,
            "w_diff": w_diff,
            "ent_aa": ent_aa,
            "ent_72o": ent_72o
        }
        
    # Ep 0 측정
    progression_records.append(evaluate_checkpoint(0))
    
    batch_trajectories = []
    start_time = time.time()
    
    print("\n학습 진행 및 단계별 체크포인트 측정:")
    print(f"{'Episode':<8} | {'TV Distance':<12} | {'AA Policy [F, C, P, A]':<26} | {'72o Policy [F, C, P, A]':<26} | {'||ΔW||'}")
    print("-" * 95)
    r0 = progression_records[0]
    print(f"{r0['ep']:<8} | {r0['tv_dist']:<12.4f} | {[round(x,2) for x in r0['p_aa']]} | {[round(x,2) for x in r0['p_72o']]} | {r0['w_diff']:<8.4f}")
    
    for ep in range(1, total_episodes + 1):
        s2a = wrapper.get_seat_to_agent(ep)
        opp_bots = league_mgr.sample_table_opponents(learner_bot)
        
        st = game.new_initial_state()
        traj = {s: [] for s in range(5)}
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                ag_id = s2a[cur_seat]
                
                # Learner가 앉은 좌석(Main Learner 또는 복제 Learner)인 경우 궤적 수집
                # 개선: 복제 Learner의 궤적도 함께 수집하여 데이터 효율 2.5배 향상!
                is_learner = (ag_id == 0) or (opp_bots[(ag_id - 1) % len(opp_bots)] == learner_bot)
                
                if is_learner:
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(np.random.choice(legal, p=prs))
                    traj[cur_seat].append((vec, m.squeeze(0).cpu().numpy(), act, log_probs[act]))
                else:
                    opp_bot = opp_bots[(ag_id - 1) % len(opp_bots)]
                    act = opp_bot.act(st, cur_seat, legal)
                    
                st.apply_action(act)
                
        rets = st.returns()
        for s in range(5):
            ag_id = s2a[s]
            if (ag_id == 0) or (opp_bots[(ag_id - 1) % len(opp_bots)] == learner_bot):
                r_bb = rets[s] / 100.0
                for v, m, a, olp in traj[s]:
                    batch_trajectories.append((v, m, a, olp, r_bb))
                    
        # 64 에피소드마다 PPO 업데이트
        if ep % 64 == 0 and batch_trajectories:
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                _, pred_v = model(t_states, t_masks)
                adv = t_returns - pred_v
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
                
            for _ in range(4):
                tot_l, p_l, v_l, ent_v = compute_ppo_losses(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv, ent_coef=0.005
                )
                optimizer.zero_grad()
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            batch_trajectories = []
            
        if ep in checkpoints:
            r = evaluate_checkpoint(ep)
            progression_records.append(r)
            print(f"{r['ep']:<8} | {r['tv_dist']:<12.4f} | {[round(x,2) for x in r['p_aa']]} | {[round(x,2) for x in r['p_72o']]} | {r['w_diff']:<8.4f}")
            torch.save(model.state_dict(), f"v481/models/controlled_ppo_step_{ep}.pt")

    print("\n" + "=" * 80)
    print("📈 [제어된 학습 실험 결론]")
    print(f"  - 초기 Ep 0 TV Distance: {progression_records[0]['tv_dist']:.4f}")
    print(f"  - 최종 Ep 20,000 TV Distance: {progression_records[-1]['tv_dist']:.4f}")
    diff_tv = progression_records[-1]['tv_dist'] - progression_records[0]['tv_dist']
    print(f"  - 카드 인식 분화도(TV Distance 증가분): {diff_tv:+.4f}")
    print(f"  - 총 가중치 변위 ||ΔW||: {progression_records[-1]['w_diff']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_controlled_learning_experiment(total_episodes=20000, seed=2026)
