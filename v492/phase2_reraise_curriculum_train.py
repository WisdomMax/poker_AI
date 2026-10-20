"""
POKER AI V4.9.2 - PHASE 2 & 6: 재레이즈 대응 학습 및 통제된 추가 학습 (Curriculum PPO Training)
- Baseline: 기존 PPO League (v49 풀)
- Treatment: 재레이즈 상황 학습 노출을 개선한 Curriculum PPO League
- 5개 독립 시드: [42, 101, 202, 303, 404]
- 체크포인트: Step 0, 10k, 20k, 30k, 40k
- 상세 메트릭 추적:
  * 최초 베팅 빈도, 상대 레이즈 빈도, 3-Bet 직면 빈도, 4-Bet 직면 빈도
  * 레이즈 직면 후 Fold, Call, Re-raise 빈도
  * AA 3-Bet 직면 시 Fold/Call 확률 추이
  * Learner Decisions, Transitions 수, Optimizer Updates 수
"""

import os
import sys
import time
import math
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, sample_chance_action
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A, compute_ppo_losses_5a
from v49.league_manager_5a import LeagueManager5A, LearnerBotWrapper5A
from v48.ppo_actor_critic import HoldemActorCritic as HoldemActorCritic4A
from v48.opponents_pool import get_8_opponents
from v492.curriculum_league_manager import CurriculumLeagueManager5A

SEEDS = [42, 101, 202, 303, 404]
TOTAL_EPISODES = 40000
CHECKPOINT_INTERVAL = 10000
EVAL_HANDS = 400

def create_transfer_model(v483_seed: int, device: torch.device) -> HoldemActorCritic5A:
    """V4.8.3 4A 모델에서 5A 신경망으로 가중치 전이"""
    path_4a = f"v483/models/seed_{v483_seed}_step_20000.pt"
    if not os.path.exists(path_4a):
        # 대안 경로
        path_4a = f"v482/models/seed_{v483_seed}_step_20000.pt"
    
    m_4a = HoldemActorCritic4A(input_dim=253, hidden_dim=256, action_dim=4).to(device)
    if os.path.exists(path_4a):
        m_4a.load_state_dict(torch.load(path_4a, map_location=device))
        
    m_5a = HoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    m_5a.load_from_4a_model(m_4a)
    return m_5a

def evaluate_aa_3bet_policy(model: HoldemActorCritic5A, game: pyspiel.Game, device: torch.device) -> Tuple[float, float]:
    """AA Half-Pot 오픈 후 3-Bet All-in 직면 상태에서 Fold 및 Call 확률 측정"""
    st = game.new_initial_state()
    deck = [c for c in range(52) if c not in (48, 51)]
    for p in range(5):
        if p == 2:
            st.apply_action(48); st.apply_action(51)
        else:
            st.apply_action(deck.pop(0)); st.apply_action(deck.pop(0))
    st.apply_action(4) # P2 HalfPot
    st.apply_action(3) # P3 3-Bet All-in
    st.apply_action(0); st.apply_action(0); st.apply_action(0) # others fold
    
    vec = st.information_state_tensor(2)
    legal = st.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    return float(probs[0]), float(probs[1])

def evaluate_model_benchmark(model: HoldemActorCritic5A, game: pyspiel.Game, device: torch.device, num_hands: int = EVAL_HANDS) -> float:
    """8종 상대 봇을 상대로 벤치마크 bb/100 측정"""
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = get_8_opponents()
    rng_eval = np.random.RandomState(99999)
    rng_opp = random.Random(88888)
    
    total_chips = 0.0
    model.eval()
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        table_opps = rng_opp.sample(opponents, 4)
        
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_eval)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    vec = st.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    p_sum = sum(prs)
                    prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval.choice(legal, p=prs))
                else:
                    ag_id = s2a[cur]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur, legal)
                st.apply_action(act)
        total_chips += st.returns()[hero_seat]
        
    return float(total_chips / num_hands)

def train_single_condition(
    condition_name: str,
    seed_val: int,
    use_curriculum: bool,
    device: torch.device = torch.device("cpu")
):
    print(f"\n[{condition_name}] Seed {seed_val} 학습 시작 (Episodes: {TOTAL_EPISODES})...")
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    
    # 모델 초기화 (V4.8.3 Warm-Start 전이)
    model = create_transfer_model(seed_val, device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, eps=1e-5)
    
    # 리그 매니저 선택
    rng_league = random.Random(seed_val + 500)
    if use_curriculum:
        league_mgr = CurriculumLeagueManager5A(device=device, rng=rng_league)
    else:
        league_mgr = LeagueManager5A(device=device, rng=rng_league)
        
    learner_bot = LearnerBotWrapper5A(model, device=device)
    
    # 메트릭 카운터
    total_learner_decisions = 0
    total_transitions_collected = 0
    total_optimizer_updates = 0
    
    initial_bet_count = 0
    opp_raise_count = 0
    facing_3bet_count = 0
    facing_4bet_count = 0
    fold_after_raise_count = 0
    call_after_raise_count = 0
    reraise_after_raise_count = 0
    
    batch_trajectories = []
    records = []
    
    rng_cards = np.random.RandomState(seed_val + 1000)
    rng_act = np.random.RandomState(seed_val + 2000)
    
    t_start = time.time()
    
    # Step 0 기록
    aa_fold_p, aa_call_p = evaluate_aa_3bet_policy(model, game, device)
    bb100_0 = evaluate_model_benchmark(model, game, device, num_hands=200)
    
    records.append({
        'condition': condition_name,
        'seed': seed_val,
        'step': 0,
        'bb100': bb100_0,
        'aa_3bet_fold_pct': aa_fold_p * 100.0,
        'aa_3bet_call_pct': aa_call_p * 100.0,
        'facing_3bet_count': 0,
        'facing_4bet_count': 0,
        'fold_after_raise_count': 0,
        'call_after_raise_count': 0,
        'reraise_after_raise_count': 0,
        'total_decisions': 0,
        'total_transitions': 0,
        'total_updates': 0,
        'elapsed_sec': 0.0
    })
    
    save_path_0 = f"v492/models/{condition_name}_seed_{seed_val}_step_0.pt"
    os.makedirs("v492/models", exist_ok=True)
    torch.save(model.state_dict(), save_path_0)

    for ep in range(1, TOTAL_EPISODES + 1):
        s2a = wrapper.get_seat_to_agent(ep)
        opp_bots = league_mgr.sample_table_opponents(learner_bot)
        
        st = game.new_initial_state()
        traj = {s: [] for s in range(5)}
        
        # 핸드 내부 레이즈/3-bet 추적 플래그
        learner_opened = False
        learner_faced_3bet = False
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_cards)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                ag_id = s2a[cur_seat]
                is_learner = (ag_id == 0) or (opp_bots[(ag_id - 1) % len(opp_bots)] == learner_bot)
                
                info = st.information_state_string(cur_seat)
                seq = info.split('Sequences: ')[1].split(']')[0] if 'Sequences: ' in info else ""
                
                if is_learner:
                    total_learner_decisions += 1
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        
                    prs = [probs[a] for a in legal]
                    p_sum = sum(prs)
                    prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_act.choice(legal, p=prs))
                    
                    # 베팅 분석 카운팅
                    # 1. 최초 베팅(오픈) 여부: 프리플롭에서 아무도 레이즈 안 했을 때 2, 3, 4 선택
                    if 'r' not in seq and act in [2, 3, 4]:
                        initial_bet_count += 1
                        learner_opened = True
                        
                    # 2. 3-Bet 직면 여부: 자신이 오픈 후 상대 레이즈로 재진입
                    if learner_opened and seq.count('r') >= 2:
                        facing_3bet_count += 1
                        learner_faced_3bet = True
                        if act == 0:
                            fold_after_raise_count += 1
                        elif act == 1:
                            call_after_raise_count += 1
                        elif act in [2, 3, 4]:
                            reraise_after_raise_count += 1
                            
                    # 3. 4-Bet 직면 여부
                    if seq.count('r') >= 3:
                        facing_4bet_count += 1

                    traj[cur_seat].append((vec, m.squeeze(0).cpu().numpy(), act, log_probs[act]))
                else:
                    opp_bot = opp_bots[(ag_id - 1) % len(opp_bots)]
                    act = opp_bot.act(st, cur_seat, legal)
                    if act in [2, 3, 4]:
                        opp_raise_count += 1
                        
                st.apply_action(act)
                
        # 트랜지션 수집
        rets = st.returns()
        for s in range(5):
            ag_id = s2a[s]
            if (ag_id == 0) or (opp_bots[(ag_id - 1) % len(opp_bots)] == learner_bot):
                r_bb = rets[s] / 100.0
                for v, m, a, olp in traj[s]:
                    batch_trajectories.append((v, m, a, olp, r_bb))
                    total_transitions_collected += 1
                    
        # 64 에피소드마다 PPO 업데이트
        if ep % 64 == 0 and batch_trajectories:
            total_optimizer_updates += 1
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                _, values = model(t_states, t_masks)
                advantages = t_returns - values
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std
                    
            model.train()
            for _ in range(4): # 4 PPO epochs
                pol_loss, val_loss, ent_loss, tot_loss = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, advantages
                )
                optimizer.zero_grad()
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
            batch_trajectories = []
            
        # 체크포인트 저장 및 평가
        if ep % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - t_start
            aa_f_p, aa_c_p = evaluate_aa_3bet_policy(model, game, device)
            bb100_val = evaluate_model_benchmark(model, game, device, num_hands=EVAL_HANDS)
            
            save_path = f"v492/models/{condition_name}_seed_{seed_val}_step_{ep}.pt"
            torch.save(model.state_dict(), save_path)
            
            print(f"  Step {ep:5d} | bb100: {bb100_val:+6.1f} | AA 3-Bet Fold: {aa_f_p*100:5.1f}%, Call: {aa_c_p*100:5.1f}% | 3-Bet Faced: {facing_3bet_count:,} | Elapsed: {elapsed:5.1f}s")
            
            records.append({
                'condition': condition_name,
                'seed': seed_val,
                'step': ep,
                'bb100': bb100_val,
                'aa_3bet_fold_pct': aa_f_p * 100.0,
                'aa_3bet_call_pct': aa_c_p * 100.0,
                'facing_3bet_count': facing_3bet_count,
                'facing_4bet_count': facing_4bet_count,
                'fold_after_raise_count': fold_after_raise_count,
                'call_after_raise_count': call_after_raise_count,
                'reraise_after_raise_count': reraise_after_raise_count,
                'total_decisions': total_learner_decisions,
                'total_transitions': total_transitions_collected,
                'total_updates': total_optimizer_updates,
                'elapsed_sec': elapsed
            })

    return records

def run_phase2_training():
    print("=" * 80)
    print("POKER AI V4.9.2 - PHASE 2 & 6: 재레이즈 대응 커리큘럼 PPO 학습")
    print("=" * 80)
    
    device = torch.device("cpu")
    all_records = []
    
    # 1. Baseline 조건 학습 (5개 시드)
    print("\n" + "=" * 50)
    print("조건 1: Baseline PPO League 학습 시작")
    print("=" * 50)
    for s in SEEDS:
        recs = train_single_condition("Baseline", s, use_curriculum=False, device=device)
        all_records.extend(recs)
        
    # 2. Treatment (Curriculum) 조건 학습 (5개 시드)
    print("\n" + "=" * 50)
    print("조건 2: Curriculum PPO League 학습 시작 (재레이즈 노출 강화)")
    print("=" * 50)
    for s in SEEDS:
        recs = train_single_condition("Treatment_Curriculum", s, use_curriculum=True, device=device)
        all_records.extend(recs)
        
    # CSV 저장
    os.makedirs("v492/data", exist_ok=True)
    out_csv = "v492/data/phase2_training_metrics.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nPhase 2 & 6 전체 학습 메트릭 저장 완료: {out_csv} (총 {len(all_records)} 행)")

if __name__ == '__main__':
    run_phase2_training()
