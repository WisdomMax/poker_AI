"""
POKER AI V4.9 - PHASE 4: 5-Action 독립 대조 실험 (A/B/C Controlled Experiment)
- 비교 3대 조건:
  Condition A: 4-Action PPO League (V4.8.3 체크포인트를 호환 어댑터로 5A 환경에서 평가)
  Condition B: 5-Action PPO League (무작위 초기화 Scratch 학습, 20k)
  Condition C: 5-Action PPO League (V4.8.3 가중치 이전 Warm-Start 전이 학습, 20k)
- 5개 독립 시드: [42, 101, 202, 303, 404]
- 체크포인트: Step 0, Step 5k, Step 10k, Step 15k, Step 20k
- 기록 메트릭:
  수익률(bb/100), Half-Pot 사용률(%), All-in 빈도, Fold 빈도, TV Distance(AA-72o),
  9개 핸드 정책, Learner 결정 수, Transitions 수, Updates 수, 소요 시간
"""

import sys
import os
import time
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, ACTION_NAMES_5A
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A, compute_ppo_losses_5a
from v49.league_manager_5a import LeagueManager5A, LearnerBotWrapper5A
from v49.model_adapter_4a_to_5a import FourActionToFiveActionAdapter, FourActionAdapterBot
from v48.ppo_actor_critic import HoldemActorCritic
from v481.phase7_controlled_learning import compute_tv_distance
from v48.opponents_pool import get_8_opponents
from v49.hand_tensors_5a import get_standard_hand_tensors_5a

SEEDS = [42, 101, 202, 303, 404]
CHECKPOINTS = [0, 5000, 10000, 15000, 20000]
TOTAL_EPISODES = 20000
EVAL_HANDS = 500
EVAL_SEED_BASE = 55555

def evaluate_model_5a(
    model: torch.nn.Module,
    num_hands: int = EVAL_HANDS,
    eval_seed: int = EVAL_SEED_BASE,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """5-Action 환경에서 고정된 8종 봇을 상대로 500판 성적 및 행동 분포 측정"""
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = get_8_opponents()
    rng_eval = np.random.RandomState(eval_seed)
    rng_opp = random.Random(eval_seed + 1)
    
    chips_total = 0.0
    wins = 0
    action_counts = {a: 0 for a in range(5)}
    total_acts = 0
    
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
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                if cur_seat == hero_seat:
                    total_acts += 1
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval.choice(legal, p=prs))
                    action_counts[act] += 1
                else:
                    ag_id = s2a[cur_seat]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur_seat, legal)
                st.apply_action(act)
                
        ret = st.returns()[hero_seat]
        chips_total += ret
        if ret == max(st.returns()) and ret > 0: wins += 1
        
    bb100 = chips_total / float(num_hands)
    return {
        "bb100": bb100,
        "win_rate": (wins / float(num_hands)) * 100.0,
        "fold_pct": (action_counts[0] / max(1, total_acts)) * 100.0,
        "call_pct": (action_counts[1] / max(1, total_acts)) * 100.0,
        "pot_pct": (action_counts[2] / max(1, total_acts)) * 100.0,
        "allin_pct": (action_counts[3] / max(1, total_acts)) * 100.0,
        "halfpot_pct": (action_counts[4] / max(1, total_acts)) * 100.0,
        "total_acts": total_acts
    }

def inspect_hand_policies_5a(
    model: torch.nn.Module,
    hand_tensors: Dict,
    device: torch.device = torch.device("cpu")
) -> Dict:
    model.eval()
    policies = {}
    with torch.no_grad():
        for hname, (x, m, _) in hand_tensors.items():
            x = x.to(device)
            m = m.to(device)
            logits, _ = model(x, m)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            policies[hname] = probs
            
    tv_aa_72o = compute_tv_distance(policies['AA'][:4], policies['72o'][:4])
    return policies, tv_aa_72o

def train_5a_model(
    seed_val: int,
    condition_name: str,
    warm_start_path: str = None,
    hand_tensors: Dict = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[Dict], Dict]:
    print(f"\n>>> [{condition_name} | 시드 {seed_val}] 학습 시작 (20,000 에피소드)...")
    
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    rng_cards = np.random.RandomState(seed_val + 1000)
    rng_act = np.random.RandomState(seed_val + 2000)
    rng_league = random.Random(seed_val + 3000)
    
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    league_mgr = LeagueManager5A(device=device, rng=rng_league)
    
    model = HoldemActorCritic5A(265, 256, 5).to(device)
    
    # Warm-Start 가중치 전이
    if warm_start_path and os.path.exists(warm_start_path):
        m4 = HoldemActorCritic(253, 256, 4).to(device)
        m4.load_state_dict(torch.load(warm_start_path, map_location=device))
        model.load_from_4a_model(m4)
        print(f"  ✓ {warm_start_path} 가중치 성공적으로 전이 완료")
        
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    learner_bot = LearnerBotWrapper5A(model, device=device)
    
    total_learner_decisions = 0
    total_transitions_collected = 0
    total_optimizer_updates = 0
    
    batch_trajectories = []
    checkpoint_records = []
    t_start = time.time()
    
    # Step 0 체크포인트
    policies, tv_dist = inspect_hand_policies_5a(model, hand_tensors, device)
    eval_res = evaluate_model_5a(model, num_hands=EVAL_HANDS, eval_seed=EVAL_SEED_BASE, device=device)
    torch.save(model.state_dict(), f"v49/models/{condition_name}_seed_{seed_val}_step_0.pt")
    
    checkpoint_records.append({
        "condition": condition_name,
        "seed": seed_val,
        "step": 0,
        "bb100": eval_res["bb100"],
        "win_rate": eval_res["win_rate"],
        "halfpot_pct": eval_res["halfpot_pct"],
        "pot_pct": eval_res["pot_pct"],
        "allin_pct": eval_res["allin_pct"],
        "fold_pct": eval_res["fold_pct"],
        "tv_aa_72o": tv_dist,
        "aa_allin": policies["AA"][3],
        "aa_halfpot": policies["AA"][4],
        "jts_allin": policies["JTs"][3],
        "jts_halfpot": policies["JTs"][4],
        "72o_fold": policies["72o"][0]
    })
    
    for ep in range(1, TOTAL_EPISODES + 1):
        s2a = wrapper.get_seat_to_agent(ep)
        opp_bots = league_mgr.sample_table_opponents(learner_bot)
        
        st = game.new_initial_state()
        traj = {s: [] for s in range(5)}
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_cards)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                ag_id = s2a[cur_seat]
                
                is_learner = (ag_id == 0) or (opp_bots[(ag_id - 1) % len(opp_bots)] == learner_bot)
                
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
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_act.choice(legal, p=prs))
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
                    total_transitions_collected += 1
                    
        # 64 에피소드마다 On-Policy 갱신
        if ep % 64 == 0 and batch_trajectories:
            total_optimizer_updates += 1
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
                tot_l, _, _, _ = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv, ent_coef=0.005
                )
                optimizer.zero_grad()
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            batch_trajectories = []
            
        if ep in CHECKPOINTS and ep > 0:
            torch.save(model.state_dict(), f"v49/models/{condition_name}_seed_{seed_val}_step_{ep}.pt")
            policies, tv_dist = inspect_hand_policies_5a(model, hand_tensors, device)
            eval_res = evaluate_model_5a(model, num_hands=EVAL_HANDS, eval_seed=EVAL_SEED_BASE + ep, device=device)
            
            checkpoint_records.append({
                "condition": condition_name,
                "seed": seed_val,
                "step": ep,
                "bb100": eval_res["bb100"],
                "win_rate": eval_res["win_rate"],
                "halfpot_pct": eval_res["halfpot_pct"],
                "pot_pct": eval_res["pot_pct"],
                "allin_pct": eval_res["allin_pct"],
                "fold_pct": eval_res["fold_pct"],
                "tv_aa_72o": tv_dist,
                "aa_allin": policies["AA"][3],
                "aa_halfpot": policies["AA"][4],
                "jts_allin": policies["JTs"][3],
                "jts_halfpot": policies["JTs"][4],
                "72o_fold": policies["72o"][0]
            })
            print(f"  [Step {ep:5d}] 수익률: {eval_res['bb100']:>+7.1f} bb/100 | HalfPot: {eval_res['halfpot_pct']:>4.1f}% | AA Allin: {policies['AA'][3]:.2f} (HalfPot: {policies['AA'][4]:.2f}) | JTs Allin: {policies['JTs'][3]:.2f}")
            
    t_elapsed = time.time() - t_start
    volume_rec = {
        "condition": condition_name,
        "seed": seed_val,
        "total_games": TOTAL_EPISODES,
        "learner_decisions": total_learner_decisions,
        "transitions_collected": total_transitions_collected,
        "optimizer_updates": total_optimizer_updates,
        "training_time_sec": round(t_elapsed, 2)
    }
    return checkpoint_records, volume_rec

def run_phase4_controlled_experiments():
    print("=" * 90)
    print("🛸 [POKER AI V4.9] PHASE 4: 5-Action 독립 대조 실험 (Cond A vs Cond B vs Cond C)")
    print("=" * 90)
    
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors_5a(device=device)
    
    all_metrics = []
    all_volumes = []
    
    # -------------------------------------------------------------
    # 1. Condition A: 4-Action Baseline (어댑터 5A 환경 평가)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [Condition A] 4-Action PPO 모델 5종 (어댑터 평가)")
    print("=" * 60)
    for s in SEEDS:
        ckpt_path = f"v483/models/seed_{s}_step_20000.pt"
        if os.path.exists(ckpt_path):
            m4 = HoldemActorCritic(253, 256, 4).to(device)
            m4.load_state_dict(torch.load(ckpt_path, map_location=device))
            adapter = FourActionToFiveActionAdapter(m4).to(device)
            
            policies, tv_dist = inspect_hand_policies_5a(adapter, hand_tensors, device)
            eval_res = evaluate_model_5a(adapter, num_hands=EVAL_HANDS, eval_seed=EVAL_SEED_BASE + 20000, device=device)
            
            rec = {
                "condition": "CondA_4A_Baseline",
                "seed": s,
                "step": 20000,
                "bb100": eval_res["bb100"],
                "win_rate": eval_res["win_rate"],
                "halfpot_pct": eval_res["halfpot_pct"],
                "pot_pct": eval_res["pot_pct"],
                "allin_pct": eval_res["allin_pct"],
                "fold_pct": eval_res["fold_pct"],
                "tv_aa_72o": tv_dist,
                "aa_allin": policies["AA"][3],
                "aa_halfpot": policies["AA"][4],
                "jts_allin": policies["JTs"][3],
                "jts_halfpot": policies["JTs"][4],
                "72o_fold": policies["72o"][0]
            }
            all_metrics.append(rec)
            print(f"  ✓ Seed {s:<4d}: 수익률 {eval_res['bb100']:>+7.1f} bb/100 | HalfPot: {eval_res['halfpot_pct']:.1f}% | AA Allin: {policies['AA'][3]:.2f} | JTs Allin: {policies['JTs'][3]:.2f}")

    # -------------------------------------------------------------
    # 2. Condition B: 5-Action Scratch 학습 (5개 시드)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [Condition B] 5-Action PPO Scratch 학습 (5개 시드 x 20k)")
    print("=" * 60)
    for s in SEEDS:
        ckpt_recs, vol_rec = train_5a_model(
            seed_val=s, condition_name="CondB_5A_Scratch", warm_start_path=None,
            hand_tensors=hand_tensors, device=device
        )
        all_metrics.extend(ckpt_recs)
        all_volumes.append(vol_rec)

    # -------------------------------------------------------------
    # 3. Condition C: 5-Action Transfer 학습 (5개 시드)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [Condition C] 5-Action PPO Transfer (Warm-Start) 학습 (5개 시드 x 20k)")
    print("=" * 60)
    for s in SEEDS:
        warm_path = f"v483/models/seed_{s}_step_20000.pt"
        ckpt_recs, vol_rec = train_5a_model(
            seed_val=s, condition_name="CondC_5A_Transfer", warm_start_path=warm_path,
            hand_tensors=hand_tensors, device=device
        )
        all_metrics.extend(ckpt_recs)
        all_volumes.append(vol_rec)
        
    # CSV 저장
    csv_metrics = "v49/data/phase4_experiment_metrics.csv"
    with open(csv_metrics, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
        
    csv_vol = "v49/data/phase4_training_volume.csv"
    with open(csv_vol, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_volumes[0].keys()))
        writer.writeheader()
        writer.writerows(all_volumes)
        
    print("\n" + "=" * 90)
    print("🎉 PHASE 4 독립 대조 실험 전체 완료!")
    print(f"  - 메트릭 CSV 저장: {csv_metrics}")
    print(f"  - 훈련 볼륨 CSV 저장: {csv_vol}")
    print("=" * 90)

if __name__ == "__main__":
    run_phase4_controlled_experiments()
