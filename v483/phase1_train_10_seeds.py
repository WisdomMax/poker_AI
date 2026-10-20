"""
POKER AI V4.8.3 - PHASE 1: 독립 학습 10회 (10 Independent Seeds Training)
- 목적: 동일한 PPO League 학습 구조 하에서 10개의 독립적인 초기 시드로 각 20,000판을 학습하여
  학습 안정성 및 정책 분화의 재현성을 통계적으로 검증.
- 10개 초기 시드: [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
- 체크포인트: Step 0, Step 5k, Step 10k, Step 15k, Step 20k
- 기록 메트릭:
  AA, KK, AKs, JTs, 72o, 83o 정책, TV Distance, 벤치마크 수익률(bb/100), 올인 빈도, 폴드 빈도, 정책 엔트로피
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
from v48.poker_environment import (
    get_5p_holdem_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic, compute_ppo_losses
from v48.league_manager import LeagueManager, BaseBot
from v48.phase5_ablation_training import compute_entropy, LearnerBotWrapper
from v481.phase7_controlled_learning import compute_tv_distance
from v48.opponents_pool import get_8_opponents
from v483.hand_tensors import get_standard_hand_tensors

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
CHECKPOINTS = [0, 5000, 10000, 15000, 20000]
TOTAL_EPISODES = 20000
EVAL_HANDS = 500
EVAL_SEED_BASE = 77777

def evaluate_model_benchmark(
    model: HoldemActorCritic,
    num_hands: int = EVAL_HANDS,
    eval_seed: int = EVAL_SEED_BASE,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """고정된 평가 환경에서 8종 리그 봇을 상대로 500판 성적 측정"""
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = get_8_opponents()
    rng_eval = np.random.RandomState(eval_seed)
    rng_opp = random.Random(eval_seed + 1)
    
    chips_total = 0.0
    wins = 0
    allin_acts = 0
    fold_acts = 0
    total_acts = 0
    
    model.eval()
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        
        # 4명 상대 샘플링
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
                    m = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval.choice(legal, p=prs))
                    if act == 3: allin_acts += 1
                    elif act == 0: fold_acts += 1
                else:
                    ag_id = s2a[cur_seat]
                    opp_bot = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp_bot.act(st, cur_seat, legal)
                st.apply_action(act)
                
        ret = st.returns()[hero_seat]
        chips_total += ret
        if ret == max(st.returns()) and ret > 0: wins += 1
        
    bb100 = chips_total / float(num_hands)
    return {
        "bb100": bb100,
        "win_rate": (wins / float(num_hands)) * 100.0,
        "allin_pct": (allin_acts / max(1, total_acts)) * 100.0,
        "fold_pct": (fold_acts / max(1, total_acts)) * 100.0,
        "total_acts": total_acts
    }

def inspect_hand_policies(
    model: HoldemActorCritic,
    hand_tensors: Dict,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """표준 8개 핸드 상태에 대한 소프트맥스 정책 및 엔트로피 산출"""
    model.eval()
    policies = {}
    entropies = {}
    
    with torch.no_grad():
        for hname, (x, m, _) in hand_tensors.items():
            x = x.to(device)
            m = m.to(device)
            logits, _ = model(x, m)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            policies[hname] = probs
            entropies[hname] = compute_entropy(probs)
            
    tv_aa_72o = compute_tv_distance(policies['AA'], policies['72o'])
    return {
        "policies": policies,
        "entropies": entropies,
        "tv_aa_72o": tv_aa_72o
    }

def train_single_seed(seed_val: int, hand_tensors: Dict, device: torch.device = torch.device("cpu")):
    print(f"\n>>> [시드 {seed_val}] 독립 학습 시작 (20,000 에피소드)...")
    
    # 1. 시드 독립 초기화
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    rng_cards = np.random.RandomState(seed_val + 1000)
    rng_act = np.random.RandomState(seed_val + 2000)
    rng_league = random.Random(seed_val + 3000)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    league_mgr = LeagueManager(device=device)
    # 리그 매니저 내부 샘플링에 독립 rng 주입
    league_mgr.rng = rng_league
    
    model = HoldemActorCritic(253, 256, 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    learner_bot = LearnerBotWrapper(model, is_actor_critic=True, device=device)
    
    # 누적 감사 메트릭
    total_learner_decisions = 0
    total_transitions_collected = 0
    total_optimizer_updates = 0
    stale_data_count = 0
    
    batch_trajectories = []
    checkpoint_records = []
    
    t_start = time.time()
    
    # Step 0 체크포인트 기록
    p_info = inspect_hand_policies(model, hand_tensors, device)
    eval_res = evaluate_model_benchmark(model, num_hands=EVAL_HANDS, eval_seed=EVAL_SEED_BASE, device=device)
    torch.save(model.state_dict(), f"v483/models/seed_{seed_val}_step_0.pt")
    
    checkpoint_records.append({
        "seed": seed_val,
        "step": 0,
        "bb100": eval_res["bb100"],
        "win_rate": eval_res["win_rate"],
        "allin_pct": eval_res["allin_pct"],
        "fold_pct": eval_res["fold_pct"],
        "tv_aa_72o": p_info["tv_aa_72o"],
        "ent_aa": p_info["entropies"]["AA"],
        "ent_72o": p_info["entropies"]["72o"],
        "aa_fold": p_info["policies"]["AA"][0],
        "aa_call": p_info["policies"]["AA"][1],
        "aa_pot": p_info["policies"]["AA"][2],
        "aa_allin": p_info["policies"]["AA"][3],
        "kk_allin": p_info["policies"]["KK"][3],
        "aks_allin": p_info["policies"]["AKs"][3],
        "jts_allin": p_info["policies"]["JTs"][3],
        "72o_fold": p_info["policies"]["72o"][0],
        "72o_allin": p_info["policies"]["72o"][3],
        "83o_fold": p_info["policies"]["83o"][0],
        "83o_allin": p_info["policies"]["83o"][3],
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
                    m = torch.zeros((1, 4), dtype=torch.bool, device=device)
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
                    
        # 64 에피소드마다 On-Policy PPO 업데이트
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
                tot_l, _, _, _ = compute_ppo_losses(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv, ent_coef=0.005
                )
                optimizer.zero_grad()
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Stale data 방지를 위해 롤아웃 버퍼 즉시 초기화
            batch_trajectories = []
            
        if ep in CHECKPOINTS and ep > 0:
            torch.save(model.state_dict(), f"v483/models/seed_{seed_val}_step_{ep}.pt")
            p_info = inspect_hand_policies(model, hand_tensors, device)
            eval_res = evaluate_model_benchmark(model, num_hands=EVAL_HANDS, eval_seed=EVAL_SEED_BASE + ep, device=device)
            
            checkpoint_records.append({
                "seed": seed_val,
                "step": ep,
                "bb100": eval_res["bb100"],
                "win_rate": eval_res["win_rate"],
                "allin_pct": eval_res["allin_pct"],
                "fold_pct": eval_res["fold_pct"],
                "tv_aa_72o": p_info["tv_aa_72o"],
                "ent_aa": p_info["entropies"]["AA"],
                "ent_72o": p_info["entropies"]["72o"],
                "aa_fold": p_info["policies"]["AA"][0],
                "aa_call": p_info["policies"]["AA"][1],
                "aa_pot": p_info["policies"]["AA"][2],
                "aa_allin": p_info["policies"]["AA"][3],
                "kk_allin": p_info["policies"]["KK"][3],
                "aks_allin": p_info["policies"]["AKs"][3],
                "jts_allin": p_info["policies"]["JTs"][3],
                "72o_fold": p_info["policies"]["72o"][0],
                "72o_allin": p_info["policies"]["72o"][3],
                "83o_fold": p_info["policies"]["83o"][0],
                "83o_allin": p_info["policies"]["83o"][3],
            })
            print(f"  [Step {ep:5d}] TV Dist: {p_info['tv_aa_72o']:.4f} | AA Allin: {p_info['policies']['AA'][3]:.2f} | 72o Fold: {p_info['policies']['72o'][0]:.2f} | 벤치마크: {eval_res['bb100']:>+7.1f} bb/100")
            
    t_elapsed = time.time() - t_start
    print(f"  ✓ 시드 {seed_val} 학습 완료 ({t_elapsed:.1f}초 소요, 갱신 {total_optimizer_updates}회, 수집 {total_transitions_collected:,}개)")
    
    training_volume_record = {
        "seed": seed_val,
        "total_games": TOTAL_EPISODES,
        "learner_decisions": total_learner_decisions,
        "transitions_collected": total_transitions_collected,
        "optimizer_updates": total_optimizer_updates,
        "training_time_sec": round(t_elapsed, 2),
        "stale_data_count": stale_data_count
    }
    
    return checkpoint_records, training_volume_record

def run_phase1_main():
    print("=" * 85)
    print("🛸 [POKER AI V4.8.3] PHASE 1: 10개 독립 시드 PPO League 20k 학습 시작")
    print(f"   시드 목록: {SEEDS}")
    print(f"   체크포인트: {CHECKPOINTS}")
    print("=" * 85)
    
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors(device=device)
    
    all_checkpoint_records = []
    all_volume_records = []
    
    t_total_start = time.time()
    for s in SEEDS:
        ckpt_recs, vol_rec = train_single_seed(s, hand_tensors, device=device)
        all_checkpoint_records.extend(ckpt_recs)
        all_volume_records.append(vol_rec)
        
    t_total_elapsed = time.time() - t_total_start
    print("\n" + "=" * 85)
    print(f"🎉 10개 시드 전체 학습 완료! (총 {t_total_elapsed:.1f}초 소요)")
    print("=" * 85)
    
    # CSV 저장
    metrics_csv = "v483/data/phase1_seed_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_checkpoint_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_checkpoint_records)
    print(f"✓ 체크포인트 메트릭 저장: {metrics_csv}")
    
    volume_csv = "v483/data/phase3_training_volume.csv"
    with open(volume_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_volume_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_volume_records)
    print(f"✓ 훈련 데이터량 감사 저장: {volume_csv}")

if __name__ == "__main__":
    run_phase1_main()
