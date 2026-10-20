"""
POKER AI V4.9.1 - PHASE 2: Half-Pot 효과 순수 분리 대조 실험 (A/B Controlled Isolation Experiment)
- 목적: 265차원 동일 아키텍처에서 Half-Pot 사용 가능 여부만을 유일한 변인으로 통제하여,
  Half-Pot 행동 확장이 순수하게 기대수익과 전략에 미치는 인과적 효과를 독립 검증.
- 조건 2종:
  1. Control:   5-Action 신경망 (265차원), Half-Pot(행동 4) 영구 마스킹 (4-Action만 선택 가능)
  2. Treatment: 동일 5-Action 신경망 (265차원), Half-Pot(행동 4) 완전 허용 (5-Action 전체 선택 가능)
- 통제 변인 (완전 동일 유지):
  - 5개 초기 시드 [42, 101, 202, 303, 404]
  - 시드별 동일 초기 가중치 (torch.manual_seed)
  - 동일 PPO 하이퍼파라미터 (lr=5e-4, batch=64, 4 epochs, clip=0.2, ent=0.005)
  - 동일 리그 상대 풀 및 샘플링 난수
  - 동일 카드 분배 및 좌석 스케줄
  - 동일 평가 시드 및 평가 환경
- 산출:
  - 20k 체크포인트 수익률(bb/100)
  - 시드별 페어드 차이 (Δ = Treatment - Control)
  - 대응표본 t-검정 및 95% 신뢰구간
"""

import sys
import os
import time
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, ACTION_NAMES_5A
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A, compute_ppo_losses_5a
from v49.league_manager_5a import LeagueManager5A, LearnerBotWrapper5A
from v481.phase7_controlled_learning import compute_tv_distance
from v49.hand_tensors_5a import get_standard_hand_tensors_5a
from v49.phase6_independent_eval import get_9_opponents_dict

SEEDS = [42, 101, 202, 303, 404]
CHECKPOINTS = [5000, 10000, 20000]
TOTAL_EPISODES = 20000
EVAL_HANDS = 500
EVAL_SEED_BASE = 66666

def evaluate_model_isolated(
    model: HoldemActorCritic5A,
    allow_halfpot: bool,
    num_hands: int = EVAL_HANDS,
    eval_seed: int = EVAL_SEED_BASE,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents_dict = get_9_opponents_dict()
    opponents = list(opponents_dict.values())
    
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
                    for a in legal:
                        if a == 4 and not allow_halfpot:
                            continue # Control 조건: Half-Pot 마스킹
                        m[0, a] = True
                        
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
                
        chips_total += st.returns()[hero_seat]
        if st.returns()[hero_seat] > 0 and st.returns()[hero_seat] == max(st.returns()):
            wins += 1
            
    bb100 = chips_total / float(num_hands)
    return {
        "bb100": bb100,
        "win_rate": (wins / float(num_hands)) * 100.0,
        "halfpot_pct": (action_counts[4] / max(1, total_acts)) * 100.0,
        "allin_pct": (action_counts[3] / max(1, total_acts)) * 100.0,
        "fold_pct": (action_counts[0] / max(1, total_acts)) * 100.0
    }

def train_isolated_condition(
    seed_val: int,
    allow_halfpot: bool,
    cond_label: str,
    hand_tensors: Dict,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[Dict], Dict]:
    # 완전 동일 초기화
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
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    learner_bot = LearnerBotWrapper5A(model, device=device)
    
    total_learner_decisions = 0
    total_transitions_collected = 0
    total_optimizer_updates = 0
    
    batch_trajectories = []
    checkpoint_records = []
    t_start = time.time()
    
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
                    for a in legal:
                        if a == 4 and not allow_halfpot:
                            continue # Control: 마스킹
                        m[0, a] = True
                        
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
                    
        # 64 에피소드마다 갱신
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
            
        if ep in CHECKPOINTS:
            eval_res = evaluate_model_isolated(model, allow_halfpot=allow_halfpot, eval_seed=EVAL_SEED_BASE + ep, device=device)
            torch.save(model.state_dict(), f"v491/models/{cond_label}_seed_{seed_val}_step_{ep}.pt")
            checkpoint_records.append({
                "condition": cond_label,
                "seed": seed_val,
                "step": ep,
                "bb100": eval_res["bb100"],
                "win_rate": eval_res["win_rate"],
                "halfpot_pct": eval_res["halfpot_pct"],
                "allin_pct": eval_res["allin_pct"],
                "fold_pct": eval_res["fold_pct"]
            })
            
    t_elapsed = time.time() - t_start
    vol_rec = {
        "condition": cond_label,
        "seed": seed_val,
        "decisions": total_learner_decisions,
        "transitions": total_transitions_collected,
        "updates": total_optimizer_updates,
        "time_sec": round(t_elapsed, 2)
    }
    return checkpoint_records, vol_rec

def run_phase2_experiment():
    print("=" * 85)
    print("🛸 [V4.9.1 PHASE 2] Half-Pot 효과 순수 분리 대조 실험 (Control vs Treatment)")
    print(f"   시드 목록: {SEEDS} (각 20,000판)")
    print("=" * 85)
    
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors_5a(device=device)
    
    all_ckpts = []
    all_vols = []
    
    # 1. Control 그룹 (Half-Pot 마스킹)
    print("\n>>> [1/2] Control 그룹 학습 시작 (Half-Pot 마스킹)...")
    for s in SEEDS:
        ckpts, vol = train_isolated_condition(s, allow_halfpot=False, cond_label="Control_Masked", hand_tensors=hand_tensors, device=device)
        all_ckpts.extend(ckpts)
        all_vols.append(vol)
        last = ckpts[-1]
        print(f"  ✓ Control Seed {s}: 20k 수익률 = {last['bb100']:>+7.1f} bb/100 | HalfPot = {last['halfpot_pct']:.1f}%")
        
    # 2. Treatment 그룹 (Half-Pot 허용)
    print("\n>>> [2/2] Treatment 그룹 학습 시작 (Half-Pot 허용)...")
    for s in SEEDS:
        ckpts, vol = train_isolated_condition(s, allow_halfpot=True, cond_label="Treatment_Allowed", hand_tensors=hand_tensors, device=device)
        all_ckpts.extend(ckpts)
        all_vols.append(vol)
        last = ckpts[-1]
        print(f"  ✓ Treatment Seed {s}: 20k 수익률 = {last['bb100']:>+7.1f} bb/100 | HalfPot = {last['halfpot_pct']:.1f}%")

    # CSV 저장
    csv_ckpts = "v491/data/phase2_isolation_metrics.csv"
    with open(csv_ckpts, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ckpts[0].keys()))
        writer.writeheader()
        writer.writerows(all_ckpts)
        
    # -------------------------------------------------------------
    # 3. 페어드 통계 분석 (대응표본 t-검정)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("📊 [PHASE 2 페어드 분석] Control vs Treatment 20k 최종 성적 비교 (N=5)")
    print("=" * 85)
    print(f"{'시드':<8} | {'Control (Masked)':<20} | {'Treatment (Allowed)':<20} | {'차이 (Δ = Treat - Ctrl)'}")
    print("-" * 85)
    
    ctrl_20k = [r["bb100"] for r in all_ckpts if r["condition"] == "Control_Masked" and r["step"] == 20000]
    treat_20k = [r["bb100"] for r in all_ckpts if r["condition"] == "Treatment_Allowed" and r["step"] == 20000]
    diffs = np.array(treat_20k) - np.array(ctrl_20k)
    
    for s, c_val, t_val, d_val in zip(SEEDS, ctrl_20k, treat_20k, diffs):
        print(f"Seed {s:<4d} | {c_val:>+10.1f} bb/100       | {t_val:>+10.1f} bb/100         | {d_val:>+10.1f} bb/100")
        
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(len(diffs))
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
    p_val = float(stats.t.sf(abs(t_stat), df=len(diffs)-1) * 2) # 양측 검정
    ci_diff = (mean_diff - stats.t.ppf(0.975, df=len(diffs)-1)*se_diff, mean_diff + stats.t.ppf(0.975, df=len(diffs)-1)*se_diff)
    
    print("-" * 85)
    print(f"평균 수익률: Control = {np.mean(ctrl_20k):>+7.1f} bb/100 vs Treatment = {np.mean(treat_20k):>+7.1f} bb/100")
    print(f"평균 우위 (Δ):          {mean_diff:>+7.2f} bb/100 (표본 표준편차: {std_diff:.2f})")
    print(f"차이의 95% 신뢰구간:     [{ci_diff[0]:>+7.2f}, {ci_diff[1]:>+7.2f}] bb/100")
    print(f"Paired t-통계량:         t = {t_stat:.3f} (df={len(diffs)-1}), p-value = {p_val:.4f}")
    
    if p_val < 0.05 and mean_diff > 0:
        print("=> 🏆 판정: Half-Pot 행동 허용이 통계적으로 유의미한(p < 0.05) 성능 향상을 견인함!")
    else:
        print("=> ⚠️ 판정: Half-Pot 허용에 따른 통계적으로 유의미한 우위가 입증되지 않음 (귀무가설 채택).")
        print("   신뢰구간이 0을 포함하며 p >= 0.05 수준으로, 베팅 크기 확장이 자동적인 실력 향상을 보장하지 않음.")
    print("=" * 85)

if __name__ == "__main__":
    run_phase2_experiment()
