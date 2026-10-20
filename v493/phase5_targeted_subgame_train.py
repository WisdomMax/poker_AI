"""
POKER AI V4.9.3 - PHASE 4 & 5: Targeted Subgame 혼합 학습 및 통제된 비교 실험
- 조건 비교:
  Condition A: Full-Game 100% (Subgame 0%)
  Condition B: Full-Game 90% + Targeted Subgame 10%
  Condition C: Full-Game 70% + Targeted Subgame 30%
  Condition D: Full-Game 50% + Targeted Subgame 50%
- 독립 시드: [42, 101, 202]
- 학습량: 40,000 에피소드 (체크포인트: 0, 10k, 20k, 40k)
- Subgame 무결성:
  * OpenSpiel 합법 상태에서 시작하여 온-폴리시 정책으로 정상 터미널 롤아웃
  * Old Log Prob 실시간 기록, 오래된 궤적 재사용 배제
  * 풀-게임 데이터와 서브게임 데이터의 출처별 카운팅
- 상세 정책 메트릭 추적:
  * AA 3-Bet Fold%, Call%, All-in%
  * 72o 3-Bet Fold%
  * JTs 3-Bet Fold%, Call%
  * Flop Defense%, Turn Defense%, River Defense%
  * Full-Game 벤치마크 bb/100 (정확한 공식)
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
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot, RuleBasedBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator, TARGET_HOLE_CARDS

SEEDS = [42, 101, 202]
TOTAL_EPISODES = 40000
CHECKPOINTS = [0, 10000, 20000, 40000]

SUBGAME_SITUATIONS = [
    ('SitA_Facing_OpenRaise', 3),
    ('SitB_Facing_3Bet_AllIn', 2),
    ('SitB_Facing_3Bet_Pot', 2),
    ('SitC_Facing_4Bet', 3),
    ('SitD_Flop_Facing_Bet', 2),
    ('SitE_Turn_Facing_Raise', 2),
    ('SitF_River_Facing_AllIn', 2)
]
HAND_KEYS = list(TARGET_HOLE_CARDS.keys())

def evaluate_subgame_policies(model: HoldemActorCritic5A, gen: TargetedSubgameGenerator, device: torch.device) -> Dict[str, float]:
    """6대 핵심 상황 및 핸드에 대한 정책 확률 정밀 측정"""
    model.eval()
    results = {}
    rng_eval = np.random.RandomState(49300)

    # 1. AA Facing 3-Bet All-in
    st_aa_3b, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', 'AA', hero_seat=2, rng=rng_eval)
    vec = st_aa_3b.information_state_tensor(2)
    legal = st_aa_3b.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['aa_3bet_fold_pct'] = float(probs[0]) * 100.0
    results['aa_3bet_call_pct'] = float(probs[1]) * 100.0
    results['aa_3bet_allin_pct'] = float(probs[3]) * 100.0 if 3 in legal else 0.0

    # 2. 72o Facing 3-Bet All-in
    st_72_3b, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', '72o', hero_seat=2, rng=rng_eval)
    vec = st_72_3b.information_state_tensor(2)
    legal = st_72_3b.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['72o_3bet_fold_pct'] = float(probs[0]) * 100.0

    # 3. JTs Facing 3-Bet Pot
    st_jt_3b, _ = gen.generate_subgame_state('SitB_Facing_3Bet_Pot', 'JTs', hero_seat=2, rng=rng_eval)
    vec = st_jt_3b.information_state_tensor(2)
    legal = st_jt_3b.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['jts_3bet_fold_pct'] = float(probs[0]) * 100.0
    results['jts_3bet_call_pct'] = float(probs[1]) * 100.0

    # 4. Flop Defense% (SitD에서 Fold(0)가 아닌 Call/Raise를 선택할 확률)
    st_flop, _ = gen.generate_subgame_state('SitD_Flop_Facing_Bet', 'AA', hero_seat=2, rng=rng_eval)
    vec = st_flop.information_state_tensor(2)
    legal = st_flop.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['flop_defense_pct'] = float(1.0 - probs[0]) * 100.0

    # 5. Turn Defense% (SitE)
    st_turn, _ = gen.generate_subgame_state('SitE_Turn_Facing_Raise', 'AA', hero_seat=2, rng=rng_eval)
    vec = st_turn.information_state_tensor(2)
    legal = st_turn.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['turn_defense_pct'] = float(1.0 - probs[0]) * 100.0

    # 6. River Defense% (SitF)
    st_riv, _ = gen.generate_subgame_state('SitF_River_Facing_AllIn', 'AA', hero_seat=2, rng=rng_eval)
    vec = st_riv.information_state_tensor(2)
    legal = st_riv.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results['river_defense_pct'] = float(1.0 - probs[0]) * 100.0

    return results

def evaluate_fullgame_benchmark(model: HoldemActorCritic5A, game: pyspiel.Game, device: torch.device, num_hands: int = 300) -> float:
    """원래 풀-게임 환경에서 8종 봇 상대 bb/100 측정 (정확한 공식)"""
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    rng_eval = np.random.RandomState(88888)
    rng_opp = random.Random(77777)
    
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

    # 정확한 bb/100: total_chips / num_hands
    return float(total_chips / float(num_hands))

def train_condition(
    condition_name: str,
    subgame_ratio: float,
    seed_val: int,
    device: torch.device = torch.device("cpu")
) -> List[Dict]:
    print(f"\n[{condition_name}] Seed {seed_val} (Subgame Ratio: {subgame_ratio*100:.0f}%) 학습 시작...")
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    gen = TargetedSubgameGenerator(game)

    # V4.9 20k Transfer 가중치에서 안전하게 출발 (초기 지식 보존)
    model = HoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    init_path = f"v49/models/CondC_5A_Transfer_seed_{seed_val}_step_20000.pt"
    if os.path.exists(init_path):
        model.load_state_dict(torch.load(init_path, map_location=device))
        # 과포화된 Fold 바이어스(+37) 완화: 극단적 포화를 해소하여 탐색 활성화
        with torch.no_grad():
            model.actor_head.bias.data.clamp_(-3.0, 3.0)
        print(f"  V4.9 20k 전이 가중치 로드 완료 (바이어스 클램핑 적용): {init_path}")
    else:
        print(f"  경고: 전이 경로 없음, 직교 초기화 사용")

    optimizer = optim.Adam(model.parameters(), lr=0.0002, eps=1e-5)

    # 봇 풀
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    total_learner_decisions = 0
    total_transitions_collected = 0
    total_optimizer_updates = 0
    fullgame_episodes = 0
    subgame_episodes = 0

    batch_trajectories = []
    records = []

    rng_env = np.random.RandomState(seed_val + 1000)
    rng_choice = random.Random(seed_val + 2000)

    t_start = time.time()

    # Step 0 초기 평가
    pol_0 = evaluate_subgame_policies(model, gen, device)
    bb100_0 = evaluate_fullgame_benchmark(model, game, device, num_hands=200)

    save_path_0 = f"v493/models/{condition_name}_seed_{seed_val}_step_0.pt"
    os.makedirs("v493/models", exist_ok=True)
    torch.save(model.state_dict(), save_path_0)

    rec_0 = {
        'condition': condition_name,
        'subgame_ratio': subgame_ratio,
        'seed': seed_val,
        'step': 0,
        'fullgame_bb100': bb100_0,
        'fullgame_episodes': 0,
        'subgame_episodes': 0,
        'total_decisions': 0,
        'total_transitions': 0,
        'total_updates': 0,
        'elapsed_sec': 0.0
    }
    rec_0.update(pol_0)
    records.append(rec_0)

    print(f"  Step 0 | bb/100: {bb100_0:+6.1f} | AA 3-Bet Fold: {pol_0['aa_3bet_fold_pct']:5.1f}%, Call: {pol_0['aa_3bet_call_pct']:5.1f}% | Flop Def: {pol_0['flop_defense_pct']:5.1f}%")

    for ep in range(1, TOTAL_EPISODES + 1):
        is_subgame = (rng_choice.random() < subgame_ratio)
        hero_seat = 2

        if is_subgame:
            subgame_episodes += 1
            # 6대 희귀 상황 및 8종 핸드 무작위 샘플링
            sit_name, target_seat = rng_choice.choice(SUBGAME_SITUATIONS)
            hand_name = rng_choice.choice(HAND_KEYS)
            hero_seat = target_seat
            st, _ = gen.generate_subgame_state(sit_name, hand_name, hero_seat=hero_seat, rng=rng_env)
            # 상대 봇 배치
            table_opps = {p: rng_choice.choice(opponents) for p in range(5) if p != hero_seat}
        else:
            fullgame_episodes += 1
            s2a = wrapper.get_seat_to_agent(ep)
            a2s = wrapper.get_agent_to_seat(ep)
            hero_seat = a2s[0]
            sampled_opps = rng_choice.sample(opponents, 4)
            table_opps = {s: sampled_opps[(s2a[s] - 1) % 4] for s in range(5) if s != hero_seat}
            st = game.new_initial_state()

        traj = []

        # 정상 터미널까지 롤아웃 (온-폴리시)
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_env)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    total_learner_decisions += 1
                    vec = st.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True

                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    # 탐색 붕괴(Exploration Extinction) 극복: 에psilon 탐색 (15%) 주입
                    # 온-폴리시 중요도 샘플링을 위해 실제 선택된 행동의 log_prob를 기록
                    if rng_env.random() < 0.15:
                        act = int(rng_env.choice(legal))
                    else:
                        prs = [probs[a] for a in legal]
                        p_sum = sum(prs)
                        prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng_env.choice(legal, p=prs))
                        
                    traj.append((vec, m.squeeze(0).cpu().numpy(), act, log_probs[act]))
                else:
                    opp = table_opps[cur]
                    act = opp.act(st, cur, legal)

                st.apply_action(act)

        # 리턴 기록
        ret_chips = st.returns()[hero_seat]
        ret_bb = ret_chips / 100.0 # 100 chips = 1 bb
        for v, m, a, olp in traj:
            batch_trajectories.append((v, m, a, olp, ret_bb))
            total_transitions_collected += 1

        # 64 에피소드마다 On-policy PPO 갱신
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
            for _ in range(4):
                pol_loss, val_loss, ent_loss, tot_loss = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, advantages
                )
                optimizer.zero_grad()
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            batch_trajectories = []

        # 체크포인트 평가
        if ep in [10000, 20000, 40000]:
            elapsed = time.time() - t_start
            pol = evaluate_subgame_policies(model, gen, device)
            bb100_val = evaluate_fullgame_benchmark(model, game, device, num_hands=300)

            save_path = f"v493/models/{condition_name}_seed_{seed_val}_step_{ep}.pt"
            torch.save(model.state_dict(), save_path)

            print(f"  Step {ep:5d} | bb/100: {bb100_val:+6.1f} | AA 3-Bet Fold: {pol['aa_3bet_fold_pct']:5.1f}%, Call: {pol['aa_3bet_call_pct']:5.1f}% | Flop Def: {pol['flop_defense_pct']:5.1f}% | Subgame: {subgame_episodes:,} / {ep:,} | Elapsed: {elapsed:5.1f}s")

            rec = {
                'condition': condition_name,
                'subgame_ratio': subgame_ratio,
                'seed': seed_val,
                'step': ep,
                'fullgame_bb100': bb100_val,
                'fullgame_episodes': fullgame_episodes,
                'subgame_episodes': subgame_episodes,
                'total_decisions': total_learner_decisions,
                'total_transitions': total_transitions_collected,
                'total_updates': total_optimizer_updates,
                'elapsed_sec': elapsed
            }
            rec.update(pol)
            records.append(rec)

    return records

def run_phase5_experiment():
    print("=" * 80)
    print("POKER AI V4.9.3 - PHASE 4 & 5: 통제된 Targeted Subgame 혼합 학습 실험")
    print("=" * 80)

    device = torch.device("cpu")
    all_records = []

    conditions = [
        ('ConditionA_Subgame0', 0.00),
        ('ConditionB_Subgame10', 0.10),
        ('ConditionC_Subgame30', 0.30),
        ('ConditionD_Subgame50', 0.50)
    ]

    for cond_name, ratio in conditions:
        print(f"\n==================================================")
        print(f"실험 조건: {cond_name} (Subgame Ratio: {ratio*100:.0f}%)")
        print(f"==================================================")
        for s in SEEDS:
            recs = train_condition(cond_name, ratio, s, device=device)
            all_records.extend(recs)

    os.makedirs("v493/data", exist_ok=True)
    out_csv = "v493/data/phase5_subgame_training_metrics.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nPhase 4 & 5 학습 실험 전체 데이터 저장 완료: {out_csv} (총 {len(all_records)} 행)")

if __name__ == '__main__':
    run_phase5_experiment()
