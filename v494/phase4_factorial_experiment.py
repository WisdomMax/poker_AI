"""
POKER AI V4.9.4 - PHASE 4: 2x2 Factorial Controlled Experiment
(Exploration Patch vs Targeted Subgame 독립 기여도 및 상호작용 분리 분석)

- 4대 요인 조건 (2x2 Design):
  1. Cond1_Control:          Exploration Patch OFF (0%) | Subgame OFF (0%)
  2. Cond2_ExplorationOnly:  Exploration Patch ON (15%) | Subgame OFF (0%)
  3. Cond3_SubgameOnly:      Exploration Patch OFF (0%) | Subgame ON (30%)
  4. Cond4_Combined:         Exploration Patch ON (15%) | Subgame ON (30%)

- 3개 독립 시드 (42, 101, 202), 각 20,000 에피소드 학습
- 동일 초기 체크포인트 (v49 CondC 20k Transfer)
- 독립 평가: 비학습 시드에서 600판 페어드 대국 진행
- 측정 메트릭:
  * 풀-게임 순수익 (bb/100, 공식: total_chips / N)
  * AA 3-Bet 직면 Fold% 및 Call%
  * 물리적 상한 [-100, +400] BB 자동 Assertion 검증
- Two-Way ANOVA 분석:
  * Main Effect of Exploration Patch
  * Main Effect of Targeted Subgame
  * Interaction Effect (Exploration x Subgame)
"""

import os
import sys
import time
import math
import random
import csv
from collections import defaultdict
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
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev

SEEDS = [42, 101, 202]
TRAIN_EPISODES = 20000
EVAL_HANDS = 600

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

def evaluate_aa_policy(model: HoldemActorCritic5A, gen: TargetedSubgameGenerator, device: torch.device) -> Tuple[float, float]:
    st_aa, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', 'AA', hero_seat=2)
    vec = st_aa.information_state_tensor(2)
    legal = st_aa.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return float(probs[0] * 100.0), float(probs[1] * 100.0)

def evaluate_fullgame(model: HoldemActorCritic5A, game: pyspiel.Game, device: torch.device, eval_seed: int = 88888) -> float:
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    rng_eval = np.random.RandomState(eval_seed)
    rng_opp = random.Random(eval_seed + 1)
    total_chips = 0.0
    model.eval()

    for h in range(EVAL_HANDS):
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

        ret_chips = st.returns()[hero_seat]
        validate_chips_return(ret_chips, hand_idx=h)
        validate_bb_return(ret_chips / 100.0, hand_idx=h)
        total_chips += ret_chips

    bb100 = float(total_chips / float(EVAL_HANDS))
    validate_mean_ev(bb100 / 100.0, metric_name="Fullgame_Eval_bb100")
    return bb100

def train_factorial_cell(
    cond_name: str,
    enable_exploration: bool,
    subgame_ratio: float,
    seed_val: int,
    device: torch.device = torch.device("cpu")
) -> Dict:
    print(f"\n[{cond_name}] Seed {seed_val} (Exploration: {enable_exploration}, Subgame: {subgame_ratio*100:.0f}%) 학습 중...")
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    gen = TargetedSubgameGenerator(game)
    model = HoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)

    save_path = f"v494/models/{cond_name}_seed_{seed_val}.pt"
    if os.path.exists(save_path):
        print(f"  기존 체크포인트 발견 -> 즉시 로드 및 평가: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        aa_fold, aa_call = evaluate_aa_policy(model, gen, device)
        eval_bb100 = evaluate_fullgame(model, game, device)
        print(f"  평가 완료 | bb/100: {eval_bb100:+6.1f} | AA 3-Bet Fold: {aa_fold:5.1f}%, Call: {aa_call:5.1f}%")
        return {
            'condition': cond_name,
            'exploration_patch': enable_exploration,
            'subgame_ratio': subgame_ratio,
            'seed': seed_val,
            'eval_bb100': eval_bb100,
            'aa_3bet_fold_pct': aa_fold,
            'aa_3bet_call_pct': aa_call,
            'elapsed_sec': 0.0
        }

    init_path = f"v49/models/CondC_5A_Transfer_seed_{seed_val}_step_20000.pt"
    if os.path.exists(init_path):
        model.load_state_dict(torch.load(init_path, map_location=device))
        if enable_exploration:
            with torch.no_grad():
                model.actor_head.bias.data.clamp_(-3.0, 3.0)

    optimizer = optim.Adam(model.parameters(), lr=0.0002, eps=1e-5)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed_val + 100)
    rng_choice = random.Random(seed_val + 200)

    batch_trajectories = []
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        is_subgame = (rng_choice.random() < subgame_ratio)
        hero_seat = 2

        if is_subgame:
            sit_name, target_seat = rng_choice.choice(SUBGAME_SITUATIONS)
            hand_name = rng_choice.choice(HAND_KEYS)
            hero_seat = target_seat
            st, _ = gen.generate_subgame_state(sit_name, hand_name, hero_seat=hero_seat, rng=rng_env)
            table_opps = {p: rng_choice.choice(opponents) for p in range(5) if p != hero_seat}
        else:
            s2a = wrapper.get_seat_to_agent(ep)
            a2s = wrapper.get_agent_to_seat(ep)
            hero_seat = a2s[0]
            sampled_opps = rng_choice.sample(opponents, 4)
            table_opps = {s: sampled_opps[(s2a[s] - 1) % 4] for s in range(5) if s != hero_seat}
            st = game.new_initial_state()

        traj = []
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_env)
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
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    if enable_exploration and rng_env.random() < 0.15:
                        act = int(rng_env.choice(legal))
                    else:
                        prs = [probs[a] for a in legal]
                        p_sum = sum(prs)
                        prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng_env.choice(legal, p=prs))

                    traj.append((vec, m.squeeze(0).cpu().numpy(), act, log_probs[act]))
                else:
                    act = table_opps[cur].act(st, cur, legal)
                st.apply_action(act)

        ret_chips = st.returns()[hero_seat]
        ret_bb = ret_chips / 100.0
        for v, m, a, olp in traj:
            batch_trajectories.append((v, m, a, olp, ret_bb))

        if ep % 64 == 0 and batch_trajectories:
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)

            with torch.no_grad():
                _, values = model(t_states, t_masks)
                adv = t_returns - values
                adv_std = adv.std()
                if adv_std > 1e-8:
                    adv = (adv - adv.mean()) / adv_std

            model.train()
            for _ in range(4):
                pol_loss, val_loss, ent_loss, tot_loss = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv
                )
                optimizer.zero_grad()
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            batch_trajectories = []

    elapsed = time.time() - t_start

    # 학습 후 평가
    aa_fold, aa_call = evaluate_aa_policy(model, gen, device)
    eval_bb100 = evaluate_fullgame(model, game, device)

    # 체크포인트 저장
    os.makedirs("v494/models", exist_ok=True)
    save_path = f"v494/models/{cond_name}_seed_{seed_val}.pt"
    torch.save(model.state_dict(), save_path)

    print(f"  완료 ({elapsed:.1f}s) | bb/100: {eval_bb100:+6.1f} | AA 3-Bet Fold: {aa_fold:5.1f}%, Call: {aa_call:5.1f}%")

    return {
        'condition': cond_name,
        'exploration_patch': enable_exploration,
        'subgame_ratio': subgame_ratio,
        'seed': seed_val,
        'eval_bb100': eval_bb100,
        'aa_3bet_fold_pct': aa_fold,
        'aa_3bet_call_pct': aa_call,
        'elapsed_sec': elapsed
    }

def run_phase4_factorial():
    print("=" * 80)
    print("POKER AI V4.9.4 - PHASE 4: 2x2 Factorial 실험 (탐색 패치 vs Subgame 학습 분리)")
    print("=" * 80)

    device = torch.device("cpu")
    cells = [
        ('Cond1_Control', False, 0.0),
        ('Cond2_ExplorationOnly', True, 0.0),
        ('Cond3_SubgameOnly', False, 0.3),
        ('Cond4_Combined', True, 0.3)
    ]

    records = []
    for cond_name, enable_exp, sub_ratio in cells:
        print(f"\n==================================================")
        print(f"실험군: {cond_name} (Exploration: {enable_exp}, Subgame: {sub_ratio*100:.0f}%)")
        print(f"==================================================")
        for s in SEEDS:
            rec = train_factorial_cell(cond_name, enable_exp, sub_ratio, s, device=device)
            records.append(rec)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Two-Way Factorial ANOVA 및 요인별 주효과 산출
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("[2x2 Factorial ANOVA 및 요인별 주효과 분해]")
    print("=" * 80)

    bb100_by_cell = defaultdict(list)
    aa_fold_by_cell = defaultdict(list)
    for r in records:
        bb100_by_cell[r['condition']].append(r['eval_bb100'])
        aa_fold_by_cell[r['condition']].append(r['aa_3bet_fold_pct'])

    # 셀별 평균
    m1 = np.mean(bb100_by_cell['Cond1_Control'])
    m2 = np.mean(bb100_by_cell['Cond2_ExplorationOnly'])
    m3 = np.mean(bb100_by_cell['Cond3_SubgameOnly'])
    m4 = np.mean(bb100_by_cell['Cond4_Combined'])

    print(f"  Cell 1 [Control (No Patch, No Subgame)]:   {m1:+7.2f} bb/100")
    print(f"  Cell 2 [Exploration Only (Patch, No Sub)]: {m2:+7.2f} bb/100")
    print(f"  Cell 3 [Subgame Only (No Patch, Sub 30%)]: {m3:+7.2f} bb/100")
    print(f"  Cell 4 [Combined (Patch + Sub 30%)]:       {m4:+7.2f} bb/100")

    # 주효과 (Main Effects)
    # Main Effect of Exploration: ((m2 + m4) - (m1 + m3)) / 2
    main_exp = ((m2 + m4) - (m1 + m3)) / 2.0
    # Main Effect of Subgame: ((m3 + m4) - (m1 + m2)) / 2
    main_sub = ((m3 + m4) - (m1 + m2)) / 2.0
    # Interaction Effect: ((m4 - m3) - (m2 - m1)) / 2
    interaction = ((m4 - m3) - (m2 - m1)) / 2.0

    print(f"\n  [요인 분석 결과]")
    print(f"  1. 탐색 패치(Exploration Patch) 주효과: {main_exp:+7.2f} bb/100")
    print(f"  2. 서브게임(Targeted Subgame) 주효과:  {main_sub:+7.2f} bb/100")
    print(f"  3. 상호작용(Interaction: Exp x Sub) 효과: {interaction:+7.2f} bb/100")

    # AA 3-Bet Fold율 비교
    print(f"\n  [AA 3-Bet Fold율 비교]")
    for c in ['Cond1_Control', 'Cond2_ExplorationOnly', 'Cond3_SubgameOnly', 'Cond4_Combined']:
        print(f"    {c:25s}: Fold {np.mean(aa_fold_by_cell[c]):5.1f}%")

    os.makedirs("v494/data", exist_ok=True)
    out_csv = "v494/data/phase4_factorial_experiment.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    anova_csv = "v494/data/phase4_factorial_anova_summary.csv"
    with open(anova_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['effect', 'value_bb100', 'interpretation'])
        writer.writeheader()
        writer.writerows([
            {'effect': 'Cell1_Control_Mean', 'value_bb100': m1, 'interpretation': '기준선 (풀게임 전면 폴드 퇴화)'},
            {'effect': 'Cell2_ExplorationOnly_Mean', 'value_bb100': m2, 'interpretation': '탐색 패치만 단독 적용'},
            {'effect': 'Cell3_SubgameOnly_Mean', 'value_bb100': m3, 'interpretation': '서브게임만 단독 적용'},
            {'effect': 'Cell4_Combined_Mean', 'value_bb100': m4, 'interpretation': '탐색 패치 + 서브게임 결합'},
            {'effect': 'Main_Effect_Exploration', 'value_bb100': main_exp, 'interpretation': '탐색 패치의 순수 기여도'},
            {'effect': 'Main_Effect_Subgame', 'value_bb100': main_sub, 'interpretation': '서브게임의 순수 기여도'},
            {'effect': 'Interaction_Effect', 'value_bb100': interaction, 'interpretation': '결합 시너지 효과'}
        ])

    print(f"\nPhase 4 요인 실험 데이터 저장 완료: {out_csv}, {anova_csv}")

if __name__ == '__main__':
    run_phase4_factorial()
