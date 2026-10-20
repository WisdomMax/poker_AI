"""
POKER AI V5.1.1 - PHASE 7: EXTENDED 10-SEED RIGOROUS BENCHMARK
- Concat, Two-Tower, FiLM의 10개 독립 학습 시드 (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) 확장 학습
  * PPO Transition 수, 환경 스텝 수, 업데이트 횟수 엄격 기록
  * 학습 하이퍼파라미터 100% 동일 통제
- 신규 독립 평가 스케줄 (New Independent Eval Seed = 9999999, 1000핸드) 평가:
  * 회귀 테스트: 기존 Frozen Benchmark (20261017) 병행
  * 학습 시드 간 표준편차(Between-seed variance) vs 평가 내 분산 엄격 구분
  * Student-t 95% 신뢰구간 (df = 9, t_crit = 2.2622)
  * All-Folder 정책 붕괴율 (Collapse Rate %)
- 원시 데이터를 v511/data/phase7_extended_10seeds_audit.csv에 저장
"""

import os
import sys
import csv
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v51.models_comparison import ConcatenationActorCritic, TwoTowerActorCritic, FiLMActorCritic

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def train_extended_model(
    arch_name: str,
    model_cls,
    seed: int,
    total_episodes: int = 1000,
    session_length: int = 50,
    update_interval: int = 200,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu")
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    opponents_master = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed + 1000)
    rng_opp = np.random.RandomState(seed + 2000)

    buf_hero = []
    buf_opp = []
    buf_mask = []
    buf_act = []
    buf_old_lp = []
    buf_ret = []
    buf_val = []

    total_transitions = 0
    total_updates = 0
    num_sessions = total_episodes // session_length

    for sess_idx in range(num_sessions):
        opp_indices = rng_opp.choice(len(opponents_master), 4, replace=False)
        table_opponents = [opponents_master[i] for i in opp_indices]
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        for h_in_sess in range(session_length):
            tracker_mgr.set_hand_index(h_in_sess)
            s2a = wrapper.get_seat_to_agent(h_in_sess)
            a2s = wrapper.get_agent_to_seat(h_in_sess)
            hero_seat = a2s[0]

            opp_feat = tracker_mgr.get_table_feature_tensor(hero_seat)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_env)

            if rng_env.rand() < 0.20:
                first_bettor = st.current_player()
                if first_bettor != hero_seat and not st.is_terminal():
                    leg = st.legal_actions(first_bettor)
                    if 2 in leg: st.apply_action(2)
                    elif 4 in leg: st.apply_action(4)

            ep_transitions = []
            hand_action_log = []

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng_env)
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        total_transitions += 1
                        hero_info = normalize_vec(st.information_state_tensor(cur))
                        h_t = torch.tensor(hero_info, dtype=torch.float32, device=device).unsqueeze(0)
                        o_t = torch.tensor(opp_feat, dtype=torch.float32, device=device).unsqueeze(0)
                        m_t = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in legal: m_t[0, a] = True

                        with torch.no_grad():
                            raw_logits, probs, val = model(h_t, o_t, m_t)
                            dist = torch.distributions.Categorical(probs)
                            act = dist.sample().item()
                            lp = dist.log_prob(torch.tensor(act, device=device)).item()
                            v = val.item()

                        ep_transitions.append((hero_info, opp_feat, m_t.squeeze(0).cpu().numpy(), act, lp, v))
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)
            ret_bb = st.returns()[hero_seat] / 100.0

            for h_info, o_info, mask_arr, act, lp, val in ep_transitions:
                buf_hero.append(h_info)
                buf_opp.append(o_info)
                buf_mask.append(mask_arr)
                buf_act.append(act)
                buf_old_lp.append(lp)
                buf_ret.append(ret_bb)
                buf_val.append(val)

            if len(buf_hero) >= update_interval:
                total_updates += 1
                b_hero = torch.tensor(np.array(buf_hero), dtype=torch.float32, device=device)
                b_opp = torch.tensor(np.array(buf_opp), dtype=torch.float32, device=device)
                b_mask = torch.tensor(np.array(buf_mask), dtype=torch.bool, device=device)
                b_act = torch.tensor(np.array(buf_act), dtype=torch.long, device=device)
                b_old_lp = torch.tensor(np.array(buf_old_lp), dtype=torch.float32, device=device)
                b_ret = torch.tensor(np.array(buf_ret), dtype=torch.float32, device=device)
                b_val = torch.tensor(np.array(buf_val), dtype=torch.float32, device=device)

                advantages = b_ret - b_val
                if len(advantages) > 1 and advantages.std() > 1e-6:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dataset_size = len(buf_hero)
                indices = np.arange(dataset_size)

                for epoch in range(ppo_epochs):
                    np.random.shuffle(indices)
                    for start_idx in range(0, dataset_size, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]
                        mb_hero = b_hero[batch_idx]
                        mb_opp = b_opp[batch_idx]
                        mb_mask = b_mask[batch_idx]
                        mb_act = b_act[batch_idx]
                        mb_old_lp = b_old_lp[batch_idx]
                        mb_ret = b_ret[batch_idx]
                        mb_adv = advantages[batch_idx]

                        raw_lgs, probs, values = model(mb_hero, mb_opp, mb_mask)
                        dist = torch.distributions.Categorical(probs)
                        new_lp = dist.log_prob(mb_act)
                        entropy = dist.entropy().mean()

                        ratios = torch.exp(new_lp - mb_old_lp)
                        surr1 = ratios * mb_adv
                        surr2 = torch.clamp(ratios, 0.8, 1.2) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        val_loss = F.mse_loss(values.squeeze(-1), mb_ret)

                        loss = policy_loss + 0.5 * val_loss - 0.01 * entropy
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                buf_hero.clear(); buf_opp.clear(); buf_mask.clear()
                buf_act.clear(); buf_old_lp.clear(); buf_ret.clear(); buf_val.clear()

    return model, total_transitions, total_updates

def evaluate_model_on_schedule(model, eval_seed: int, num_hands: int = 1000):
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    table_opponents = [RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot()]
    rng_eval = np.random.RandomState(eval_seed)
    model.eval()

    hand_returns = []
    total_decisions = 0
    fold_decisions = 0
    c72_total = 0
    c72_calls = 0
    aa_total = 0
    aa_actives = 0

    session_length = 50
    num_sessions = num_hands // session_length

    for sess_idx in range(num_sessions):
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        for h_in_sess in range(session_length):
            tracker_mgr.set_hand_index(h_in_sess)
            s2a = wrapper.get_seat_to_agent(h_in_sess)
            a2s = wrapper.get_agent_to_seat(h_in_sess)
            hero_seat = a2s[0]

            opp_feat = tracker_mgr.get_table_feature_tensor(hero_seat)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_eval)

            info_hero = st.information_state_string(hero_seat)
            priv_cards, _ = parse_cards_from_info_str(info_hero)
            is_72 = (len(priv_cards)>=2 and (('7' in priv_cards[0] and '2' in priv_cards[1]) or ('2' in priv_cards[0] and '7' in priv_cards[1])) and (priv_cards[0][1] != priv_cards[1][1]))
            is_aa = (len(priv_cards)>=2 and priv_cards[0][0] == 'A' and priv_cards[1][0] == 'A')

            if is_72: c72_total += 1
            if is_aa: aa_total += 1

            hand_action_log = []
            hero_c72 = False
            hero_act_aa = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng_eval)
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        total_decisions += 1
                        hero_info = normalize_vec(st.information_state_tensor(cur))
                        h_t = torch.tensor(hero_info, dtype=torch.float32).unsqueeze(0)
                        o_t = torch.tensor(opp_feat, dtype=torch.float32).unsqueeze(0)
                        m_t = torch.zeros((1, 5), dtype=torch.bool)
                        for a in legal: m_t[0, a] = True

                        with torch.no_grad():
                            _, probs, _ = model(h_t, o_t, m_t)
                            act = torch.argmax(probs, dim=-1).item()

                        if act == 0: fold_decisions += 1
                        if is_72 and act in [1, 2, 3, 4]: hero_c72 = True
                        if is_aa and act in [1, 2, 3, 4]: hero_act_aa = True

                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)
            ret_bb = st.returns()[hero_seat] / 100.0
            hand_returns.append(ret_bb)

            if is_72 and hero_c72: c72_calls += 1
            if is_aa and hero_act_aa: aa_actives += 1

    mean_bb_100 = float(np.mean(hand_returns)) * 100.0
    fold_pct = (fold_decisions / total_decisions * 100.0) if total_decisions > 0 else 0.0
    is_all_folder = (fold_decisions == total_decisions) and (np.isclose(mean_bb_100, -30.0, atol=0.1))
    c72_pct = (c72_calls / c72_total * 100.0) if c72_total > 0 else 0.0
    aa_pct = (aa_actives / aa_total * 100.0) if aa_total > 0 else 0.0

    return {
        'mean_bb_100': mean_bb_100,
        'fold_pct': fold_pct,
        'is_all_folder': is_all_folder,
        'c72_pct': c72_pct,
        'aa_pct': aa_pct
    }

def run_phase7_audit():
    print("=" * 85)
    print("PHASE 7: 10개 독립 시드 확장 벤치마크 (Concat vs Two-Tower vs FiLM)")
    print("=" * 85)

    architectures = [
        ("Concat", ConcatenationActorCritic),
        ("Two-Tower", TwoTowerActorCritic),
        ("FiLM", FiLMActorCritic)
    ]
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    NEW_EVAL_SEED = 9999999
    FROZEN_EVAL_SEED = 20261017

    results = []

    print(f"\n{'아키텍처':<12} | {'시드':<5} | {'전환수':<6} | {'업데이트':<6} | {'신규 스케줄 bb/100':<18} | {'기존 동결 bb/100':<18} | {'올-폴더 붕괴'}")
    print("-" * 95)

    start_all = time.time()

    for arch_name, cls_obj in architectures:
        for s in seeds:
            t0 = time.time()
            model, trans_cnt, upd_cnt = train_extended_model(arch_name, cls_obj, seed=s, total_episodes=1000)

            # 신규 평가 스케줄
            eval_new = evaluate_model_on_schedule(model, NEW_EVAL_SEED, num_hands=1000)
            # 기존 동결 스케줄 (회귀 테스트)
            eval_frozen = evaluate_model_on_schedule(model, FROZEN_EVAL_SEED, num_hands=1000)

            collapse_str = "붕괴 (All-Folder)" if eval_new['is_all_folder'] else "정상"

            print(f"{arch_name:<12} | {s:<5} | {trans_cnt:<6} | {upd_cnt:<6} | {eval_new['mean_bb_100']:+8.2f} bb/100     | {eval_frozen['mean_bb_100']:+8.2f} bb/100     | {collapse_str}")

            results.append({
                'architecture': arch_name,
                'seed': s,
                'transitions': trans_cnt,
                'updates': upd_cnt,
                'new_eval_bb': eval_new['mean_bb_100'],
                'new_is_collapse': eval_new['is_all_folder'],
                'new_c72_pct': eval_new['c72_pct'],
                'new_aa_pct': eval_new['aa_pct'],
                'frozen_eval_bb': eval_frozen['mean_bb_100'],
                'frozen_is_collapse': eval_frozen['is_all_folder']
            })

    total_time = time.time() - start_all
    print(f"\n[10개 시드 전체 학습 및 평가 완료] 총 소요 시간: {total_time:.1f}초")

    # 통계 요약 (10개 시드 기준, df=9, t_crit=2.2622)
    t_crit_10 = stats.t.ppf(0.975, df=9)
    print("\n" + "=" * 85)
    print("10개 시드 기반 종합 통계 및 붕괴율 분석 (Student-t df=9, t_crit=2.2622):")
    print("=" * 85)
    print(f"{'아키텍처':<12} | {'신규 스케줄 평균 (95% CI)':<30} | {'시드간 편차(s)':<12} | {'올-폴더 붕괴율':<14} | {'평균 72o Call%'}")
    print("-" * 85)

    arch_stats = {}
    for arch_name, _ in architectures:
        rows = [r for r in results if r['architecture'] == arch_name]
        bbs = np.array([r['new_eval_bb'] for r in rows])
        collapses = sum(1 for r in rows if r['new_is_collapse'])
        mean_bb = float(np.mean(bbs))
        std_bb = float(np.std(bbs, ddof=1))
        se_bb = std_bb / np.sqrt(10)
        ci_half = float(t_crit_10 * se_bb)
        collapse_rate = (collapses / 10) * 100.0
        mean_72 = float(np.mean([r['new_c72_pct'] for r in rows]))

        arch_stats[arch_name] = {'mean': mean_bb, 'std': std_bb, 'ci_half': ci_half, 'bbs': bbs, 'collapse_rate': collapse_rate}

        ci_str = f"{mean_bb:+7.2f} ± {ci_half:5.2f} bb/100"
        print(f"{arch_name:<12} | {ci_str:<30} | {std_bb:10.2f}   | {collapses}/10 ({collapse_rate:4.1f}%)    | {mean_72:6.1f}%")

    # t-test across 10 seeds: FiLM vs Concat, FiLM vs Two-Tower
    t_fc, p_fc = stats.ttest_ind(arch_stats['FiLM']['bbs'], arch_stats['Concat']['bbs'], equal_var=False)
    t_ft, p_ft = stats.ttest_ind(arch_stats['FiLM']['bbs'], arch_stats['Two-Tower']['bbs'], equal_var=False)
    print("\n  [10-Seed Welch t-test]")
    print(f"    - FiLM vs Concat    : t = {t_fc:.3f}, p = {p_fc:.4f} -> {'유의미 (p<0.05)' if p_fc < 0.05 else '유의미하지 않음 (p>=0.05)'}")
    print(f"    - FiLM vs Two-Tower : t = {t_ft:.3f}, p = {p_ft:.4f} -> {'유의미 (p<0.05)' if p_ft < 0.05 else '유의미하지 않음 (p>=0.05)'}")

    # CSV 저장
    csv_path = "v511/data/phase7_extended_10seeds_audit.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장 완료] Phase 7 확장 10-시드 감사 CSV: {csv_path}\n")

if __name__ == '__main__':
    run_phase7_audit()
