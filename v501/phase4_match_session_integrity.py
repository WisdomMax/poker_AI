"""
POKER AI V5.0.1 - PHASE 4: Match Session & Seat Rotation Simulation Integrity Audit
- 1. OpenSpiel universal_poker 물리적 좌석(0:SB, 1:BB, 2:UTG, 3:CO, 4:BTN)과 Agent ID 매핑 검증
- 2. 50핸드 실제 Match Session 시뮬레이션:
     * Agent 1: TightBot
     * Agent 2: AggressiveBot
     * Agent 3: AlwaysCallBot
     * Agent 4: RandomBot
     각 에이전트의 물리적 좌석 이동 추적 및 통계 격리 보존 검증 (오염 0건)
- 3. 기회(Opportunity) 중복 집계 방지 검증 (단일 핸드 내 VPIP 기회 = 정확히 1회)
"""

import os
import sys
import csv
from collections import defaultdict
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v48.opponents_pool import (
    TightBot, AggressiveBot, AlwaysCallBot, RandomBot
)
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager

def run_phase4_audit():
    print("=" * 85)
    print("PHASE 4: MATCH SESSION 및 좌석 회전 시뮬레이터 무결성 전수 감사")
    print("=" * 85)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    # 1. 좌석 순환 매핑 수학적 검증
    print("\n[검사 1] OpenSpiel 좌석 매핑 공식 일치성 검증 (0~9핸드):")
    pos_names = ["SB(0)", "BB(1)", "UTG(2)", "CO(3)", "BTN(4)"]
    for h in range(5):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        print(f"  Hand {h}: Hero Seat = {pos_names[hero_seat]:<8} | Seat->Agent: {s2a}")
        # 역변환 검증: a2s[s2a[seat]] == seat
        for s in range(5):
            assert a2s[s2a[s]] == s, "좌석-에이전트 역변환 실패!"
    print("-> PASS: 좌석-에이전트 1:1 전단사(Bijective) 매핑 및 5인 순환 무결성 확인.")

    # 2. 50핸드 실제 Match Session 시뮬레이션 및 에이전트별 통계 격리 검증
    print("\n[검사 2] 50핸드 Match Session 시뮬레이션 및 에이전트별 통계 격리 검증:")
    opp_dict = {
        1: TightBot(),
        2: AggressiveBot(),
        3: AlwaysCallBot(),
        4: RandomBot()
    }

    mgr = MatchSessionOpponentManager()
    mgr.reset_session()

    rng = np.random.RandomState(100)

    # 실제 각 에이전트가 행한 올인/레이즈 횟수를 외부 독립 카운터로 기록
    external_allin_counter = defaultdict(int)
    external_raise_counter = defaultdict(int)
    external_vpip_opp_counter = defaultdict(int)

    for h in range(50):
        mgr.set_hand_index(h)
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]

        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng)
        hand_actions = []

        # 단일 핸드 내 VPIP 기회 중복 방지 체크용
        vpip_recorded_in_hand = set()

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                if cur == hero_seat:
                    act = 1 if 1 in legal else legal[0]
                else:
                    ag_id = s2a[cur]
                    opp = opp_dict[ag_id]
                    act = opp.act(st, cur, legal)

                    if act == 3: # All-in
                        external_allin_counter[ag_id] += 1
                    if act in [2, 3, 4]:
                        external_raise_counter[ag_id] += 1

                    if round_idx == 0 and ag_id not in vpip_recorded_in_hand:
                        external_vpip_opp_counter[ag_id] += 1
                        vpip_recorded_in_hand.add(ag_id)

                hand_actions.append((cur, round_idx, legal, act))
                st.apply_action(act)

        mgr.parse_and_update_hand(hand_actions, s2a)

    print("  외부 독립 카운터 vs Tracker Manager 내부 집계 1:1 대조:")
    for ag_id in range(1, 5):
        t = mgr.trackers[ag_id]
        trk_allin = t.stats['allin']['succ']
        ext_allin = external_allin_counter[ag_id]
        trk_vpip_opp = t.stats['vpip']['opp']
        ext_vpip_opp = external_vpip_opp_counter[ag_id]

        opp_name = opp_dict[ag_id].name
        print(f"    Agent {ag_id} ({opp_name:<11}): All-in [실제: {ext_allin:2d} vs 추적: {trk_allin:2d}] | VPIP기회 [실제: {ext_vpip_opp:2d} vs 추적: {trk_vpip_opp:2d}]")

        assert trk_allin == ext_allin, f"Agent {ag_id} 올인 횟수 불일치! {trk_allin} != {ext_allin}"
        assert trk_vpip_opp == ext_vpip_opp, f"Agent {ag_id} VPIP 기회 수 불일치! {trk_vpip_opp} != {ext_vpip_opp}"

    print("-> PASS: 50핸드 동안 좌석 회전에도 불구하고 4명의 상대 통계가 외부 실측과 100% 비트 단위 일치.")

    # 3. AggressiveBot(Agent 2) vs TightBot(Agent 1) 통계 명확한 분화 확인
    t1 = mgr.trackers[1]
    t2 = mgr.trackers[2]
    pfr_tight, _, _ = t1.get_posterior('pfr')
    pfr_aggr, _, _ = t2.get_posterior('pfr')
    print(f"\n[검사 3] 상대 성향 감지 검증:")
    print(f"  Agent 1 (TightBot)      PFR 사후평균: {pfr_tight:.4f}")
    print(f"  Agent 2 (AggressiveBot) PFR 사후평균: {pfr_aggr:.4f}")
    assert pfr_aggr > pfr_tight, "Aggressive 봇의 PFR이 Tight 봇보다 커야 합니다!"
    print("-> PASS: AggressiveBot의 공격성이 TightBot 대비 명확하게 높게 추정됨.")

    # CSV 저장
    csv_out = "v501/data/phase4_match_session_integrity.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['agent_id', 'bot_name', 'external_allin', 'tracked_allin', 'vpip_opp', 'pfr_mean'])
        for ag_id in range(1, 5):
            t = mgr.trackers[ag_id]
            pfr_m, _, _ = t.get_posterior('pfr')
            w.writerow([ag_id, opp_dict[ag_id].name, external_allin_counter[ag_id], t.stats['allin']['succ'], t.stats['vpip']['opp'], pfr_m])
    print(f"\n[저장 완료] 세션 무결성 CSV: {csv_out}")

if __name__ == "__main__":
    run_phase4_audit()
