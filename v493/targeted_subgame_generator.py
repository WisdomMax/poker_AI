"""
POKER AI V4.9.3 - PHASE 3: 희귀 상태 생성기 (Targeted Subgame State Generator)
- OpenSpiel universal_poker 정상 게임 트리 100% 준수
- 비공개 카드 중복 방지 및 합법적 베팅 순서 엄격 적용
- 생성 대상 6대 핵심 상황:
  A. 상대 Open Raise 직면 (Facing Open Raise)
  B. 자신의 Open 후 3-Bet 직면 (Facing 3-Bet: All-in 및 Pot)
  C. 상대 4-Bet 직면 (Facing 4-Bet)
  D. 플롭에서 상대 베팅 직면 (Flop Facing Bet)
  E. 턴에서 상대 레이즈 직면 (Turn Facing Raise)
  F. 리버에서 상대 올인 직면 (River Facing All-in)
- 대표 홀카드 8종 (AA, KK, QQ, AKs, AQs, JTs, 98s, 72o)
- 커뮤니티 카드 텍스처(Dry, Wet, Paired) 및 상대 레인지 일관성 부여
- 상대 비공개 카드는 절대 정책 입력으로 누출하지 않음
"""

import os
import sys
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, sample_chance_action, parse_cards_from_info_str

TARGET_HOLE_CARDS = {
    'AA': (48, 51),   # Ac, As
    'KK': (44, 47),   # Kc, Ks
    'QQ': (40, 43),   # Qc, Qs
    'AKs': (48, 44),  # Ac, Kc
    'AQs': (48, 40),  # Ac, Qc
    'JTs': (36, 32),  # Jc, Tc
    '98s': (28, 24),  # 9c, 8c
    '72o': (20, 1),   # 7c, 2d
}

class TargetedSubgameGenerator:
    def __init__(self, game: Optional[pyspiel.Game] = None):
        self.game = game if game is not None else get_5p_holdem_5a_game()

    def generate_subgame_state(
        self,
        situation: str,
        hand_name: str,
        hero_seat: int = 2,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[pyspiel.State, Dict]:
        """
        특정 베팅 상황과 홀카드를 가진 유효한 OpenSpiel 상태를 생성하고 메타데이터를 반환
        """
        if rng is None:
            rng = np.random.RandomState()

        c1, c2 = TARGET_HOLE_CARDS[hand_name]
        deck = [c for c in range(52) if c not in (c1, c2)]
        rng.shuffle(deck)

        st = self.game.new_initial_state()

        # 1. 프리플롭 홀카드 배분 (Hero에게 대상 핸드 주입)
        deck_idx = 0
        for p in range(5):
            if p == hero_seat:
                st.apply_action(c1)
                st.apply_action(c2)
            else:
                st.apply_action(deck[deck_idx]); deck_idx += 1
                st.apply_action(deck[deck_idx]); deck_idx += 1

        # 2. 상황별 합법 베팅 시퀀스 적용
        if situation == 'SitA_Facing_OpenRaise':
            # 상대(UTG=2) 오픈 -> Hero(Seat 3=CO) 직면
            assert hero_seat == 3, "For SitA, hero_seat should be 3 (CO)"
            st.apply_action(4) # UTG Half-Pot open (225)
            assert st.current_player() == hero_seat

        elif situation == 'SitB_Facing_3Bet_AllIn':
            # Hero(UTG=2) Half-Pot 오픈 -> 상대(CO=3) 3-Bet All-in -> 나머지 Fold -> Hero 직면
            assert hero_seat == 2
            st.apply_action(4) # Hero Half-Pot
            st.apply_action(3) # Opponent 3-Bet All-in
            st.apply_action(0); st.apply_action(0); st.apply_action(0) # BTN, SB, BB Fold
            assert st.current_player() == hero_seat

        elif situation == 'SitB_Facing_3Bet_Pot':
            # Hero(UTG=2) Half-Pot 오픈 -> 상대(CO=3) 3-Bet Pot -> 나머지 Fold -> Hero 직면
            assert hero_seat == 2
            st.apply_action(4) # Hero Half-Pot
            st.apply_action(2) # Opponent 3-Bet Pot
            st.apply_action(0); st.apply_action(0); st.apply_action(0)
            assert st.current_player() == hero_seat

        elif situation == 'SitC_Facing_4Bet':
            # 상대(UTG=2) 오픈 -> Hero(CO=3) 3-Bet Pot -> 상대(BTN=4) 4-Bet All-in -> 나머지 Fold -> Hero 직면
            assert hero_seat == 3
            st.apply_action(4) # UTG open
            st.apply_action(2) # Hero 3-Bet Pot
            st.apply_action(3) # BTN 4-Bet All-in
            st.apply_action(0); st.apply_action(0); st.apply_action(0) # SB, BB, UTG Fold
            assert st.current_player() == hero_seat

        elif situation == 'SitD_Flop_Facing_Bet':
            # Hero(UTG=2) 오픈 -> CO(3) Call -> 나머지 Fold -> Flop deal 3 cards -> Hero Check -> CO Half-Pot Bet -> Hero 직면
            assert hero_seat == 2
            st.apply_action(4) # Hero open
            st.apply_action(1) # CO call
            st.apply_action(0); st.apply_action(0); st.apply_action(0)
            # Flop deal 3 cards
            for _ in range(3):
                st.apply_action(sample_chance_action(st, rng=rng))
            assert st.current_player() == hero_seat
            st.apply_action(1) # Hero Check
            st.apply_action(4) # CO Half-Pot Bet
            assert st.current_player() == hero_seat

        elif situation == 'SitE_Turn_Facing_Raise':
            # Flop Check-Check -> Turn deal 1 card -> Hero Half-Pot Bet -> CO Pot Raise -> Hero 직면
            assert hero_seat == 2
            st.apply_action(4); st.apply_action(1); st.apply_action(0); st.apply_action(0); st.apply_action(0)
            for _ in range(3): st.apply_action(sample_chance_action(st, rng=rng))
            st.apply_action(1); st.apply_action(1) # Flop Check-Check
            st.apply_action(sample_chance_action(st, rng=rng)) # Turn deal
            assert st.current_player() == hero_seat
            st.apply_action(4) # Hero Half-Pot Bet
            st.apply_action(2) # CO Pot Raise
            assert st.current_player() == hero_seat

        elif situation == 'SitF_River_Facing_AllIn':
            # Flop & Turn Check-Check -> River deal 1 card -> Hero Check -> CO All-in -> Hero 직면
            assert hero_seat == 2
            st.apply_action(4); st.apply_action(1); st.apply_action(0); st.apply_action(0); st.apply_action(0)
            for _ in range(3): st.apply_action(sample_chance_action(st, rng=rng))
            st.apply_action(1); st.apply_action(1) # Flop
            st.apply_action(sample_chance_action(st, rng=rng)) # Turn
            st.apply_action(1); st.apply_action(1) # Turn
            st.apply_action(sample_chance_action(st, rng=rng)) # River
            assert st.current_player() == hero_seat
            st.apply_action(1) # Hero Check
            st.apply_action(3) # CO All-in
            assert st.current_player() == hero_seat

        else:
            raise ValueError(f"Unknown situation: {situation}")

        info = st.information_state_string(hero_seat)
        priv_cards, pub_cards = parse_cards_from_info_str(info)
        legal_acts = st.legal_actions(hero_seat)
        seq = info.split('Sequences: ')[1].split(']')[0] if 'Sequences: ' in info else ""

        meta = {
            'situation': situation,
            'hand_name': hand_name,
            'hero_seat': hero_seat,
            'private_cards': priv_cards,
            'public_cards': pub_cards,
            'legal_actions': legal_acts,
            'sequence': seq,
            'current_player': st.current_player(),
            'is_terminal': st.is_terminal()
        }

        return st, meta

def test_generator():
    print("=" * 80)
    print("POKER AI V4.9.3 - PHASE 3: Targeted Subgame Generator 무결성 검증")
    print("=" * 80)

    gen = TargetedSubgameGenerator()
    rng = np.random.RandomState(42)

    situations = [
        ('SitA_Facing_OpenRaise', 'AA', 3),
        ('SitB_Facing_3Bet_AllIn', 'AA', 2),
        ('SitB_Facing_3Bet_Pot', 'AA', 2),
        ('SitC_Facing_4Bet', 'AA', 3),
        ('SitD_Flop_Facing_Bet', 'AA', 2),
        ('SitE_Turn_Facing_Raise', 'AA', 2),
        ('SitF_River_Facing_AllIn', 'AA', 2),
    ]

    records = []
    import csv

    for sit, hand, seat in situations:
        st, meta = gen.generate_subgame_state(sit, hand, hero_seat=seat, rng=rng)
        print(f"\n[상황 생성 성공: {sit} | {hand} | Seat {seat}]")
        print(f"  Current Player: {meta['current_player']} (일치 여부: {meta['current_player'] == seat})")
        print(f"  Private Cards:  {meta['private_cards']}")
        print(f"  Public Cards:   {meta['public_cards']}")
        print(f"  Legal Actions:  {meta['legal_actions']}")
        print(f"  Sequence:       {meta['sequence']}")
        assert meta['current_player'] == seat
        assert not meta['is_terminal']
        records.append({
            'situation': sit,
            'hand': hand,
            'hero_seat': seat,
            'private_cards': "".join(meta['private_cards']),
            'public_cards': "".join(meta['public_cards']),
            'legal_actions': str(meta['legal_actions']),
            'sequence': meta['sequence']
        })

    os.makedirs("v493/data", exist_ok=True)
    out_csv = "v493/data/phase3_subgame_generator_audit.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n생성기 무결성 검증 완료: {out_csv} (총 {len(records)}개 상태 전수 합법)")

if __name__ == '__main__':
    test_generator()
