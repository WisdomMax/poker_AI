"""
POKER AI V4.9.2 - PHASE 1. 기존 정책 결함 정밀 검증
- 7대 상황 (A~G) 및 8종 대표 핸드에 대한 합법 행동 정책 확률 전수 측정
- 상황 A: 프리플롭 최초 행동 (Unopened pot)
- 상황 B: 상대 Open Raise 직면
- 상황 C: 자신의 Open Raise 이후 상대 3-Bet 직면 (All-in 및 Pot)
- 상황 D: 상대 4-Bet 직면
- 상황 E: 플롭에서 상대 베팅 직면
- 상황 F: 턴에서 상대 재레이즈 직면
- 상황 G: 리버에서 상대 올인 직면
- 대표 핸드 8종: AA, KK, QQ, AKs, AQs, JTs, 98s, 72o
- AA 3-Bet 다중 독립 상태 재현 검증
"""

import os
import sys
import csv
import numpy as np
import torch
import pyspiel

# 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v49.poker_environment_5a import get_5p_holdem_5a_game, ACTION_NAMES_5A, sample_chance_action
from v49.ppo_actor_critic_5a import HoldemActorCritic5A

# 핸드 카드 매핑 (Rank * 4 + Suit: c=0, d=1, h=2, s=3)
# 2: 0..3, 3: 4..7, ..., 9: 28..31, T: 32..35, J: 36..39, Q: 40..43, K: 44..47, A: 48..51
TARGET_HANDS = {
    'AA': (48, 51),   # Ac, As
    'KK': (44, 47),   # Kc, Ks
    'QQ': (40, 43),   # Qc, Qs
    'AKs': (48, 44),  # Ac, Kc
    'AQs': (48, 40),  # Ac, Qc
    'JTs': (36, 32),  # Jc, Tc
    '98s': (28, 24),  # 9c, 8c
    '72o': (20, 1),   # 7c, 2d
}

def create_preflop_deal(game, target_player: int, card1: int, card2: int, rng_seed: int = 42):
    rng = np.random.RandomState(rng_seed)
    deck = [c for c in range(52) if c not in (card1, card2)]
    rng.shuffle(deck)
    st = game.new_initial_state()
    deck_idx = 0
    for p in range(5):
        if p == target_player:
            st.apply_action(card1)
            st.apply_action(card2)
        else:
            st.apply_action(deck[deck_idx])
            deck_idx += 1
            st.apply_action(deck[deck_idx])
            deck_idx += 1
    return st

def build_situation_state(game, sit_name: str, card1: int, card2: int, rng_seed: int = 42):
    """
    각 상황(A~G)에 맞는 유효한 OpenSpiel 상태를 생성하고,
    검사 대상 플레이어의 tensor와 legal mask를 반환한다.
    """
    if sit_name == 'SitA_Preflop_First':
        # 상황 A: UTG (P2) 최초 행동
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        target_player = 2
        assert st.current_player() == target_player

    elif sit_name == 'SitB_Facing_OpenRaise':
        # 상황 B: UTG가 Half-Pot 오픈 후 CO (P3)가 직면
        st = create_preflop_deal(game, target_player=3, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4) # UTG Half-Pot open
        target_player = 3
        assert st.current_player() == target_player

    elif sit_name == 'SitC_Facing_3Bet_AllIn':
        # 상황 C-1: UTG(P2) Half-Pot 오픈 -> CO(P3) All-in 3-Bet -> 나머지 Fold -> P2 직면
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4) # P2 Half-Pot
        st.apply_action(3) # P3 3-Bet All-in
        st.apply_action(0) # P4 fold
        st.apply_action(0) # P0 fold
        st.apply_action(0) # P1 fold
        target_player = 2
        assert st.current_player() == target_player

    elif sit_name == 'SitC_Facing_3Bet_Pot':
        # 상황 C-2: UTG(P2) Half-Pot 오픈 -> CO(P3) Pot 3-Bet -> 나머지 Fold -> P2 직면
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4) # P2 Half-Pot
        st.apply_action(2) # P3 3-Bet Pot
        st.apply_action(0) # P4 fold
        st.apply_action(0) # P0 fold
        st.apply_action(0) # P1 fold
        target_player = 2
        assert st.current_player() == target_player

    elif sit_name == 'SitD_Facing_4Bet_AllIn':
        # 상황 D: UTG(P2) 오픈(4) -> CO(P3, Target) 3-Bet Pot(2) -> BTN(P4) 4-Bet All-in(3)
        # -> SB(P0) f, BB(P1) f, UTG(P2) f -> P3 직면
        st = create_preflop_deal(game, target_player=3, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4) # P2 open
        st.apply_action(2) # P3 3-Bet Pot
        st.apply_action(3) # P4 4-Bet All-in
        st.apply_action(0) # P0 fold
        st.apply_action(0) # P1 fold
        st.apply_action(0) # P2 fold
        target_player = 3
        assert st.current_player() == target_player

    elif sit_name == 'SitE_Flop_Facing_Bet':
        # 상황 E: 플롭에서 상대 베팅 직면
        # Preflop: P2 Half-Pot(4) -> P3 Call(1) -> P4,P0,P1 Fold(0)
        # Flop deal 3 cards -> P2 Check(1) -> P3 Half-Pot Bet(4) -> P2 직면
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4)
        st.apply_action(1)
        st.apply_action(0); st.apply_action(0); st.apply_action(0)
        rng = np.random.RandomState(rng_seed + 100)
        for _ in range(3):
            st.apply_action(sample_chance_action(st, rng=rng))
        assert st.current_player() == 2
        st.apply_action(1) # P2 Check
        st.apply_action(4) # P3 Half-Pot Bet
        target_player = 2
        assert st.current_player() == target_player

    elif sit_name == 'SitF_Turn_Facing_Raise':
        # 상황 F: 턴에서 상대 재레이즈 직면
        # Preflop: P2(4), P3(1), others(0)
        # Flop: P2 Check(1), P3 Check(1)
        # Turn deal 1 card -> P2 Half-Pot Bet(4) -> P3 Pot Raise(2) -> P2 직면
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4); st.apply_action(1); st.apply_action(0); st.apply_action(0); st.apply_action(0)
        rng = np.random.RandomState(rng_seed + 100)
        for _ in range(3):
            st.apply_action(sample_chance_action(st, rng=rng))
        st.apply_action(1); st.apply_action(1) # Check-Check on Flop
        st.apply_action(sample_chance_action(st, rng=rng)) # Turn deal
        assert st.current_player() == 2
        st.apply_action(4) # P2 Half-Pot Bet
        st.apply_action(2) # P3 Pot Raise
        target_player = 2
        assert st.current_player() == target_player

    elif sit_name == 'SitG_River_Facing_AllIn':
        # 상황 G: 리버에서 상대 올인 직면
        # Preflop: P2(4), P3(1), others(0)
        # Flop: P2(1), P3(1)
        # Turn: P2(1), P3(1)
        # River: P2 Check(1) -> P3 All-in(3) -> P2 직면
        st = create_preflop_deal(game, target_player=2, card1=card1, card2=card2, rng_seed=rng_seed)
        st.apply_action(4); st.apply_action(1); st.apply_action(0); st.apply_action(0); st.apply_action(0)
        rng = np.random.RandomState(rng_seed + 100)
        for _ in range(3):
            st.apply_action(sample_chance_action(st, rng=rng))
        st.apply_action(1); st.apply_action(1) # Flop Check-Check
        st.apply_action(sample_chance_action(st, rng=rng)) # Turn
        st.apply_action(1); st.apply_action(1) # Turn Check-Check
        st.apply_action(sample_chance_action(st, rng=rng)) # River
        assert st.current_player() == 2
        st.apply_action(1) # P2 Check
        st.apply_action(3) # P3 All-in
        target_player = 2
        assert st.current_player() == target_player

    else:
        raise ValueError(f"Unknown situation {sit_name}")

    vec = st.information_state_tensor(target_player)
    legal = st.legal_actions(target_player)
    
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    mask = torch.zeros((1, 5), dtype=torch.bool)
    for a in legal:
        mask[0, a] = True
        
    return x, mask, legal, st.information_state_string(target_player)

def evaluate_model_policy(model: HoldemActorCritic5A, x: torch.Tensor, mask: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs

def run_phase1_audit():
    print("=" * 80)
    print("POKER AI V4.9.2 - PHASE 1: 기존 정책 결함 정밀 감사 시작")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    device = torch.device("cpu")

    # 검사 대상 모델 목록
    seeds = [42, 101, 202, 303, 404]
    models_dict = {}

    for s in seeds:
        path_c = f"v49/models/CondC_5A_Transfer_seed_{s}_step_20000.pt"
        if os.path.exists(path_c):
            m = HoldemActorCritic5A()
            m.load_state_dict(torch.load(path_c, map_location=device))
            models_dict[f"CondC_5A_Transfer_s{s}"] = m

    for s in seeds:
        path_b = f"v49/models/CondB_5A_Scratch_seed_{s}_step_20000.pt"
        if os.path.exists(path_b):
            m = HoldemActorCritic5A()
            m.load_state_dict(torch.load(path_b, map_location=device))
            models_dict[f"CondB_5A_Scratch_s{s}"] = m

    print(f"로드된 모델 수: {len(models_dict)}")

    situations = [
        'SitA_Preflop_First',
        'SitB_Facing_OpenRaise',
        'SitC_Facing_3Bet_AllIn',
        'SitC_Facing_3Bet_Pot',
        'SitD_Facing_4Bet_AllIn',
        'SitE_Flop_Facing_Bet',
        'SitF_Turn_Facing_Raise',
        'SitG_River_Facing_AllIn'
    ]

    records = []

    for m_name, model in models_dict.items():
        cond_type = 'Transfer' if 'Transfer' in m_name else 'Scratch'
        seed_val = int(m_name.split('_s')[1])

        for sit in situations:
            for hand_name, (c1, c2) in TARGET_HANDS.items():
                x, mask, legal, info_str = build_situation_state(game, sit, c1, c2)
                probs = evaluate_model_policy(model, x, mask)

                rec = {
                    'model_name': m_name,
                    'condition': cond_type,
                    'seed': seed_val,
                    'situation': sit,
                    'hand': hand_name,
                    'legal_actions': str(legal),
                    'prob_fold': float(probs[0]),
                    'prob_call': float(probs[1]),
                    'prob_pot': float(probs[2]),
                    'prob_allin': float(probs[3]),
                    'prob_halfpot': float(probs[4])
                }
                records.append(rec)

    out_path = "v492/data/phase1_defect_metrics.csv"
    os.makedirs("v492/data", exist_ok=True)
    if records:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    print(f"전체 정책 메트릭 저장 완료: {out_path} (총 {len(records)} 행)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AA 3-Bet 다중 독립 상태 재현 검증
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("AA 3-Bet 직면 다중 독립 상태 재현 검증")
    print("=" * 80)

    # 5가지 독립된 AA 3-bet 상황 정의
    aa_states = {}

    # 1. UTG Half-Pot(225) -> BTN All-in(10000) -> Fold to UTG
    st1 = create_preflop_deal(game, target_player=2, card1=48, card2=51, rng_seed=11)
    st1.apply_action(4); st1.apply_action(0); st1.apply_action(3); st1.apply_action(0); st1.apply_action(0)
    aa_states["State1_OpenHalfPot_BTN_3BetAllIn"] = st1

    # 2. UTG Pot(350) -> CO All-in(10000) -> Fold to UTG
    st2 = create_preflop_deal(game, target_player=2, card1=48, card2=51, rng_seed=22)
    st2.apply_action(2); st2.apply_action(3); st2.apply_action(0); st2.apply_action(0); st2.apply_action(0)
    aa_states["State2_OpenPot_CO_3BetAllIn"] = st2

    # 3. UTG Half-Pot(225) -> BB All-in(10000) -> Fold to UTG
    st3 = create_preflop_deal(game, target_player=2, card1=48, card2=51, rng_seed=33)
    st3.apply_action(4); st3.apply_action(0); st3.apply_action(0); st3.apply_action(0); st3.apply_action(3)
    aa_states["State3_OpenHalfPot_BB_3BetAllIn"] = st3

    # 4. UTG Half-Pot(225) -> CO 3-Bet Pot(825) -> Fold to UTG
    st4 = create_preflop_deal(game, target_player=2, card1=48, card2=51, rng_seed=44)
    st4.apply_action(4); st4.apply_action(2); st4.apply_action(0); st4.apply_action(0); st4.apply_action(0)
    aa_states["State4_OpenHalfPot_CO_3BetPot"] = st4

    # 5. UTG Pot(350) -> BTN 3-Bet Pot(1225) -> Fold to UTG
    st5 = create_preflop_deal(game, target_player=2, card1=48, card2=51, rng_seed=55)
    st5.apply_action(2); st5.apply_action(0); st5.apply_action(2); st5.apply_action(0); st5.apply_action(0)
    aa_states["State5_OpenPot_BTN_3BetPot"] = st5

    aa_records = []
    for s_name, st in aa_states.items():
        cur_p = st.current_player()
        assert cur_p == 2, f"Expected P2, got {cur_p}"
        vec = st.information_state_tensor(cur_p)
        legal = st.legal_actions(cur_p)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros((1, 5), dtype=torch.bool)
        for a in legal:
            mask[0, a] = True

        for m_name, model in models_dict.items():
            probs = evaluate_model_policy(model, x, mask)
            aa_records.append({
                'aa_state_id': s_name,
                'model_name': m_name,
                'condition': 'Transfer' if 'Transfer' in m_name else 'Scratch',
                'seed': int(m_name.split('_s')[1]),
                'sequences': st.information_state_string(cur_p).split('Sequences: ')[1].split(']')[0],
                'legal_actions': str(legal),
                'prob_fold': float(probs[0]),
                'prob_call': float(probs[1]),
                'prob_pot': float(probs[2]),
                'prob_allin': float(probs[3]),
                'prob_halfpot': float(probs[4])
            })

    aa_out_path = "v492/data/phase1_aa_3bet_reproduction.csv"
    if aa_records:
        with open(aa_out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(aa_records[0].keys()))
            writer.writeheader()
            writer.writerows(aa_records)
    print(f"AA 3-Bet 재현 데이터 저장 완료: {aa_out_path} (총 {len(aa_records)} 행)")

    # 핵심 요약 테이블 출력
    print("\n[요약 1: AA 3-Bet All-in 직면 시 (State 1) 모델별 Fold/Call 확률]")
    for r in aa_records:
        if r['aa_state_id'] == 'State1_OpenHalfPot_BTN_3BetAllIn':
            print(f"  {r['model_name']:25s} | Fold: {r['prob_fold']*100:6.2f}% | Call: {r['prob_call']*100:6.2f}%")

    print("\n[요약 2: AA 5가지 독립 3-Bet 상태별 Fold 확률 평균]")
    for s_name in aa_states.keys():
        t_folds = [r['prob_fold'] for r in aa_records if r['aa_state_id'] == s_name and r['condition'] == 'Transfer']
        s_folds = [r['prob_fold'] for r in aa_records if r['aa_state_id'] == s_name and r['condition'] == 'Scratch']
        print(f"  {s_name:32s} | Transfer Fold Avg: {np.mean(t_folds)*100:5.2f}% | Scratch Fold Avg: {np.mean(s_folds)*100:5.2f}%")

    print("\n[요약 3: 상황 C-1 (Half-Pot 오픈 후 3-Bet All-in 직면) 8종 핸드별 평균 Fold 확률]")
    for hand_name in TARGET_HANDS.keys():
        t_folds = [r['prob_fold'] for r in records if r['situation'] == 'SitC_Facing_3Bet_AllIn' and r['hand'] == hand_name and r['condition'] == 'Transfer']
        s_folds = [r['prob_fold'] for r in records if r['situation'] == 'SitC_Facing_3Bet_AllIn' and r['hand'] == hand_name and r['condition'] == 'Scratch']
        print(f"  Hand: {hand_name:5s} | Transfer Fold: {np.mean(t_folds)*100:5.2f}% | Scratch Fold: {np.mean(s_folds)*100:5.2f}%")

    print("\n[요약 4: 상황 A (프리플롭 최초 오픈) AA vs 72o 평균 정책]")
    for h in ['AA', '72o']:
        for cond in ['Transfer', 'Scratch']:
            sub = [r for r in records if r['situation'] == 'SitA_Preflop_First' and r['hand'] == h and r['condition'] == cond]
            f_avg = np.mean([r['prob_fold'] for r in sub]) * 100
            c_avg = np.mean([r['prob_call'] for r in sub]) * 100
            p_avg = np.mean([r['prob_pot'] for r in sub]) * 100
            a_avg = np.mean([r['prob_allin'] for r in sub]) * 100
            h_avg = np.mean([r['prob_halfpot'] for r in sub]) * 100
            print(f"  [{cond:8s}] {h:4s} -> Fold: {f_avg:5.1f}%, Call: {c_avg:5.1f}%, Pot: {p_avg:5.1f}%, All-in: {a_avg:5.1f}%, HalfPot: {h_avg:5.1f}%")


if __name__ == '__main__':
    run_phase1_audit()
