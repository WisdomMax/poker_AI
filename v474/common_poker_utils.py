"""
POKER AI V4.7.4 - PHASE 3: 공통 카드 샘플링 유틸리티 (Common Chance Action Sampler)
- 목적: 훈련, 평가, 벤치마크 전반에서 단일하고 검증된 확률적 카드 샘플링 함수를 공유하여
  결정론적 카드 고정 오류(chance_outcomes()[0][0])의 재발을 원천 차단.
- 기능:
  1. sample_chance_action(state, rng=None): 확률 분포 p에 따른 안전한 액션 샘플링
  2. apply_chance_actions_until_player_node(state, rng=None): 플레이어 노드까지 자동 찬스 처리
  3. verify_sampler_integrity(): 시드 재현성, 다양성, 불변식 단위 테스트
"""

import random
from typing import Optional, List, Tuple
import numpy as np
import pyspiel

def sample_chance_action(state: pyspiel.State, rng: Optional[np.random.RandomState] = None) -> int:
    """
    OpenSpiel 찬스 노드에서 (action, probability)를 기반으로 확률 가중치 샘플링을 수행.
    결정론적 고정([0][0])을 엄격히 방지.
    """
    if not state.is_chance_node():
        raise ValueError(f"State is not a chance node! Current player: {state.current_player()}")
        
    outcomes = state.chance_outcomes()
    if not outcomes:
        raise RuntimeError("State is chance node but chance_outcomes() returned empty list!")
        
    actions, probs = zip(*outcomes)
    
    # 확률 정규화 (부동소수점 오차 보정)
    prob_sum = sum(probs)
    if prob_sum <= 0:
        raise ValueError(f"Sum of probabilities is non-positive: {prob_sum}")
    norm_probs = np.array(probs, dtype=np.float64) / prob_sum
    
    if rng is not None:
        chosen_action = int(rng.choice(actions, p=norm_probs))
    else:
        chosen_action = int(np.random.choice(actions, p=norm_probs))
        
    return chosen_action

def advance_through_chance_nodes(state: pyspiel.State, rng: Optional[np.random.RandomState] = None) -> List[int]:
    """
    플레이어 의사결정 노드 또는 터미널 노드에 도달할 때까지 모든 찬스 액션을 확률적으로 적용.
    적용된 찬스 액션 리스트 반환.
    """
    actions_taken = []
    while state.is_chance_node() and not state.is_terminal():
        act = sample_chance_action(state, rng=rng)
        state.apply_action(act)
        actions_taken.append(act)
    return actions_taken

def verify_sampler_integrity(num_trials: int = 1000):
    """
    공통 카드 샘플링 함수의 수학적/구현적 무결성을 검증하는 단위 테스트.
    1. 동일 시드 재현성 (Reproducibility)
    2. 상이 시드 다양성 (Diversity)
    3. 카드 고정(단일 패 반복) 발생 시 즉각 실패
    """
    print("=" * 80)
    print("🧪 [V4.7.4 PHASE 3] 공통 카드 샘플링 함수 무결성 단위 테스트")
    print("=" * 80)
    
    params = {
        "betting": "nolimit",
        "bettingAbstraction": "fcpa",
        "numPlayers": 5,
        "numRanks": 13,
        "numSuits": 4,
        "numHoleCards": 2,
        "numRounds": 4,
        "numBoardCards": "0 3 1 1",
        "blind": "50 100 0 0 0",
        "stack": "10000 10000 10000 10000 10000",
        "firstPlayer": "3 1 1 1"
    }
    game = pyspiel.load_game("universal_poker", params)
    
    # 1. 동일 시드 재현성 검증
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    
    st1 = game.new_initial_state()
    acts1 = advance_through_chance_nodes(st1, rng=rng1)
    
    st2 = game.new_initial_state()
    acts2 = advance_through_chance_nodes(st2, rng=rng2)
    
    assert acts1 == acts2, f"❌ [동일 시드 불일치] acts1 != acts2! {acts1} vs {acts2}"
    print(f"  ✅ 1. 동일 시드 재현성 통과 (Seed 42 일치: {len(acts1)}개 카드 액션 동일)")
    
    # 2. 상이 시드 다양성 검증
    rng3 = np.random.RandomState(999)
    st3 = game.new_initial_state()
    acts3 = advance_through_chance_nodes(st3, rng=rng3)
    assert acts1 != acts3, "❌ [상이 시드 중복] 서로 다른 시드인데 카드 배분이 동일합니다!"
    print(f"  ✅ 2. 상이 시드 다양성 통과 (Seed 42 vs 999 차이 확인)")
    
    # 3. 1,000회 연속 배분 중 고유 핸드 수 다양성 검증 (카드 고정 버그 즉각 감지)
    unique_deals = set()
    for i in range(num_trials):
        st = game.new_initial_state()
        acts = tuple(advance_through_chance_nodes(st))
        unique_deals.add(acts)
        
    diversity_ratio = len(unique_deals) / num_trials
    print(f"  ✅ 3. 1,000회 연속 배분 고유성: {len(unique_deals)}/{num_trials}개 ({diversity_ratio*100:.1f}%)")
    assert diversity_ratio > 0.99, f"❌ [카드 고정 의심] 고유 배분 비율이 너무 낮습니다 ({diversity_ratio*100:.1f}%)!"
    print("  👉 결론: 공통 카드 샘플링 함수가 확률적 무작위성과 시드 재현성을 완벽히 충족합니다.")
    print("=" * 80)

if __name__ == "__main__":
    verify_sampler_integrity()
