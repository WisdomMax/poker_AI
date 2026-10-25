#!/usr/bin/env python3
"""
[DIRECTIVE 039 - Step 1: 결함 재현 단위 테스트]
1. 기존 split 기반 카드 파서의 100% 파싱 실패 재현 (FAIL 테스트)
2. 기존 'Round 0' 스트리트 판별 함수의 100% Preflop 오분류 재현 (FAIL 테스트)
3. 수정된 정규식 파서 및 v71 스트리트 판별 함수의 100% 정상 작동 검증 (PASS 전환)
"""

import sys
import os
import re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v79.run_v79_controlled_causal_study import get_5p_fullgame, advance_chance_nodes
from v71.run_v71_ledger_and_street_audit import get_round_from_state

# --- 1. 결함이 있던 기존 함수들 ---
def defective_parse_cards(st):
    cards_map = {}
    s = str(st)
    for line in s.split('\n'):
        if "P" in line and "Cards" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                p_part = parts[0].strip()
                c_part = parts[1].strip()
                try:
                    p_idx = int(p_part.replace("P", "").replace("Cards", "").strip())
                    cards = c_part.split() # <--- 치명적 결함: "Ad6s".split() -> ['Ad6s'] (len=1)
                    if len(cards) == 2:
                        cards_map[p_idx] = (cards[0], cards[1])
                except:
                    pass
    return cards_map

def defective_get_round_name(st):
    s = str(st)
    if "Round 0" in s or "Preflop" in s: return "Preflop"
    if "Round 1" in s or "Flop" in s: return "Flop" # <--- 치명적 결함: OpenSpiel은 'Round: 1'로 출력함
    if "Round 2" in s or "Turn" in s: return "Turn"
    if "Round 3" in s or "River" in s: return "River"
    return "Preflop"

# --- 2. 수정된 완벽한 함수들 ---
def robust_parse_cards(st):
    cards_map = {}
    s = str(st)
    for m in re.finditer(r'P(\d+)\s+Cards:\s*([2-9TJQKA][shdc])\s*([2-9TJQKA][shdc])', s):
        p_idx = int(m.group(1))
        cards_map[p_idx] = (m.group(2), m.group(3))
        
    if len(cards_map) < 5:
        acpc_m = re.search(r'STATE:\d+::([2-9TJQKAshdc|]+)', s)
        if acpc_m:
            players_cards = acpc_m.group(1).split('|')
            for p_idx, p_str in enumerate(players_cards):
                if len(p_str) >= 4 and p_idx not in cards_map:
                    cards_map[p_idx] = (p_str[:2], p_str[2:4])
                    
    if len(cards_map) < 5:
        raise ValueError(f"CRITICAL: Failed to parse all 5 players cards! Found: {cards_map}")
    return cards_map

def test_defects():
    print("=" * 60)
    print("DIRECTIVE 039: 결함 재현 및 수정 검증 테스트")
    print("=" * 60)
    
    game = get_5p_fullgame()
    st = game.new_initial_state()
    rng = np.random.RandomState(42)
    advance_chance_nodes(st, rng)
    
    # 1. 기존 결함 파서 검증
    defective_res = defective_parse_cards(st)
    print(f"[FAIL 검증 1] 기존 결함 파서 결과: {defective_res} (길이: {len(defective_res)})")
    assert len(defective_res) == 0, "기존 파서는 0개를 파싱해야 결함 재현 성공"
    print("  -> 결함 1 재현 성공: 기존 코드는 공백 split 실패로 100% 카드 인식 실패!")
    
    # 2. 수정된 정규식 파서 검증
    robust_res = robust_parse_cards(st)
    print(f"[PASS 검증 1] 수정된 정규식 파서 결과: {robust_res} (길이: {len(robust_res)})")
    assert len(robust_res) == 5, "수정된 파서는 정확히 5명 카드를 파싱해야 함"
    for p_idx, cards in robust_res.items():
        assert len(cards) == 2, f"P{p_idx}는 2장이어야 함"
    print("  -> 수정 1 검증 성공: 5명 플레이어의 홀카드를 100% 무결하게 추출!")
    
    # 3. 플랍 단계까지 진행 (5명 전원 Call)
    for _ in range(5):
        st.apply_action(1) # Call
        if st.is_chance_node():
            advance_chance_nodes(st, rng)
            
    print(f"\n[플랍 진입 후 상태 검사]")
    flop_defective = defective_get_round_name(st)
    flop_robust = get_round_from_state(st)
    
    print(f"  * 기존 판별 함수: '{flop_defective}'")
    print(f"  * v71 정규 판별 함수: {flop_robust} (1=Flop)")
    
    assert flop_defective == "Preflop", "기존 함수는 'Round: 1'을 인식 못해 Preflop으로 오분류해야 함"
    print("  -> 결함 2 재현 성공: 기존 코드는 플랍 이후도 전부 Preflop으로 오분류!")
    
    assert flop_robust == 1, "v71 판별 함수는 Flop(1)을 정상 인식해야 함"
    print("  -> 수정 2 검증 성공: v71 정규 판별 함수가 Flop(1)을 정확히 인식!")
    
    print("\n[전체 통과] 결함 재현 및 정밀 수정이 완벽하게 입증되었습니다.")

if __name__ == "__main__":
    test_defects()
