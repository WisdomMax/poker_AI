"""
POKER AI V4.8.3 - 표준 8개 핸드 상태 텐서 생성기 (UTG Preflop First Action)
- 핸드: AA, KK, QQ, AKs, AQs, JTs, 72o, 83o
- 위치: UTG (Player 2, 첫 번째 의사결정)
- 법적 행동: [0: Fold, 1: Call, 2: Pot, 3: All-in]
"""

import numpy as np
import torch
from typing import Dict, Tuple
from v48.poker_environment import get_5p_holdem_game, advance_chance_nodes, parse_cards_from_info_str

TARGET_HAND_DEFS = {
    'AA': lambda r1, r2, s1, s2: r1 == 'A' and r2 == 'A',
    'KK': lambda r1, r2, s1, s2: r1 == 'K' and r2 == 'K',
    'QQ': lambda r1, r2, s1, s2: r1 == 'Q' and r2 == 'Q',
    'AKs': lambda r1, r2, s1, s2: ((r1=='A' and r2=='K') or (r1=='K' and r2=='A')) and s1 == s2,
    'AQs': lambda r1, r2, s1, s2: ((r1=='A' and r2=='Q') or (r1=='Q' and r2=='A')) and s1 == s2,
    'JTs': lambda r1, r2, s1, s2: ((r1=='J' and r2=='T') or (r1=='T' and r2=='J')) and s1 == s2,
    '72o': lambda r1, r2, s1, s2: ((r1=='7' and r2=='2') or (r1=='2' and r2=='7')) and s1 != s2,
    '83o': lambda r1, r2, s1, s2: ((r1=='8' and r2=='3') or (r1=='3' and r2=='8')) and s1 != s2,
}

# 검증된 결정론적 시드
HAND_SEEDS = {
    'KK': 3,
    'JTs': 37,
    '72o': 67,
    '83o': 98,
    'QQ': 112,
    'AA': 147,
    'AKs': 178,
    'AQs': 276
}

def get_standard_hand_tensors(device: torch.device = torch.device("cpu")) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, list]]:
    """
    각 핸드별 (state_tensor, mask_tensor, hole_cards) 반환
    state_tensor shape: [1, 253]
    mask_tensor shape: [1, 4]
    """
    game = get_5p_holdem_game()
    hand_data = {}
    
    for hname, seed in HAND_SEEDS.items():
        rng = np.random.RandomState(seed)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng)
        cur_p = st.current_player()
        assert cur_p == 2, f"Expected UTG (seat 2), got {cur_p}"
        
        info = st.information_state_string(cur_p)
        hc, _ = parse_cards_from_info_str(info)
        vec = st.information_state_tensor(cur_p)
        legal = st.legal_actions(cur_p)
        
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.zeros((1, 4), dtype=torch.bool, device=device)
        for a in legal:
            m[0, a] = True
            
        hand_data[hname] = (x, m, hc)
        
    return hand_data
