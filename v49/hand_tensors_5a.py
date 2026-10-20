"""
POKER AI V4.9 - 표준 9개 핸드 상태 텐서 생성기 (UTG Preflop First Action in 5A FCHPA)
- 핸드: AA, KK, QQ, AKs, AQs, JTs, 98s, 72o, 83o
- 위치: UTG (Player 2, 첫 번째 의사결정)
- 법적 행동: [0: Fold, 1: Call, 2: Pot, 3: All-in, 4: Half-Pot]
"""

import numpy as np
import torch
from typing import Dict, Tuple
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes, parse_cards_from_info_str

HAND_SEEDS_5A = {
    'KK': 3,
    'JTs': 37,
    '98s': 53,
    '72o': 67,
    '83o': 98,
    'QQ': 112,
    'AA': 147,
    'AKs': 178,
    'AQs': 276
}

def get_standard_hand_tensors_5a(device: torch.device = torch.device("cpu")) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, list]]:
    game = get_5p_holdem_5a_game()
    hand_data = {}
    
    for hname, seed in HAND_SEEDS_5A.items():
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
        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
        for a in legal:
            m[0, a] = True
            
        hand_data[hname] = (x, m, hc)
        
    return hand_data
