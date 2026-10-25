"""
DIRECTIVE 041 League Opponent Pool
Implements 5 Distinct Opponent Policies for League Training & Benchmarking:
1. SelfPlayLeagueBot (30%): Samples historical checkpoints (5k, 20k, 35k, 50k)
2. BalancedBot (25%): Balanced sizing (Half-pot, Pot, Check/Call, Fold) based on Chen & Board
3. RationalExploiter (20%): Traps and punishes aggressive 100% Pot/All-in spam
4. Aggressive3BetBot (15%): High-frequency 3-Bet preflop and C-bet flop pressure
5. DynamicRandomBot (10%): Unseen stochastic exploratory sizing
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

# Load Chen formula
RANKS = '23456789TJQKA'
RANK_VALS = {r: i + 2 for i, r in enumerate(RANKS)}

def calc_chen_score(c1: str, c2: str) -> float:
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    v1, v2 = RANK_VALS[r1], RANK_VALS[r2]
    high_val = max(v1, v2)
    low_val = min(v1, v2)
    
    if high_val == 14: base = 10.0
    elif high_val == 13: base = 8.0
    elif high_val == 12: base = 7.0
    elif high_val == 11: base = 6.0
    else: base = high_val / 2.0
    
    if r1 == r2:
        base = max(base * 2.0, 5.0)
        
    if s1 == s2:
        base += 2.0
        
    gap = high_val - low_val
    if gap == 1: base -= 1.0
    elif gap == 2: base -= 2.0
    elif gap == 3: base -= 4.0
    elif gap >= 4: base -= 5.0
    
    if gap in (1, 2) and high_val < 12 and (r1 != r2):
        base += 1.0
        
    return base

_POOL_CACHE = None

def get_cached_pool(pool_dir: str = "v1024/models_e2e"):
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE
    models = []
    ckpt_files = [
        "PPO_E2E_Pilot_5k.pt",
        "PPO_E2E_League_20k.pt",
        "PPO_E2E_Safeguard_35k.pt",
        "PPO_E2E_Final_50k.pt"
    ]
    for cf in ckpt_files:
        cp_path = os.path.join(pool_dir, cf)
        if os.path.exists(cp_path):
            m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
            m.load_state_dict(torch.load(cp_path, map_location="cpu", weights_only=False))
            m.eval()
            models.append(m)
    if not models:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.eval()
        models.append(m)
    _POOL_CACHE = models
    return _POOL_CACHE

class SelfPlayLeagueBot:
    """과거 체크포인트(5k, 20k, 35k, 50k) 풀에서 샘플링된 자가대련 에이전트"""
    name = "SelfPlay_Checkpoint_League"
    def __init__(self, seat: int, seed: int, pool_dir: str = "v1024/models_e2e"):
        self.seat = seat
        self.rng = random.Random(seed)
        self.models = get_cached_pool(pool_dir)
        self.active_model = self.rng.choice(self.models)

    def act(self, st, cp: int, legal_6: List[int], priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        with torch.no_grad():
            t_inf = torch.from_numpy(inf).unsqueeze(0)
            logits, _ = self.active_model(t_inf)
            mask = torch.full_like(logits, -1e9)
            for a in legal_6:
                mask[0, a] = 0.0
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            
        return int(self.rng.choices(range(6), weights=probs)[0])

class BalancedBot:
    """균형형 정책: 하프팟(액션 2)과 팟(액션 3), 체크/콜(액션 1)을 고르게 안배"""
    name = "Balanced_GTO_Baseline"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: List[int], priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        
        if is_allin_facing and to_call > 0:
            # 타이트하게 Chen >= 8.5 이상만 올인 콜
            if chen >= 8.5 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1

        if round_idx == 0:  # Preflop
            if chen >= 9.0: # Premium (AA, KK, AKs, QQ)
                if 2 in legal_6 and self.rng.random() < 0.6: return 2 # 2x/3x Open Raise
                if 3 in legal_6: return 3 # Pot Raise
                return 1 if 1 in legal_6 else 0
            elif chen >= 6.5: # Solid playable
                if to_call == 0 and 1 in legal_6: return 1
                if to_call <= 100.0 and 1 in legal_6: return 1
                return 0 if 0 in legal_6 else 1
            else: # Fold trash
                if to_call == 0 and 1 in legal_6: return 1 # BB free check
                return 0 if 0 in legal_6 else 1
        else:  # Postflop
            if chen >= 8.0:
                # 밸류 벳: 50% 확률로 Half-pot(2), 30% 확률로 Pot(3), 20% 확률로 Check/Call(1)
                r = self.rng.random()
                if r < 0.50 and 2 in legal_6: return 2
                elif r < 0.80 and 3 in legal_6: return 3
                return 1 if 1 in legal_6 else 0
            elif chen >= 5.0:
                # 마진 핸드: 체크/콜
                if to_call <= 200.0 and 1 in legal_6: return 1
                return 0 if (to_call > 0 and 0 in legal_6) else 1
            else:
                # 미스 핸드
                if to_call == 0 and 1 in legal_6: return 1
                return 0 if 0 in legal_6 else 1

class RationalExploiter:
    """과도한 팟 베팅 및 무차별 올인을 트랩/응징하는 착취형 봇"""
    name = "Rational_Anti_Spam_Exploiter"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: List[int], priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        
        # 상대가 큰 팟베팅 또는 올인으로 압박할 때
        if to_call >= 500.0 or is_allin_facing:
            # 몬스터 핸드(Chen >= 10.0)로만 스택 전액 승부 (트랩 응징)
            if chen >= 10.0 and 1 in legal_6:
                return 1 # Trap call
            return 0 if 0 in legal_6 else 1 # 미련 없이 칼폴드 (상대 블러프 스팸에 칩 헌납 거부)

        if round_idx == 0:
            if chen >= 8.0:
                if 2 in legal_6 and self.rng.random() < 0.5: return 2
                return 1 if 1 in legal_6 else 0
            if to_call == 0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1
        else:
            if chen >= 8.5:
                if 3 in legal_6 and self.rng.random() < 0.4: return 3
                return 1 if 1 in legal_6 else 0
            if to_call == 0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1

class Aggressive3BetBot:
    """공격적 3-Bet 및 플랍 C-bet 압박 정책"""
    name = "Aggressive_3Bet_Pressure"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: List[int], priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        
        if is_allin_facing and to_call > 0:
            if chen >= 9.0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1

        if round_idx == 0:
            if to_call > 100.0: # Someone already raised -> 3-Bet chance!
                if chen >= 7.0 and 3 in legal_6:
                    return 3 # Aggressive 3-Bet (Pot)
                if chen >= 6.0 and 1 in legal_6:
                    return 1
                return 0 if 0 in legal_6 else 1
            else:
                if chen >= 7.0 and 2 in legal_6: return 2
                if chen >= 5.0 and 1 in legal_6: return 1
                return 0 if 0 in legal_6 else 1
        elif round_idx == 1: # Flop C-Bet
            if 2 in legal_6 and self.rng.random() < 0.65:
                return 2 # 65% C-Bet Half Pot
            return 1 if 1 in legal_6 else 0
        else:
            if chen >= 7.5 and 2 in legal_6: return 2
            return 1 if 1 in legal_6 else 0

class DynamicRandomBot:
    """미학습 신규 정책: 예측 불가능한 다양한 사이징으로 탐색 촉진"""
    name = "Dynamic_Exploratory_Random"
    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp: int, legal_6: List[int], priv_cards, round_idx: int, to_call: float, is_allin_facing: bool) -> int:
        if is_allin_facing and to_call > 0:
            # 80% 폴드, 20% 콜
            return 1 if (self.rng.random() < 0.20 and 1 in legal_6) else (0 if 0 in legal_6 else 1)
            
        weights = []
        for a in legal_6:
            if a == 0: weights.append(0.20)
            elif a == 1: weights.append(0.35)
            elif a == 2: weights.append(0.25)
            elif a == 3: weights.append(0.15)
            elif a == 4: weights.append(0.04)
            elif a == 5: weights.append(0.01)
            else: weights.append(0.01)
            
        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights]
        return int(self.rng.choices(legal_6, weights=norm_weights)[0])

def sample_league_opponent(seat: int, seed: int, pool_dir: str = "v1024/models_e2e"):
    """
    ChatGPT 승인 DIRECTIVE 041 리그 비율 샘플러:
    - 자가대련 30%
    - 균형형 정책 25%
    - 착취형 정책 20%
    - 공격적 3-Bet 15%
    - 미학습 신규 정책 10%
    """
    r = random.Random(seed).random()
    if r < 0.30:
        return SelfPlayLeagueBot(seat, seed, pool_dir=pool_dir)
    elif r < 0.55:
        return BalancedBot(seat, seed)
    elif r < 0.75:
        return RationalExploiter(seat, seed)
    elif r < 0.90:
        return Aggressive3BetBot(seat, seed)
    else:
        return DynamicRandomBot(seat, seed)
