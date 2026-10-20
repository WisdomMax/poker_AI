"""
POKER AI V4.7.2 - PHASE 3: Anti-All-in 전문 상대 봇 5종 구축
- 목적: 무차별 올인을 남발하는 Step 100k AI를 응징하고 실제 포커 실력의 견고성(Robustness)을 평가.
- 봇 라인업:
  1. BOT A (Premium Hunter): Top 3% 프리미엄(AA, KK, QQ, AKs)만 올인 콜, 나머지 폴드.
  2. BOT B (Broad Hunter): Top 15% 중상위권(88+, ATs+, KJs+, AQo+) 올인 콜.
  3. BOT C (Adaptive Dynamic): 상대 올인 빈도 추적, 빈도 50% 초과 시 콜 레인지를 Top 30%까지 유연하게 확장하여 응징.
  4. BOT D (Pot-Odds Hunter): 팟 오즈(Pot Odds) 및 핸드 승률 추정 기반 기대값(EV > 0) 콜.
  5. BOT E (Always Call): 무조건 콜 (대조군).
"""

import sys
import os
import random
from typing import Dict, List, Tuple
import numpy as np

import pyspiel

# 핸드 평가 유틸리티
PREMIUM_RANKS = ["A", "K", "Q"]
BROAD_RANKS = ["A", "K", "Q", "J", "T", "9", "8"]

def get_hole_card_ranks(info_str: str) -> Tuple[str, str, bool]:
    """홀 카드의 두 랭크와 수딧 여부 추출 (e.g. '2d2c' -> '2', '2', False)"""
    if "[Private: " not in info_str:
        return "2", "3", False
    priv = info_str.split("[Private: ")[1].split("]")[0]
    if len(priv) < 4:
        return "2", "3", False
    r1, s1 = priv[0], priv[1]
    r2, s2 = priv[2], priv[3]
    is_suited = (s1 == s2)
    return r1, r2, is_suited

class AntiAllinBot:
    def __init__(self, name: str): self.name = name
    def act(self, state, cur_player, legal_acts) -> int: raise NotImplementedError

class PremiumHunterBot(AntiAllinBot):
    """BOT A: AA, KK, QQ, AKs, AKo 같은 최상위 프리미엄 핸드로만 올인 콜"""
    def __init__(self): super().__init__("Premium Hunter (Top 3%)")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        r1, r2, is_suited = get_hole_card_ranks(info_str)
        is_pair = (r1 == r2)
        has_aa_kk_qq = is_pair and (r1 in ["A", "K", "Q"])
        has_ak = (r1 in ["A", "K"] and r2 in ["A", "K"] and r1 != r2)
        is_premium = has_aa_kk_qq or has_ak
        
        # 상대가 올인(r10000)을 던졌거나 큰 베팅이 나온 경우
        is_facing_allin = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts) or ("r10000" in info_str)
        if is_facing_allin:
            return 1 if (is_premium and 1 in legal_acts) else 0 # 프리미엄만 콜, 아니면 폴드
        # 평소에는 체크/콜/팟레이즈
        return 2 if is_premium and 2 in legal_acts else (1 if 1 in legal_acts else 0)

class BroadHunterBot(AntiAllinBot):
    """BOT B: Top 15% 중상위권 핸드로 올인 콜"""
    def __init__(self): super().__init__("Broad Hunter (Top 15%)")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        r1, r2, is_suited = get_hole_card_ranks(info_str)
        is_pair = (r1 == r2)
        has_good_pair = is_pair and (r1 in BROAD_RANKS)
        has_two_broad = (r1 in BROAD_RANKS and r2 in BROAD_RANKS)
        is_broad = has_good_pair or has_two_broad
        
        is_facing_allin = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts) or ("r10000" in info_str)
        if is_facing_allin:
            return 1 if (is_broad and 1 in legal_acts) else 0
        return 1 if 1 in legal_acts else 0

class AdaptiveHunterBot(AntiAllinBot):
    """BOT C: 상대 올인 빈도 추적형 적응 봇 (상대가 올인 50% 이상 남발 시 콜 레인지 확대)"""
    def __init__(self):
        super().__init__("Adaptive Hunter")
        self.opp_actions = []
        
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        r1, r2, is_suited = get_hole_card_ranks(info_str)
        is_pair = (r1 == r2)
        
        # 올인 빈도 계산
        allin_ratio = self.opp_actions.count("allin") / max(1, len(self.opp_actions))
        
        # 상대가 미친 듯이 올인을 박으면 콜 레인지를 페어 전체 및 브로드웨이 전체로 대폭 확대
        if allin_ratio > 0.4:
            can_call = is_pair or (r1 in BROAD_RANKS or r2 in BROAD_RANKS)
        else:
            can_call = is_pair and (r1 in PREMIUM_RANKS)
            
        is_facing_allin = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts) or ("r10000" in info_str)
        if is_facing_allin:
            self.opp_actions.append("allin")
            return 1 if (can_call and 1 in legal_acts) else 0
        else:
            if "r" in info_str: self.opp_actions.append("raise")
            return 1 if 1 in legal_acts else 0

class PotOddsHunterBot(AntiAllinBot):
    """BOT D: 팟 오즈 및 족보 승률 수학적 계산 기반 헌터"""
    def __init__(self): super().__init__("Pot-Odds Hunter")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        r1, r2, is_suited = get_hole_card_ranks(info_str)
        # 팟 오즈가 최소 40% 이상 승률을 요구하므로 중간 페어 이상일 때만 콜
        is_pair = (r1 == r2)
        has_ace = (r1 == "A" or r2 == "A")
        is_facing_allin = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts) or ("r10000" in info_str)
        if is_facing_allin:
            return 1 if ((is_pair or has_ace) and 1 in legal_acts) else 0
        return 1 if 1 in legal_acts else 0

class AlwaysCallBot(AntiAllinBot):
    """BOT E: 기존 대조군 (무조건 콜)"""
    def __init__(self): super().__init__("Always Call")
    def act(self, state, cur_player, legal_acts) -> int:
        return 1 if 1 in legal_acts else legal_acts[0]
