"""
POKER AI V4.9.2 - 커리큘럼 리그 대국 관리자 (Curriculum League Manager)
- 정상 게임 규칙 100% 준수
- 재레이즈(3-Bet, 4-Bet) 및 포스트플롭 레이즈 상황에 에이전트가 자연스럽게 노출되도록
  능동적 베팅 성향의 봇(Aggressive, RuleBased, RigorousAdaptive)을 체계적으로 배치
"""

import random
from typing import List, Optional
import torch

from v48.opponents_pool import (
    BaseBot, RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot, RuleBasedBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v49.league_manager_5a import LearnerBotWrapper5A

class CurriculumLeagueManager5A:
    def __init__(self, device: torch.device = torch.device("cpu"), rng: Optional[random.Random] = None):
        self.device = device
        self.rng = rng if rng is not None else random.Random()
        
        # 적극적 재레이즈/베팅 풀 (Active Bet/Raise Bots)
        self.active_reraise_bots = [
            AggressiveBot(),
            RuleBasedBot(),
            RigorousAdaptiveBot()
        ]
        
        # 보조 풀 (Diverse Opponents)
        self.general_bots = [
            TightBot(),
            PremiumHunterBot(),
            AlwaysCallBot(),
            RandomBot(),
            SelectiveCallerBot(),
            BluffCatcherBot()
        ]

    def sample_table_opponents(self, learner_bot: LearnerBotWrapper5A) -> List[BaseBot]:
        """
        커리큘럼 상대 샘플링:
        - 4개 상대 좌석 중:
          1. 1개 좌석: 적극적 재레이즈 봇(Aggressive, RuleBased, RigorousAdaptive 중 1개) 필수 보장!
          2. 40% 확률로 1~2개 복제 Learner 배치 (상호 레이즈 학습)
          3. 나머지 좌석은 일반 상대 풀에서 무작위 선택
        """
        table_opps = []
        
        # 1. 적극적 재레이즈 봇 1개 이상 필수 배치
        table_opps.append(self.rng.choice(self.active_reraise_bots))
        
        # 2. 복제 Learner 배치 여부 결정
        r = self.rng.random()
        if r < 0.40:
            num_clones = self.rng.choice([1, 2])
            for _ in range(num_clones):
                if len(table_opps) < 4:
                    table_opps.append(learner_bot)
                    
        # 3. 빈 좌석 채우기 (50% 확률로 active_reraise_bots 또는 general_bots)
        combined_pool = self.active_reraise_bots + self.general_bots
        while len(table_opps) < 4:
            table_opps.append(self.rng.choice(combined_pool))
            
        self.rng.shuffle(table_opps)
        return table_opps
