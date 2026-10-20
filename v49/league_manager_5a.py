"""
POKER AI V4.9 - 5-Action 리그 대국 관리자 (League Manager 5A)
- 5-Action(fchpa) 환경에 최적화된 상대 풀 샘플링 및 복제 Learner 지원
- 휴리스틱 봇 풀 (Random, Always Call, Tight, Aggressive, Premium Hunter, Selective Caller, Bluff Catcher)
- LearnerBotWrapper5A: HoldemActorCritic5A 신경망 정책 기반 의사결정
"""

import os
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import pyspiel

from v48.opponents_pool import (
    BaseBot, RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import (
    SelectiveCallerBot, BluffCatcherBot, FixedOpponentAdaptiveBot
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A

class LearnerBotWrapper5A(BaseBot):
    """5-Action PPO 신경망 모델을 리그 봇 인터페이스로 감싸는 래퍼"""
    def __init__(self, model: HoldemActorCritic5A, device: torch.device = torch.device("cpu")):
        super().__init__("Learner5A")
        self.model = model
        self.device = device
        
    def act(self, state: pyspiel.State, cur_player: int, legal_acts: List[int]) -> int:
        vec = state.information_state_tensor(cur_player)
        x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 5), dtype=torch.bool, device=self.device)
        for a in legal_acts: mask[0, a] = True
        
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x, mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        prs = [probs[a] for a in legal_acts]
        p_sum = sum(prs)
        prs = np.array(prs) / p_sum if p_sum > 0 else np.ones(len(legal_acts)) / len(legal_acts)
        return int(np.random.choice(legal_acts, p=prs))

class LeagueManager5A:
    def __init__(self, device: torch.device = torch.device("cpu"), rng: Optional[random.Random] = None):
        self.device = device
        self.rng = rng if rng is not None else random.Random()
        
        self.pool_bots = [
            RandomBot(),
            AlwaysCallBot(),
            TightBot(),
            AggressiveBot(),
            PremiumHunterBot(),
            SelectiveCallerBot(),
            BluffCatcherBot()
        ]
        
    def sample_table_opponents(self, learner_bot: LearnerBotWrapper5A) -> List[BaseBot]:
        """
        테이블의 4개 상대 좌석을 샘플링:
        - 40% 확률로 1~2개의 복제 Learner(Cloned Learner) 배치 (자가대국 및 전환 수집 가속)
        - 나머지 좌석은 다양한 스타일의 휴리스틱 상대 봇으로 무작위 배치
        """
        table_opps = []
        r = self.rng.random()
        
        if r < 0.40:
            num_clones = self.rng.choice([1, 2])
            for _ in range(num_clones):
                table_opps.append(learner_bot)
                
        while len(table_opps) < 4:
            table_opps.append(self.rng.choice(self.pool_bots))
            
        self.rng.shuffle(table_opps)
        return table_opps
