"""
POKER AI V4.8 - PHASE 3: 리그 대국 관리자 (League Training Manager)
- 목적: 동일 AI 폐쇄형 자기대국에서 벗어나, 다양한 스타일의 상대 및 과거 체크포인트와 대결하도록 상대 풀을 동적 구성.
- 리그 구성원 (League Pool):
  1. Main Learner (현재 학습 중인 정책)
  2. Past Checkpoint (Step 10k, Step 50k 등 고정 신경망)
  3. Tight Bot
  4. Aggressive Bot
  5. Always Call Bot
  6. Premium Hunter Bot
  7. Broad Hunter Bot
  8. Random Bot
- 일반화 검증용 격리 봇 (Unseen / Hold-out Opponents):
  - Adaptive Hunter Bot (훈련 시 절대 노출하지 않음)
  - Rule-Based Bot (훈련 시 절대 노출하지 않음)
"""

import os
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import pyspiel

from v48.opponents_pool import (
    BaseBot, RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot, BroadHunterBot, AdaptiveHunterBot, RuleBasedBot
)
from v47.phase3_holdem_ai_architecture import Holdem5PNet

class CheckpointBot(BaseBot):
    """과거 체크포인트 가중치를 로드하여 고정 정책으로 플레이하는 봇"""
    def __init__(self, name: str, model_path: str, device: torch.device):
        super().__init__(name)
        self.device = device
        self.net = Holdem5PNet(253, 256, 4).to(device)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        
    def act(self, state: pyspiel.State, cur_player: int, legal_acts: List[int]) -> int:
        vec = state.information_state_tensor(cur_player)
        x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 4), dtype=torch.bool, device=self.device)
        for a in legal_acts: mask[0, a] = True
        with torch.no_grad():
            logits = self.net(x, mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        prs = [probs[a] for a in legal_acts]
        prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal_acts))/len(legal_acts)
        return int(np.random.choice(legal_acts, p=prs))

class LeagueManager:
    """
    5인 테이블의 4개 상대 좌석을 확률 분포에 따라 리그 풀에서 샘플링.
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        
        # 1. 고정 봇 풀 (휴리스틱 & 과거 체크포인트)
        self.rule_bots = [
            RandomBot(),
            AlwaysCallBot(),
            TightBot(),
            AggressiveBot(),
            PremiumHunterBot(),
            BroadHunterBot()
        ]
        
        # 과거 체크포인트 추가
        self.checkpoint_bots = []
        if os.path.exists("v47/models/holdem_5p_step_10k.pt"):
            self.checkpoint_bots.append(CheckpointBot("Step 10k Bot", "v47/models/holdem_5p_step_10k.pt", device))
        if os.path.exists("v47/models/holdem_5p_step_50k.pt"):
            self.checkpoint_bots.append(CheckpointBot("Step 50k Bot", "v47/models/holdem_5p_step_50k.pt", device))
            
        # 2. 리그 구성 비율 (Sampling Probabilities)
        # - Self-Play (Main Learner 복제): 40%
        # - Past Checkpoint Bot: 20%
        # - Heuristic / Rule-based Bot: 40%
        self.self_play_prob = 0.40
        self.checkpoint_prob = 0.20
        self.rule_bot_prob = 0.40
        
    def sample_table_opponents(self, current_learner_bot: BaseBot) -> List[BaseBot]:
        """
        4명의 상대를 리그 풀에서 샘플링하여 반환.
        """
        opponents = []
        for _ in range(4):
            r = random.random()
            if r < self.self_play_prob or not self.checkpoint_bots:
                opponents.append(current_learner_bot)
            elif r < (self.self_play_prob + self.checkpoint_prob) and self.checkpoint_bots:
                opponents.append(random.choice(self.checkpoint_bots))
            else:
                opponents.append(random.choice(self.rule_bots))
        return opponents
