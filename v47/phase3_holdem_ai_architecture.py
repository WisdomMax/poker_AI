"""
POKER AI V4.7 - PHASE 3: 5인 Texas Hold'em 기본 AI 아키텍처 구축
- 입력: OpenSpiel 5인 홀덤 관측 텐서 (253차원)
  (개인 홀카드 2장, 공개 커뮤니티 카드, 팟, 플레이어별 스택 및 베팅, 라운드, 시퀀스 등 완벽 인코딩, 타인 카드 누출 0%)
- 출력: 4개 합법 액션 로짓 (FCPA: 0: Fold, 1: Call/Check, 2: Pot Raise, 3: All-in)
- 아키텍처:
  Linear(253, 256) -> ReLU -> LayerNorm -> Linear(256, 128) -> ReLU -> Linear(128, 4)
- Legal Action Masking 지원 (불법 행동 로짓을 -1e9로 차단)
"""

import sys
import os
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Holdem5PNet(nn.Module):
    def __init__(self, input_dim: int = 253, hidden_dim: int = 256, num_actions: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(x)
        if legal_mask is not None:
            # legal_mask: bool 텐서
            logits = logits.masked_fill(~legal_mask, -1e9)
        return logits
        
    def get_action_probs(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(x, legal_mask)
        return F.softmax(logits, dim=-1)

class Holdem5PAgent:
    """5인 홀덤 의사결정 에이전트 클래스"""
    def __init__(self, agent_id: int, model: Optional[Holdem5PNet] = None, device=torch.device("cpu")):
        self.agent_id = agent_id
        self.device = device
        self.model = model if model is not None else Holdem5PNet(253, 256, 4).to(device)
        
    def act(self, info_vec: List[float], legal_acts: List[int], deterministic: bool = False) -> Tuple[int, float, np.ndarray]:
        x = torch.tensor(info_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 4), dtype=torch.bool, device=self.device)
        for a in legal_acts: mask[0, a] = True
        
        with torch.no_grad():
            probs = self.model.get_action_probs(x, mask).squeeze(0).cpu().numpy()
            
        prs = [probs[a] for a in legal_acts]
        tot = sum(prs)
        if tot > 0:
            prs = np.array(prs) / tot
        else:
            prs = np.ones(len(legal_acts)) / len(legal_acts)
            
        if deterministic:
            chosen_act = legal_acts[int(np.argmax(prs))]
        else:
            chosen_act = int(np.random.choice(legal_acts, p=prs))
            
        log_prob = math.log(prs[legal_acts.index(chosen_act)] + 1e-12)
        return chosen_act, log_prob, probs

def test_architecture():
    print("=" * 80)
    print("🧠 [V4.7 PHASE 3] 5인 Texas Hold'em 신경망 아키텍처 단위 테스트")
    print("=" * 80)
    
    model = Holdem5PNet(253, 256, 4)
    dummy_x = torch.randn(2, 253)
    dummy_mask = torch.tensor([[True, True, True, False], [True, True, False, False]], dtype=torch.bool)
    
    logits = model(dummy_x, dummy_mask)
    probs = model.get_action_probs(dummy_x, dummy_mask)
    
    print(f"  * 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,} 개")
    print(f"  * 출력 텐서 형태: {logits.shape}")
    print(f"  * 마스킹된 액션 확률 (배치 0, 액션 3 마스킹): {probs[0].detach().numpy()}")
    print(f"  * 마스킹된 액션 확률 (배치 1, 액션 2,3 마스킹): {probs[1].detach().numpy()}")
    
    assert probs[0, 3].item() < 1e-6, "마스킹된 액션 3의 확률이 0이어야 합니다."
    assert probs[1, 2].item() < 1e-6, "마스킹된 액션 2의 확률이 0이어야 합니다."
    assert probs[1, 3].item() < 1e-6, "마스킹된 액션 3의 확률이 0이어야 합니다."
    print("✅ 신경망 아키텍처 및 불법 행동 마스킹 테스트 100% 통과!")
    print("=" * 80)

if __name__ == "__main__":
    test_architecture()
