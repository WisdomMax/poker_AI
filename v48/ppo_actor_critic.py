"""
POKER AI V4.8 - PHASE 4: PPO Actor-Critic 신경망 및 손실 함수 구현
- 공통 Feature Encoder (253 -> 256 -> 128)
- Policy Head (Actor): 4개 액션 로짓 (합법 행동 마스킹 지원)
- Value Head (Critic): 스칼라 가치 V(s) 추정 (상대 비공개 카드 누출 없이 동일한 253차원 관측만 사용)
- PPO Clipped Objective (eps=0.2), Value Loss (MSE), Entropy Bonus (beta=0.01)
- Gradient Clipping (norm=1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import numpy as np

class HoldemActorCritic(nn.Module):
    """
    5인 텍사스 홀덤 PPO Actor-Critic 신경망
    - input_dim: 253 (OpenSpiel universal_poker FCPA information state tensor)
    - action_dim: 4 (0: Fold, 1: Check/Call, 2: Bet/Raise, 3: All-in)
    """
    def __init__(self, input_dim: int = 253, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        
        # 공통 Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor Head (Policy Logits)
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic Head (Value function V(s))
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, 253]
        mask: [batch, 4] (True for legal actions)
        반환: (masked_logits, value)
        """
        feats = self.encoder(x)
        logits = self.actor_head(feats)
        
        # 불법 행동 마스킹 (-1e9)
        masked_logits = logits.masked_fill(~mask, -1e9)
        value = self.critic_head(feats).squeeze(-1)
        return masked_logits, value
        
    def get_action_and_value(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        행동 샘플링 또는 주어진 행동에 대한 log_prob, entropy, value 계산
        """
        masked_logits, value = self.forward(x, mask)
        probs = F.softmax(masked_logits, dim=-1)
        
        # 엔트로피 계산 (합법 행동에 대해서만)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        # 0 * log(0) 방지
        entropy = -torch.sum(probs * torch.nan_to_num(log_probs, nan=0.0, neginf=0.0), dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob, entropy, value

def compute_ppo_losses(
    model: HoldemActorCritic,
    b_states: torch.Tensor,
    b_masks: torch.Tensor,
    b_actions: torch.Tensor,
    b_old_log_probs: torch.Tensor,
    b_returns: torch.Tensor,
    b_advantages: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PPO 손실 함수 계산:
    - policy_loss: Clipped Surrogate Objective
    - value_loss: MSE(V(s), Return)
    - entropy_loss: -Entropy
    """
    _, new_log_probs, entropy, new_values = model.get_action_and_value(b_states, b_masks, b_actions)
    
    # 1. Policy Loss (Clipped Objective)
    ratio = torch.exp(new_log_probs - b_old_log_probs)
    surr1 = ratio * b_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 2. Value Loss
    value_loss = F.mse_loss(new_values, b_returns)
    
    # 3. Entropy Regularization (엔트로피 최대화)
    entropy_loss = -entropy.mean()
    
    total_loss = policy_loss + (vf_coef * value_loss) + (ent_coef * entropy_loss)
    return total_loss, policy_loss, value_loss, entropy.mean()
