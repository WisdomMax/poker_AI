"""
POKER AI V4.9 - 5-ACTION PPO Actor-Critic 신경망 및 손실 함수
- 구조:
  Input: Information State Tensor (265차원)
  Shared Encoder: Linear(265, 256) -> ReLU -> Linear(256, 128) -> ReLU
  Actor Head: Linear(128, 5) -> [Fold, Call, Pot, All-in, Half-Pot]
  Critic Head: Linear(128, 1) -> State Value V(s)
- Legal Action Mask: 합법 행동만 선택하도록 -1e9 마스킹
- 가중치 전이(Transfer Learning) 메서드 내장 (기존 4A 모델 가중치 흡수)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class HoldemActorCritic5A(nn.Module):
    def __init__(self, input_dim: int = 265, hidden_dim: int = 256, action_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Shared Encoder (265 -> 256 -> 128)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Policy Head (Actor)
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Value Head (Critic)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
        # 가중치 직교 초기화 (PPO 표준)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        logits = self.actor_head(feats)
        value = self.critic_head(feats).squeeze(-1)
        
        # Legal Action Masking
        masked_logits = logits.masked_fill(~mask, -1e9)
        return masked_logits, value

    def load_from_4a_model(self, m_4a: nn.Module):
        """
        기존 4-Action 모델(253->256->128, Actor 4, Critic 1)의 가중치를 5-Action 모델로 안전하게 전이:
        - 입력 레이어: 253개 차원은 기존 가중치 1:1 복사, 신규 12개 차원은 0으로 초기화
        - 은닉 레이어 및 Critic: 100% 동일 복사
        - Actor: 행동 0, 1, 2, 3은 1:1 복사, 행동 4(Half-Pot)는 행동 2(Pot)의 가중치를 복사하여 안정적인 베팅 탐색 시작
        """
        with torch.no_grad():
            # 1. 인코더 1번째 레이어 (Linear 265 -> 256)
            old_w0 = m_4a.encoder[0].weight # [256, 253]
            old_b0 = m_4a.encoder[0].bias   # [256]
            self.encoder[0].weight.data[:, :253] = old_w0
            self.encoder[0].weight.data[:, 253:] = 0.0 # 신규 12차원 0 초기화
            self.encoder[0].bias.data.copy_(old_b0)
            
            # 2. 인코더 2번째 레이어 (Linear 256 -> 128)
            self.encoder[2].weight.data.copy_(m_4a.encoder[2].weight.data)
            self.encoder[2].bias.data.copy_(m_4a.encoder[2].bias.data)
            
            # 3. Critic Head (Linear 128 -> 1)
            self.critic_head.weight.data.copy_(m_4a.critic_head.weight.data)
            self.critic_head.bias.data.copy_(m_4a.critic_head.bias.data)
            
            # 4. Actor Head (Linear 128 -> 5)
            # 행동 0..3 복사
            self.actor_head.weight.data[:4, :].copy_(m_4a.actor_head.weight.data)
            self.actor_head.bias.data[:4].copy_(m_4a.actor_head.bias.data)
            # 행동 4(Half-Pot) 초기화: 행동 2(Pot)와 동일한 방향성 부여
            self.actor_head.weight.data[4, :].copy_(m_4a.actor_head.weight.data[2, :])
            self.actor_head.bias.data[4].copy_(m_4a.actor_head.bias.data[2])

def compute_ppo_losses_5a(
    model: HoldemActorCritic5A,
    states: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.005
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO Clipped Objective Loss 계산"""
    logits, pred_values = model(states, masks)
    
    # Policy Loss (PPO Clip)
    log_probs = torch.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    ratio = torch.exp(action_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value Loss (MSE)
    value_loss = nn.functional.mse_loss(pred_values, returns)
    
    # Entropy Bonus
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
    
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total_loss, policy_loss, value_loss, entropy
