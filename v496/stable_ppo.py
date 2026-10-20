"""
POKER AI V4.9.6 - Stable Pure On-Policy PPO Actor-Critic Module
- ε-greedy 탐색 완전 제거: 순수 On-Policy Masked Categorical 샘플링
- action ~ π_actor(a|s), old_log_prob = log π_actor(action|state)
- Importance Ratio = exp(new_log_prob - old_log_prob)
- PPO Clipped Objective with Entropy Bonus (- c_ent * Entropy)
- Orthogonal Initialization & Numerical Stability
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StableHoldemActorCritic5A(nn.Module):
    """5-Action No-Limit Texas Hold'em Stable PPO Model"""
    def __init__(
        self,
        input_dim: int = 265,
        hidden_dim: int = 256,
        action_dim: int = 5,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )

        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """PPO 표준 직교 초기화 (Orthogonal Initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor Head: gain=0.01 (초기 정책을 균등 분포에 가깝게 설정)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        # Critic Head: gain=1.0
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, input_dim]
        mask: [B, action_dim] (True: 합법 행동, False: 불법 행동)
        Returns: masked_logits [B, action_dim], value [B]
        """
        feats = self.encoder(x)
        raw_logits = self.actor_head(feats)
        value = self.critic_head(feats).squeeze(-1)

        # 불법 행동을 -1e9로 마스킹 (소프트맥스 시 정확히 0.0 보장)
        masked_logits = raw_logits.masked_fill(~mask, -1e9)
        return masked_logits, value

    def sample_action(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, np.ndarray, float]:
        """
        순수 On-Policy Masked Categorical 샘플링
        Returns: action, log_prob, probs_array, value
        """
        with torch.no_grad():
            masked_logits, value = self.forward(x, mask)
            probs = F.softmax(masked_logits, dim=-1).squeeze(0)
            log_probs = F.log_softmax(masked_logits, dim=-1).squeeze(0)

            legal_indices = torch.where(mask.squeeze(0))[0]
            if len(legal_indices) == 0:
                raise ValueError("No legal actions available in mask!")

            if deterministic:
                action = int(legal_indices[torch.argmax(probs[legal_indices])].item())
            else:
                # Masked Categorical Distribution
                dist = torch.distributions.Categorical(probs=probs)
                action_tensor = dist.sample()
                action = int(action_tensor.item())

            log_prob = float(log_probs[action].item())
            val = float(value.item())
            probs_np = probs.cpu().numpy()

        return action, log_prob, probs_np, val

def compute_stable_ppo_losses(
    model: StableHoldemActorCritic5A,
    states: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    수학적으로 올바른 PPO Clipped Loss 및 엔트로피 보너스 계산
    - L_total = L_clip + c_vf * L_vf - c_ent * Entropy
    - approx_kl = ((ratio - 1) - log(ratio)).mean()
    - clip_fraction = (ratio < 1-eps or ratio > 1+eps).mean()
    """
    masked_logits, pred_values = model(states, masks)

    # 1. New Log Probabilities & Importance Ratio
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    log_ratio = action_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 2. PPO Clipped Policy Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. Value Loss (MSE)
    value_loss = F.mse_loss(pred_values, returns)

    # 4. Entropy Bonus (Masked Safe Entropy)
    probs = F.softmax(masked_logits, dim=-1)
    # 0 * log(0) = 0 처리 (불법 행동 확률은 0이므로 마스킹 안전)
    safe_log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

    # 5. Total Loss
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    # 6. Diagnostics (KL, Clip Fraction)
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_frac = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean().item()

    return total_loss, policy_loss, value_loss, entropy, approx_kl, clip_frac
