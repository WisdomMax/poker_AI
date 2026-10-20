"""
POKER AI V5.0 - PHASE 3: Bayesian Opponent-Aware PPO Actor-Critic Module
- 입력 차원: 317차원 (기존 정규화 Information State 265차원 + 상대 4인 베이지안 통계 52차원)
- 상대 피처는 반드시 현재 판 시작 이전(과거 완료된 핸드)의 이력만 사용 (미래 정보 누출 0%)
- 상대 정보가 없는 조건(Cold Start 또는 대조군)을 위한 52차원 Neutral Prior Padding 제공
- 직교 초기화 (Orthogonal Init), Masked Categorical 샘플링, Pure On-Policy PPO
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v50.bayesian_opponent_tracker import SingleOpponentBayesianTracker, METRIC_NAMES

# 52차원 Cold Start 중립 사전분포 벡터 생성 함수
def get_neutral_prior_features() -> np.ndarray:
    """상대 정보가 없을 때 주입할 52차원 Neutral Beta(2,2) 사전분포 피처"""
    tracker = SingleOpponentBayesianTracker(agent_id=1, prior_alpha=2.0, prior_beta=2.0)
    single_feat = tracker.get_feature_vector() # 13차원
    return np.tile(single_feat, 4) # 13 * 4 = 52차원

class BayesianHoldemActorCritic5A(nn.Module):
    """317차원 입력을 처리하는 5-Action No-Limit Texas Hold'em 베이지안 PPO 모델"""
    def __init__(
        self,
        input_dim: int = 317,
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
        # Actor Head: gain=0.01
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        # Critic Head: gain=1.0
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 317] (265 info_state + 52 opponent_stats)
        mask: [B, 5] (True: 합법 행동, False: 불법 행동)
        Returns: masked_logits [B, 5], value [B]
        """
        feats = self.encoder(x)
        raw_logits = self.actor_head(feats)
        value = self.critic_head(feats).squeeze(-1)

        masked_logits = raw_logits.masked_fill(~mask, -1e9)
        return masked_logits, value

    def sample_action(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, np.ndarray, float]:
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
                dist = torch.distributions.Categorical(probs=probs)
                action_tensor = dist.sample()
                action = int(action_tensor.item())

            log_prob = float(log_probs[action].item())
            val = float(value.item())
            probs_np = probs.cpu().numpy()

        return action, log_prob, probs_np, val

def compute_bayesian_ppo_losses(
    model: BayesianHoldemActorCritic5A,
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
    masked_logits, pred_values = model(states, masks)

    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    log_ratio = action_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(pred_values, returns)

    probs = F.softmax(masked_logits, dim=-1)
    safe_log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_frac = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean().item()

    return total_loss, policy_loss, value_loss, entropy, approx_kl, clip_frac
