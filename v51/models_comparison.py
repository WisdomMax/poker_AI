"""
POKER AI V5.1 - PHASE 4: OPPONENT-AWARE ARCHITECTURE COMPARISON MODELS
- 공정 비교를 위한 3대 아키텍처 구현:
  1. Model A (Concatenation Baseline):
     Hero(265D) + Opp(52D) = 317D -> 256 -> 128 -> Heads (약 115k params)
  2. Model B (Two-Tower MLP):
     Hero Tower(265 -> 224 -> 112) & Opp Tower(52 -> 48 -> 32)
     -> Concat(144) -> Fusion(176) -> Heads (약 115.4k params, 오차 0.3%)
  3. Model C (FiLM Context Gating):
     Hero Stream(265 -> 248 -> 128) & Opp Conditioner(52 -> 48 -> Gamma/Beta 256)
     -> Feature-wise Linear Modulation: h = (1 + gamma) * h_hero + beta -> Heads (약 113.7k params, 오차 1.2%)
- 모든 모델은 동일한 Orthogonal 초기화(Actor gain=0.01) 및 Legal Action Masking 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_actor_head(layer):
    nn.init.orthogonal_(layer.weight, gain=0.01)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def init_critic_head(layer):
    nn.init.orthogonal_(layer.weight, gain=1.0)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ConcatenationActorCritic(nn.Module):
    """Model A: 표준 조기 결합 (Early Fusion / Concatenation)"""
    def __init__(self, hero_dim: int = 265, opp_dim: int = 52, hidden_1: int = 256, hidden_2: int = 128, num_actions: int = 5):
        super().__init__()
        self.hero_dim = hero_dim
        self.opp_dim = opp_dim
        self.input_dim = hero_dim + opp_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_2, num_actions)
        self.critic_head = nn.Linear(hidden_2, 1)

        self.apply(init_weights)
        init_actor_head(self.actor_head)
        init_critic_head(self.critic_head)

    def forward(
        self,
        hero_tensor: torch.Tensor,
        opp_tensor: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([hero_tensor, opp_tensor], dim=-1)
        h = self.net(x)
        raw_logits = self.actor_head(h)
        value = self.critic_head(h)

        if legal_actions_mask is not None:
            masked_logits = raw_logits.clone()
            masked_logits[~legal_actions_mask] = -1e9
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(raw_logits, dim=-1)

        return raw_logits, probs, value


class TwoTowerActorCritic(nn.Module):
    """Model B: 독립 인코더 결합 (Late Fusion / Two-Tower MLP)"""
    def __init__(
        self,
        hero_dim: int = 265,
        opp_dim: int = 52,
        hero_h1: int = 224,
        hero_h2: int = 112,
        opp_h1: int = 48,
        opp_h2: int = 32,
        fusion_dim: int = 176,
        num_actions: int = 5
    ):
        super().__init__()
        self.hero_dim = hero_dim
        self.opp_dim = opp_dim

        # Hero Tower
        self.hero_tower = nn.Sequential(
            nn.Linear(hero_dim, hero_h1),
            nn.ReLU(),
            nn.Linear(hero_h1, hero_h2),
            nn.ReLU()
        )

        # Opponent Tower
        self.opp_tower = nn.Sequential(
            nn.Linear(opp_dim, opp_h1),
            nn.ReLU(),
            nn.Linear(opp_h1, opp_h2),
            nn.ReLU()
        )

        # Fusion Layer
        combined_dim = hero_h2 + opp_h2
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(fusion_dim, num_actions)
        self.critic_head = nn.Linear(fusion_dim, 1)

        self.apply(init_weights)
        init_actor_head(self.actor_head)
        init_critic_head(self.critic_head)

    def forward(
        self,
        hero_tensor: torch.Tensor,
        opp_tensor: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_hero = self.hero_tower(hero_tensor)
        h_opp = self.opp_tower(opp_tensor)
        h_comb = torch.cat([h_hero, h_opp], dim=-1)
        h_fused = self.fusion(h_comb)

        raw_logits = self.actor_head(h_fused)
        value = self.critic_head(h_fused)

        if legal_actions_mask is not None:
            masked_logits = raw_logits.clone()
            masked_logits[~legal_actions_mask] = -1e9
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(raw_logits, dim=-1)

        return raw_logits, probs, value


class FiLMActorCritic(nn.Module):
    """Model C: 조건부 피처 모듈레이션 (FiLM: Feature-wise Linear Modulation)"""
    def __init__(
        self,
        hero_dim: int = 265,
        opp_dim: int = 52,
        hero_h1: int = 248,
        hero_h2: int = 128,
        opp_cond_h1: int = 48,
        num_actions: int = 5
    ):
        super().__init__()
        self.hero_dim = hero_dim
        self.opp_dim = opp_dim
        self.hero_h2 = hero_h2

        # Hero Stream
        self.hero_l1 = nn.Linear(hero_dim, hero_h1)
        self.hero_l2 = nn.Linear(hero_h1, hero_h2)

        # Opponent Conditioner (출력: gamma(128) + beta(128) = 256)
        self.opp_cond = nn.Sequential(
            nn.Linear(opp_dim, opp_cond_h1),
            nn.ReLU(),
            nn.Linear(opp_cond_h1, hero_h2 * 2)
        )

        self.actor_head = nn.Linear(hero_h2, num_actions)
        self.critic_head = nn.Linear(hero_h2, 1)

        self.apply(init_weights)
        init_actor_head(self.actor_head)
        init_critic_head(self.critic_head)

        # FiLM 파라미터 초기화: 초기에 gamma=0, beta=0으로 시작하여 Hero 기본 정책을 왜곡하지 않음
        nn.init.zeros_(self.opp_cond[-1].weight)
        nn.init.zeros_(self.opp_cond[-1].bias)

    def forward(
        self,
        hero_tensor: torch.Tensor,
        opp_tensor: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hero Stream
        h = F.relu(self.hero_l1(hero_tensor))
        h = F.relu(self.hero_l2(h))

        # FiLM Conditioning: gamma, beta
        film_params = self.opp_cond(opp_tensor)
        gamma, beta = torch.split(film_params, self.hero_h2, dim=-1)

        # FiLM modulation: h_mod = (1 + gamma) * h + beta
        h_mod = (1.0 + gamma) * h + beta
        h_out = F.relu(h_mod)

        raw_logits = self.actor_head(h_out)
        value = self.critic_head(h_out)

        if legal_actions_mask is not None:
            masked_logits = raw_logits.clone()
            masked_logits[~legal_actions_mask] = -1e9
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(raw_logits, dim=-1)

        return raw_logits, probs, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    mA = ConcatenationActorCritic()
    mB = TwoTowerActorCritic()
    mC = FiLMActorCritic()

    pA = count_parameters(mA)
    pB = count_parameters(mB)
    pC = count_parameters(mC)

    print("=" * 65)
    print("V5.1 ARCHITECTURE COMPARISON - PARAMETER COUNT AUDIT")
    print("=" * 65)
    print(f"Model A (Concatenation): {pA:,} parameters (Baseline)")
    print(f"Model B (Two-Tower):     {pB:,} parameters (Diff: {pB - pA:+,} / {(pB - pA)/pA*100:+.2f}%)")
    print(f"Model C (FiLM):          {pC:,} parameters (Diff: {pC - pA:+,} / {(pC - pA)/pA*100:+.2f}%)")
    print("-" * 65)

    # Dummy forward pass test
    hero_dummy = torch.randn(2, 265)
    opp_dummy = torch.randn(2, 52)
    mask_dummy = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])

    for name, model in [("Model A", mA), ("Model B", mB), ("Model C", mC)]:
        r_log, probs, val = model(hero_dummy, opp_dummy, mask_dummy)
        assert probs.shape == (2, 5), f"{name} probs shape mismatch"
        assert val.shape == (2, 1), f"{name} val shape mismatch"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2)), f"{name} probs do not sum to 1"
        assert (probs[0, 2:] == 0.0).all(), f"{name} masking violation"
        print(f"[{name}] Forward & Masking Check: PASSED")
    print("=" * 65)
