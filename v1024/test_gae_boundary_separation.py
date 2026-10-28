#!/usr/bin/env python3
"""
v1024/test_gae_boundary_separation.py
Unit test verifying that GAE advantages NEVER leak across distinct poker hand boundaries.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def compute_corrected_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrected GAE with strict episode boundary masking:
    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    gae_t   = delta_t + gamma * gae_lambda * (1 - done_t) * gae_{t+1}
    returns_t = gae_t + V(s_t)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        # If done[t] == 1, the episode terminates here!
        # Next state belongs to an independent new hand, so bootstrap value is ZERO and GAE accumulator resets!
        non_terminal = 1.0 - dones[t].item()
        delta = rewards[t] + gamma * non_terminal * next_val - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

def test_hand_boundary_independence():
    print("Testing Hand Boundary Independence in GAE...")
    
    # Hand 1: 2 decisions, rewards [0.0, +10.0 BB], dones [0, 1]
    # Hand 2: 2 decisions, rewards [0.0, -100.0 BB], dones [0, 1]
    rewards = torch.tensor([0.0, 10.0, 0.0, -100.0], dtype=torch.float32)
    values = torch.tensor([1.0, 2.0, -5.0, -10.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    
    adv_combined, ret_combined = compute_corrected_gae(rewards, values, dones)
    
    # Run Hand 1 independently
    adv_h1, ret_h1 = compute_corrected_gae(rewards[:2], values[:2], dones[:2])
    # Run Hand 2 independently
    adv_h2, ret_h2 = compute_corrected_gae(rewards[2:], values[2:], dones[2:])
    
    # Assert exact mathematical equality
    assert torch.allclose(adv_combined[:2], adv_h1, atol=1e-6), "Hand 1 advantage leaked into or from Hand 2!"
    assert torch.allclose(adv_combined[2:], adv_h2, atol=1e-6), "Hand 2 advantage leaked into or from Hand 1!"
    assert torch.allclose(ret_combined[:2], ret_h1, atol=1e-6), "Hand 1 returns leaked!"
    assert torch.allclose(ret_combined[2:], ret_h2, atol=1e-6), "Hand 2 returns leaked!"
    
    print("  -> Hand 1 Adv:", adv_combined[:2].tolist(), "vs Independent:", adv_h1.tolist())
    print("  -> Hand 2 Adv:", adv_combined[2:].tolist(), "vs Independent:", adv_h2.tolist())
    print("  [PASS] GAE Episode Boundary Masking is 100% mathematically independent across hands!")

if __name__ == "__main__":
    from typing import Tuple
    test_hand_boundary_independence()
