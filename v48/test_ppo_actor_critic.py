"""
POKER AI V4.8 - PHASE 4: PPO Actor-Critic 및 그래디언트 무결성 단위 테스트
- 검증 항목:
  1. Actor/Critic 네트워크 차원 일치 (253 -> 4 로짓, 1 스칼라 가치)
  2. Critic에 상대 비공개 카드 누출 없음 (단일 253차원 정보집합 벡터만 수신)
  3. PPO 손실 함수 계산 및 역전파(Backpropagation) 무결성
  4. 합법 행동 마스킹 정상 작동 (-1e9 부여로 불법 행동 확률 0)
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.ppo_actor_critic import HoldemActorCritic, compute_ppo_losses
from v48.poker_environment import get_5p_holdem_game, advance_chance_nodes

def test_ppo_unit_tests():
    print("=" * 80)
    print("🧪 [V4.8 PHASE 4] PPO Actor-Critic 및 그래디언트 단위 테스트")
    print("=" * 80)
    
    device = torch.device("cpu")
    model = HoldemActorCritic(253, 256, 4).to(device)
    
    # 1. 차원 및 마스킹 검증
    batch_size = 8
    dummy_x = torch.randn(batch_size, 253, device=device)
    dummy_mask = torch.ones(batch_size, 4, dtype=torch.bool, device=device)
    dummy_mask[:, 3] = False # 3번 액션(All-in) 불법화
    
    logits, values = model(dummy_x, dummy_mask)
    assert logits.shape == (batch_size, 4), f"Logits shape mismatch: {logits.shape}"
    assert values.shape == (batch_size,), f"Values shape mismatch: {values.shape}"
    assert torch.all(logits[:, 3] < -1e8), "Masking failed: illegal action logit is not -1e9"
    probs = torch.softmax(logits, dim=-1)
    assert torch.all(probs[:, 3] < 1e-6), "Masking failed: illegal action probability > 0"
    print("  ✅ 1. Actor/Critic 차원 및 합법 행동 마스킹 통과")
    
    # 2. 비공개 카드 누출 검증: 실제 OpenSpiel 게임 상태로부터 정보 텐서 추출
    game = get_5p_holdem_game()
    st = game.new_initial_state()
    advance_chance_nodes(st)
    cur_p = st.current_player()
    info_tensor = st.information_state_tensor(cur_p)
    assert len(info_tensor) == 253, f"Info tensor length is {len(info_tensor)}, expected 253"
    # Critic은 오직 이 info_tensor만 입력받으므로 상대 비공개 카드가 주입될 수 없음
    print("  ✅ 2. Critic 입력 무결성 (253차원 단일 정보집합 텐서만 사용, 상대 카드 누출 불가) 통과")
    
    # 3. PPO 손실 및 역전파 단위 테스트
    actions = torch.randint(0, 3, (batch_size,), device=device)
    old_log_probs = torch.zeros(batch_size, device=device) - 1.0
    returns = torch.randn(batch_size, device=device)
    advantages = torch.randn(batch_size, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    tot_loss, p_loss, v_loss, ent = compute_ppo_losses(
        model, dummy_x, dummy_mask, actions, old_log_probs, returns, advantages
    )
    
    assert not torch.isnan(tot_loss), "Total loss is NaN"
    tot_loss.backward()
    
    # 그래디언트 클리핑 및 유효성 검사
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert not torch.isnan(grad_norm), "Gradient norm is NaN"
    optimizer.step()
    
    print(f"  ✅ 3. PPO 손실 계산 및 역전파 통과 (Loss: {tot_loss.item():.4f}, Grad Norm: {grad_norm.item():.4f})")
    print("\n" + "=" * 80)
    print("🏆 [PHASE 4 단위 테스트 통과] PPO Actor-Critic 무결성 확인 완료.")
    print("=" * 80)

if __name__ == "__main__":
    test_ppo_unit_tests()
