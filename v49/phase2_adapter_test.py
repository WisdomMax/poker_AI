"""
POKER AI V4.9 - PHASE 2: 기존 4-Action 모델 호환 어댑터 정밀 단위 테스트
- 검증 대상 모델:
  - v483/models/seed_42_step_20000.pt
  - v483/models/seed_202_step_20000.pt
- 검증 내용:
  1. 265차원 텐서가 253차원으로 정확히 투영되는지 확인
  2. 5차원 마스크에서 Half-Pot(행동 4)의 선택 확률이 0.0000%인지 확인
  3. Fold, Call, Pot, All-in 선택 확률이 기존 4-Action 모델의 순전파와 100% 일치하는지 검증
  4. 5-Action 실제 게임 100판 시뮬레이션에서 무결하게 완주하는지 확인
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.ppo_actor_critic import HoldemActorCritic
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes
from v49.model_adapter_4a_to_5a import FourActionToFiveActionAdapter, FourActionAdapterBot, map_fchpa_to_fcpa_tensor

def test_adapter_compatibility():
    print("=" * 85)
    print("🔬 [V4.9 PHASE 2] 4-Action 모델 -> 5-Action 환경 호환 어댑터 단위 테스트")
    print("=" * 85)
    
    device = torch.device("cpu")
    checkpoints = [
        "v483/models/seed_42_step_20000.pt",
        "v483/models/seed_202_step_20000.pt"
    ]
    
    for ckpt_path in checkpoints:
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        print(f"\n검증 체크포인트: {ckpt_path}")
        
        # 1. 4-Action 모델 로드
        m_4a = HoldemActorCritic(253, 256, 4).to(device)
        m_4a.load_state_dict(torch.load(ckpt_path, map_location=device))
        m_4a.eval()
        
        # 2. 어댑터 생성
        adapter = FourActionToFiveActionAdapter(m_4a).to(device)
        adapter.eval()
        
        # 3. 임의의 5-Action 텐서 및 마스크 생성
        x_5a = torch.randn(1, 265, device=device)
        m_5a = torch.tensor([[True, True, True, True, True]], dtype=torch.bool, device=device)
        
        # 어댑터 순전파
        with torch.no_grad():
            logits_5a, val_5a = adapter(x_5a, m_5a)
            probs_5a = torch.softmax(logits_5a, dim=-1).squeeze(0).cpu().numpy()
            
            # 기준 4A 모델 순전파
            x_4a = map_fchpa_to_fcpa_tensor(x_5a)
            logits_4a, val_4a = m_4a(x_4a, m_5a[:, :4])
            probs_4a = torch.softmax(logits_4a, dim=-1).squeeze(0).cpu().numpy()
            
        print(f"  - 원본 4A 확률: {[round(float(p), 4) for p in probs_4a]}")
        print(f"  - 어댑터 5A 확률: {[round(float(p), 4) for p in probs_5a]}")
        
        # 검증: Half-Pot(행동 4) 확률이 정확히 0.0인가?
        assert probs_5a[4] < 1e-7, f"Half-Pot probability must be ~0, got {probs_5a[4]}"
        
        # 검증: 0..3 행동의 상대 확률이 원본과 정확히 일치하는가?
        ratio_diff = np.max(np.abs(probs_5a[:4] - probs_4a))
        assert ratio_diff < 1e-5, f"Probability mismatch between 4A and 5A adapter: {ratio_diff}"
        print(f"  ✓ 4대 행동 확률 오차: {ratio_diff:.2e} (완벽 일치)")
        print(f"  ✓ Half-Pot 확률: {probs_5a[4]:.2e} (선택 완전 차단 확인)")
        
        # 4. 실제 5-Action 게임 100판 시뮬레이션
        game = get_5p_holdem_5a_game()
        bot = FourActionAdapterBot(adapter, device=device)
        rng = np.random.RandomState(42)
        
        all_passed_sim = True
        actions_chosen = {a: 0 for a in range(5)}
        
        for g_idx in range(100):
            st = game.new_initial_state()
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng=rng)
                else:
                    p = st.current_player()
                    legal = st.legal_actions(p)
                    if p == 0:
                        act = bot.act(st, p, legal)
                        actions_chosen[act] += 1
                    else:
                        act = legal[0] # 체크/콜/폴드
                    st.apply_action(act)
                    
        assert actions_chosen[4] == 0, f"Adapter chose HalfPot during simulation: {actions_chosen}"
        print(f"  ✓ 100판 게임 완주: 선택 행동 통계 = {actions_chosen} (HalfPot 0회 보장)")

    print("\n" + "=" * 85)
    print("🎉 PHASE 2 호환 어댑터 단위 테스트 100% 통과!")
    print("=" * 85)

if __name__ == "__main__":
    test_adapter_compatibility()
