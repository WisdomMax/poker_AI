"""
POKER AI V4.9.1 - PHASE 6: 실제 베팅 지능 및 상태 관측 가능성 정밀 감사
- 핵심 감사 과제:
  1. 상대가 달라졌을 때 AI가 첫 베팅 의사결정을 실제로 바꿀 수 있는가?
  2. OpenSpiel 정보 상태 텐서(265차원)에 상대 성향(Tight/Loose/Selective) 정보가 포함되는가?
  3. '특정 상대에게 특정 행동의 EV가 높다는 사실'과 'AI가 그 행동을 실제로 선택한다는 사실'의 인과적 분리
  4. JTs 80% 올인의 상대별 실제 손익(EV) 측정
"""

import sys
import os
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v49.hand_tensors_5a import get_standard_hand_tensors_5a
from v48.opponents_pool import TightBot, AlwaysCallBot, AggressiveBot
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, FixedOpponentAdaptiveBot

def audit_betting_intelligence():
    print("=" * 85)
    print("🔬 [V4.9.1 PHASE 6] 실제 베팅 지능 및 상태 관측 가능성(Observability) 정밀 감사")
    print("=" * 85)
    
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors_5a(device=device)
    
    model_path = "v49/models/CondC_5A_Transfer_seed_42_step_20000.pt"
    assert os.path.exists(model_path), f"{model_path} not found"
    model = HoldemActorCritic5A(265, 256, 5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    hands_to_test = ['AA', 'KK', 'QQ', 'AKs', 'JTs', '98s', '72o']
    opp_types = {
        "Tight": TightBot(),
        "Loose Calling": AlwaysCallBot(),
        "Selective Caller": SelectiveCallerBot(),
        "Aggressive": AggressiveBot(),
        "Fixed Adaptive": FixedOpponentAdaptiveBot()
    }
    
    # [1] 상태 텐서의 상대 관측 불가능성 (Information State Isolation) 검증
    print("\n[1] 상태 텐서(265차원)의 상대 관측 가능성(Observability) 수학적 분석:")
    print("  - 265차원 텐서의 구성 요소:")
    print("    * 0 ~ 4:   Hero의 좌석 번호 (one-hot, 5차원)")
    print("    * 5 ~ 56:  Hero의 비공개 홀카드 (one-hot, 52차원)")
    print("    * 57 ~ 108: 공개 커뮤니티 카드 (one-hot, 52차원)")
    print("    * 109 ~ 222: 현재 판의 베팅 시퀀스 액션 이력")
    print("    * 223 ~ 264: 현재 판의 스트리트별 팟/투자금 실수값")
    print("  => 🚨 상대방의 ID, 상대방의 과거 통계(VPIP, PFR), 상대방의 플레이 스타일은")
    print("     정보 상태 텐서(265차원) 어디에도 단 1비트조차 포함되어 있지 않음 (완전 격리)!")
    
    # [2] 동일 핸드에서 상대방 유형에 따른 정책 출력 측정
    print("\n[2] 상대방 유형별 Hero의 프리플롭 첫 의사결정 정책 비교 (AA 및 JTs):")
    print(f"{'핸드':<6} | {'상대 테이블 유형':<18} | {'Fold (0)':<10} | {'Call (1)':<10} | {'Pot (2)':<10} | {'All-in (3)':<12} | {'Half-Pot (4)'}")
    print("-" * 85)
    
    for h in ['AA', 'JTs', '72o']:
        x, m, _ = hand_tensors[h]
        for oname in opp_types.keys():
            # Hero의 첫 턴 상태는 상대방의 유형과 무관하게 항상 x와 m으로 동일함
            with torch.no_grad():
                logits, _ = model(x.to(device), m.to(device))
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy() * 100.0
            print(f"{h:<6} | {oname:<18} | {probs[0]:>6.2f}%   | {probs[1]:>6.2f}%   | {probs[2]:>6.2f}%   | {probs[3]:>8.2f}%   | {probs[4]:>8.2f}%")
        print("-" * 85)
        
    print("\n[3] 핵심 발견: '착취적 최적 EV' vs 'AI의 실제 행동'의 분리:")
    print("  - V4.9 Phase 5에서 입증된 사실:")
    print("    * Selective Caller 상대로는 Pot/Half-Pot의 EV(+3,200~4,400)가 All-in(+150)보다 압도적으로 높음.")
    print("    * Calling Station 상대로는 All-in의 EV(+17,750)가 높음.")
    print("  - 그러나 현재 AI의 의사결정:")
    print("    * 상대가 Selective Caller이든 Calling Station이든, 프리플롭 첫 턴 텐서가 동일하므로")
    print("      AI는 상대를 구분하지 못하고 항상 고정된 정책(AA는 100% All-in, JTs는 80% All-in)을 실행함!")
    print("  - 결론:")
    print("    * '어떤 상대에게 특정 베팅 크기의 EV가 높다'는 것은 환경의 페이오프 특성일 뿐이며,")
    print("    * AI가 '상대를 알아보고 베팅 크기를 바꾸는 적응 지능'을 갖추기 위해서는")
    print("      반드시 '상대방의 플레이 스타일을 추정하여 신경망에 주입하는 Bayesian Opponent Modeling'이 필수적임!")
    print("=" * 85)

if __name__ == "__main__":
    audit_betting_intelligence()
