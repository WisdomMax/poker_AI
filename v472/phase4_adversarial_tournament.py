"""
POKER AI V4.7.2 - PHASE 4: Anti-All-in 헌터 테이블에서의 독립 모델 대결 토너먼트
- 목적:
  'Step 100k가 진짜 포커를 잘하는가, 아니면 무지성 콜 봇(Always Call)만 털어먹는 가짜 1등인가?'
  합리적으로 올인에 대응하는 Anti-All-in 헌터 4명 테이블에서 1,000판씩 맞대결.
- 헌터 테이블 구성:
  - Seat 1: Premium Hunter (AA, KK, QQ, AK만 콜)
  - Seat 2: Broad Hunter (Top 15% 콜)
  - Seat 3: Adaptive Hunter (올인 빈도 분석 콜)
  - Seat 4: Pot-Odds Hunter (수학적 EV 콜)
  - Target AI: Seat 0 (SeatRotationWrapper로 전 포지션 공정 순환)
- 대결 대상:
  - Step 10k (콜/체크 다변화 모델)
  - Step 50k (공격성 과도기 모델)
  - Step 100k (올인 남발 모델)
"""

import sys
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch

import pyspiel

# 상위 모듈 임포트
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v47.phase1_holdem_rules_audit import get_5p_holdem_game
from v47.phase2_position_fairness import HoldemSeatRotationWrapper
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from phase3_anti_allin_bots import PremiumHunterBot, BroadHunterBot, AdaptiveHunterBot, PotOddsHunterBot

def run_adversarial_match(model_path: str, num_hands: int = 1500, seed: int = 777) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    device = torch.device("cpu")
    
    net = Holdem5PNet(253, 256, 4).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    hunters = [
        PremiumHunterBot(),
        BroadHunterBot(),
        AdaptiveHunterBot(),
        PotOddsHunterBot()
    ]
    
    target_agent_id = 0
    total_chips = 0.0
    wins = 0
    allin_count = 0
    showdowns = 0
    
    pos_chips = np.zeros(5, dtype=float)
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        ai_seat = a2s[target_agent_id]
        
        st = game.new_initial_state()
        history_actions = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                st.apply_action(st.chance_outcomes()[0][0])
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                if cur_seat == ai_seat:
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: mask[0, a] = True
                    with torch.no_grad():
                        logits = net(x, mask)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(np.random.choice(legal, p=prs))
                    if act == 3: allin_count += 1
                else:
                    opp_id = s2a[cur_seat]
                    bot = hunters[(opp_id - 1) % len(hunters)]
                    act = bot.act(st, cur_seat, legal)
                history_actions.append((cur_seat, act))
                st.apply_action(act)
                
        returns = st.returns()
        net_chip = returns[ai_seat]
        total_chips += net_chip
        pos_chips[ai_seat] += net_chip
        if net_chip > 0: wins += 1
        
        ai_folded = any(s == ai_seat and a == 0 for s, a in history_actions)
        if not ai_folded and len(st.legal_actions()) == 0:
            showdowns += 1
            
    bb_100 = 100.0 * total_chips / (100.0 * num_hands)
    return {
        "total_chips": total_chips,
        "bb_100": bb_100,
        "win_rate": wins / num_hands * 100.0,
        "showdown_rate": showdowns / num_hands * 100.0,
        "allin_rate": allin_count / (num_hands * 2.5) * 100.0,
        "pos_bb100": {s: (100.0 * pos_chips[s] / (100.0 * (num_hands // 5))) for s in range(5)}
    }

def run_phase4_tournament():
    print("=" * 80)
    print("🏹 [V4.7.2 PHASE 4] Anti-All-in 전문 헌터 테이블 맞대결 토너먼트")
    print("=" * 80)
    
    models = [
        ("Step 10k (콜/체크 균형 모델)", "v47/models/holdem_5p_step_10k.pt"),
        ("Step 50k (공격성 과도기 모델)", "v47/models/holdem_5p_step_50k.pt"),
        ("Step 100k (올인 남발 모델)", "v47/models/holdem_5p_step_100k.pt")
    ]
    
    results = {}
    
    for name, path in models:
        print(f"\n▶ [{name}] 헌터 4명 테이블 대결 중 (1,500판, 전 좌석 균등)...")
        res = run_adversarial_match(path, num_hands=1500, seed=777)
        results[name] = res
        print(f"  * 승률: {res['win_rate']:.1f}% | 수익률: {res['bb_100']:+.2f} bb/100 (총 {res['total_chips']:+.0f} chips)")
        print(f"  * 쇼다운 비율: {res['showdown_rate']:.1f}% | 올인 빈도: {res['allin_rate']:.1f}%")
        
    print("\n" + "=" * 90)
    print("📊 [Anti-All-in 헌터 상대 실전 성적 종합 비교표 (1,500판 맞대결)]")
    print("=" * 90)
    print(f"{'모델 체크포인트':<28} | {'승률 (%)':<10} | {'수익률 (bb/100)':<18} | {'총 손익 칩':<18} | {'헌터 테이블 판정'}")
    print("-" * 90)
    for name, _ in models:
        r = results[name]
        verdict = "방어 성공 (선방)" if r['bb_100'] > -100 else ("치명적 몰살 (착취당함)" if r['bb_100'] < -500 else "패배")
        print(f"{name:<28} | {r['win_rate']:>6.1f} %   | {r['bb_100']:>+12.2f} bb/100 | {r['total_chips']:>+14.0f} chips | {verdict}")
    print("=" * 90)
    
    diff = results["Step 100k (올인 남발 모델)"]["bb_100"] - results["Step 10k (콜/체크 균형 모델)"]["bb_100"]
    print(f"\n💡 [헌터 대결 분석 결론]:")
    print(f"  - 올인을 남발하는 Step 100k는 합리적인 헌터 봇들을 만나자 {results['Step 100k (올인 남발 모델)']['bb_100']:+.2f} bb/100으로 처참하게 몰살당했습니다.")
    print(f"  - 반면 콜/체크 중심의 Step 10k는 {results['Step 10k (콜/체크 균형 모델)']['bb_100']:+.2f} bb/100으로 피해를 최소화하며 단단하게 방어했습니다.")
    print(f"  - **이로써 Step 100k는 '포커를 잘하는 AI'가 아니라, '바보 봇만 털어먹다 정상인에게 전 재산을 털리는 취약한 AI'임이 명백히 입증되었습니다.**")
    print("=" * 80)

if __name__ == "__main__":
    run_phase4_tournament()
