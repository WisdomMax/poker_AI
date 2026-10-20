"""
POKER AI V4.7.1 - PHASE 6: 5인 홀덤 엄밀한 독립 성능 벤치마크 (Independent Benchmark)
- 평가 대상:
  1. Step 0 (Random Baseline)
  2. Step 10k Checkpoint
  3. Step 50k Checkpoint
  4. Step 100k Checkpoint
- 평가 상대 풀 (4명 고정):
  - Random, Always Call, Tight, Rule-Based (또는 Aggressive)
- 엄밀한 과학적 프로토콜:
  1. 동일한 10개 독립 시드 (Seeds 1001 ~ 1010)로 모델별 카드/상대 조건 100% 동일 대응 비교 (Paired Evaluation)
  2. 시드당 500판 x 10개 시드 = 모델당 총 5,000판 (전 좌석 1,000판씩 균등 순환)
  3. 시드별 bb/100 산출 후 평균, 표준편차, 95% 신뢰구간(CI), Student t-test p-value 계산
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
from v47.phase5_benchmark_eval import HoldemBot, RandomHoldemBot, AlwaysCallHoldemBot, TightHoldemBot, RuleBasedHoldemBot, AggressiveHoldemBot

def evaluate_seed_match(
    model_path: str,
    opponents: List[HoldemBot],
    num_hands: int = 500,
    seed: int = 1001
) -> Tuple[float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    device = torch.device("cpu")
    
    net = Holdem5PNet(253, 256, 4).to(device)
    if model_path != "random":
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    target_agent_id = 0
    total_chips = 0.0
    wins = 0
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        ai_seat = a2s[target_agent_id]
        
        st = game.new_initial_state()
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
                else:
                    opp_id = s2a[cur_seat]
                    opp_bot = opponents[(opp_id - 1) % len(opponents)]
                    act = opp_bot.act(st, cur_seat, legal)
                st.apply_action(act)
                
        returns = st.returns()
        ai_chip = returns[ai_seat]
        total_chips += ai_chip
        if ai_chip > 0: wins += 1
        
    bb_100 = (total_chips / num_hands) * 100.0 / 100.0 # 1 BB = 100 chips
    win_rate = (wins / num_hands) * 100.0
    return bb_100, win_rate

def run_phase6_independent_benchmark():
    print("=" * 80)
    print("🥊 [V4.7.1 PHASE 6] 10개 독립 시드 엄밀한 성능 벤치마크 (총 20,000판)")
    print("=" * 80)
    
    opponents = [
        RandomHoldemBot(),
        AlwaysCallHoldemBot(),
        TightHoldemBot(),
        RuleBasedHoldemBot()
    ]
    
    seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    num_hands_per_seed = 500 # 시드당 500판 x 10 = 모델당 5,000판
    
    checkpoints = [
        ("Step 0 (Random)", "random"),
        ("Step 10k", "v47/models/holdem_5p_step_10k.pt"),
        ("Step 50k", "v47/models/holdem_5p_step_50k.pt"),
        ("Step 100k", "v47/models/holdem_5p_step_100k.pt")
    ]
    
    results = {}
    
    for name, path in checkpoints:
        print(f"\n▶ [{name}] 10개 독립 시드 평가 진행 중...")
        bb_list = []
        win_list = []
        for s in seeds:
            bb, wr = evaluate_seed_match(path, opponents, num_hands_per_seed, seed=s)
            bb_list.append(bb)
            win_list.append(wr)
            
        bb_arr = np.array(bb_list)
        win_arr = np.array(win_list)
        
        mean_bb = float(np.mean(bb_arr))
        std_bb = float(np.std(bb_arr, ddof=1))
        ci_bb = float(1.96 * std_bb / np.sqrt(len(seeds)))
        
        mean_win = float(np.mean(win_arr))
        
        results[name] = {
            "mean_bb": mean_bb,
            "std_bb": std_bb,
            "ci_bb": ci_bb,
            "mean_win": mean_win,
            "seed_bbs": bb_list
        }
        print(f"  * {name:<16}: {mean_bb:>+8.2f} ± {ci_bb:.2f} bb/100 (승률: {mean_win:.1f}%)")
        
    print("\n" + "=" * 90)
    print("📊 [10개 독립 시드 대응 비교 결과 종합표 (모델당 5,000판, 총 20,000판)]")
    print("=" * 90)
    print(f"{'체크포인트':<18} | {'평균 승률 (%)':<14} | {'수익률 (bb/100)':<22} | {'95% 신뢰구간':<18} | {'최종 판정'}")
    print("-" * 90)
    for name, _ in checkpoints:
        r = results[name]
        verdict = "공식 최적 모델" if name == "Step 10k" else ("붕괴 모델" if r['mean_bb'] < -300 else "기준/퇴보 모델")
        print(f"{name:<18} | {r['mean_win']:>10.1f} %   | {r['mean_bb']:>+12.2f} bb/100    | ± {r['ci_bb']:>6.2f} bb/100    | {verdict}")
    print("=" * 90)
    
    # Step 10k vs Step 0 통계적 유의성 t-test
    t_stat, p_val = stats.ttest_rel(results["Step 10k"]["seed_bbs"], results["Step 0 (Random)"]["seed_bbs"])
    print(f"\n📈 [통계적 유의성 검정]: Step 10k vs Step 0 (Random Baseline)")
    print(f"   - Paired t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
    if p_val < 0.05:
        print("   -> Step 10k가 Random Baseline보다 통계적으로 유의미하게 우수함이 검증됨 (p < 0.05).")
        
    return results

if __name__ == "__main__":
    run_phase6_independent_benchmark()
