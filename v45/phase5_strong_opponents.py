"""
POKER AI V4.5 - PHASE 5: 10개 독립 시드 대칭 벤치마크 (Strong Opponents Benchmark)
- 평가 대상:
  1. Pure Distilled Student (NashConv 0.008725의 순수 GTO 신경망)
  2. Safe Bayesian v2 (Student + Entropy-Gated Bayesian 착취 하이브리드)
- 상대 봇 8종:
  1. Random, 2. Rock, 3. Maniac, 4. Calling Station, 5. Tricky, 6. Randomized, 7. Unseen Counter, 8. CFR+ 2000
- 10개 독립 시드 (각 시드당 1,000판, 포지션 교대 대칭 각 500판, 상대당 총 10,000판)
- 통계 지표: 총 칩, chips/hand, chips/100, bb/100 (1 bb = 2 chips), 95% 신뢰구간(CI), Student t-test
"""

import sys
import os
import time
import random
import pickle
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch

import pyspiel

# 단위 및 모델 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from units import calculate_metrics, format_metrics, CHIPS_PER_BB
from phase2_distillation import LeducStudentNet
from phase4_safe_bayesian_v2 import LeducSafeBayesianPlayerV2

# 8대 상대 봇 정의
class LeducBot:
    def __init__(self, name: str): self.name = name
    def act(self, state, cur_player, legal_acts) -> int: raise NotImplementedError

class RandomBot(LeducBot):
    def __init__(self): super().__init__("Random")
    def act(self, state, cur_player, legal_acts) -> int: return random.choice(legal_acts)

class RockBot(LeducBot):
    def __init__(self): super().__init__("Rock")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        has_k = "[Private: 4]" in info_str or "[Private: 5]" in info_str
        has_pair = ("Pair" in info_str) or ("[Public: 4]" in info_str and has_k)
        if has_k or has_pair:
            return 2 if 2 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

class ManiacBot(LeducBot):
    def __init__(self): super().__init__("Maniac")
    def act(self, state, cur_player, legal_acts) -> int:
        return 2 if 2 in legal_acts else 1

class CallingStationBot(LeducBot):
    def __init__(self): super().__init__("Calling Station")
    def act(self, state, cur_player, legal_acts) -> int:
        return 1 if 1 in legal_acts else 0

class TrickyBot(LeducBot):
    def __init__(self): super().__init__("Tricky")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        is_round2 = "[Round 2]" in info_str
        if not is_round2:
            return 1 if 1 in legal_acts else 0
        return 2 if 2 in legal_acts else 1

class RandomizedBot(LeducBot):
    def __init__(self): super().__init__("Randomized")
    def act(self, state, cur_player, legal_acts) -> int:
        p_dist = []
        for a in legal_acts:
            w = 0.6 if a == 1 else (0.3 if a == 2 else 0.1)
            p_dist.append(w)
        p_dist = np.array(p_dist) / sum(p_dist)
        return int(np.random.choice(legal_acts, p=p_dist))

class UnseenCounterBot(LeducBot):
    def __init__(self): super().__init__("Unseen Counter")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        has_j = "[Private: 0]" in info_str or "[Private: 1]" in info_str
        if has_j:
            return 2 if 2 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

class TabularCFRBot(LeducBot):
    def __init__(self, policy_dict):
        super().__init__("CFR+ 2000")
        self.policy_dict = policy_dict
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        probs_map = self.policy_dict.get(info_str, {a: 1.0/len(legal_acts) for a in legal_acts})
        actions = list(probs_map.keys())
        p_vals = [probs_map[a] for a in actions]
        p_vals = np.array(p_vals) / sum(p_vals)
        return int(np.random.choice(actions, p=p_vals))

def extract_cards_and_history(state, target_player):
    opp_player = 1 - target_player
    my_card, opp_card, board_card = None, None, None
    my_str = state.information_state_string(target_player)
    
    for c in range(6):
        if f"[Private: {c}]" in my_str:
            my_card = c
            break
            
    if state.is_terminal():
        opp_str = state.information_state_string(opp_player)
        for c in range(6):
            if f"[Private: {c}]" in opp_str:
                opp_card = c
                break
                
    for c in range(6):
        if f"[Public: {c}]" in my_str:
            board_card = c
            break
            
    opp_history = []
    return my_card, opp_card, board_card, opp_history

def play_match(game, ai_type: str, student_model, safe_bayesian, opponent_bot, num_hands: int = 1000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if ai_type == "safe_bayesian":
        safe_bayesian.reset_model()
        
    ai_total_chips = 0.0
    half = num_hands // 2
    
    for h in range(num_hands):
        ai_pos = 0 if h < half else 1
        opp_pos = 1 - ai_pos
        
        state = game.new_initial_state()
        opp_acts_hand = []
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, prs = zip(*outcomes)
                ch_act = np.random.choice(acts, p=prs)
                state.apply_action(ch_act)
                continue
                
            cur_p = state.current_player()
            legal_acts = state.legal_actions(cur_p)
            info_vec = state.information_state_tensor(cur_p)
            
            if cur_p == ai_pos:
                if ai_type == "pure_student":
                    x = torch.tensor(info_vec, dtype=torch.float32).unsqueeze(0)
                    mask = torch.zeros((1, 3), dtype=torch.bool)
                    for a in legal_acts: mask[0, a] = True
                    with torch.no_grad():
                        probs = student_model.get_action_probs(x, mask).squeeze(0).numpy()
                    prs = [probs[a] for a in legal_acts]
                    prs = np.array(prs) / sum(prs)
                    act = int(np.random.choice(legal_acts, p=prs))
                elif ai_type == "safe_bayesian":
                    my_card, _, board_card, _ = extract_cards_and_history(state, ai_pos)
                    act = safe_bayesian.act(info_vec, legal_acts, my_card, board_card, opp_acts_hand)
                else:
                    raise ValueError(f"Unknown ai_type: {ai_type}")
            else:
                act = opponent_bot.act(state, cur_p, legal_acts)
                info_str = state.information_state_string(cur_p)
                r_idx = 1 if "[Round 2]" in info_str else 0
                opp_acts_hand.append((r_idx, act))
                
            state.apply_action(act)
            
        returns = state.returns()
        ai_chip = returns[ai_pos]
        ai_total_chips += ai_chip
        
        if ai_type == "safe_bayesian":
            my_card, opp_card_shown, board_card, _ = extract_cards_and_history(state, ai_pos)
            # 폴드한 경우 상대 카드는 비공개 (None)
            is_showdown = (len(state.legal_actions()) == 0 and "[Player: " not in state.information_state_string(ai_pos))
            # returns 절대값이나 라운드로 쇼다운 여부 판정: Leduc에서는 둘 다 폴드하지 않았으면 쇼다운
            # history에서 0(Fold)이 없으면 쇼다운
            history_acts = [a for a in state.history() if a in [0, 1, 2]]
            had_fold = (0 in history_acts)
            shown_card = opp_card_shown if not had_fold else None
            safe_bayesian.model.update_hand(opp_acts_hand, my_card, board_card, shown_card)
            
    return ai_total_chips

def run_phase5_benchmark():
    print("=" * 80)
    print("🥊 [V4.5 PHASE 5] 10개 독립 시드 대칭 벤치마크 (Pure Student vs Safe Bayesian v2)")
    print("=" * 80)
    
    game = pyspiel.load_game("leduc_poker")
    
    # 1. 모델 로드
    student_model = LeducStudentNet(30, 128, 3)
    student_model.load_state_dict(torch.load("v45/models/leduc_student_best.pt", map_location="cpu"))
    student_model.eval()
    
    safe_bayesian = LeducSafeBayesianPlayerV2(student_model, max_exploit_weight=0.65)
    
    # CFR+ 봇 로드
    with open("v44/models/leduc_cfr_plus_2000.pkl", "rb") as f:
        cfr_dict = pickle.load(f)
    cfr_bot = TabularCFRBot(cfr_dict)
    
    opponents = [
        RandomBot(),
        RockBot(),
        ManiacBot(),
        CallingStationBot(),
        TrickyBot(),
        RandomizedBot(),
        UnseenCounterBot(),
        cfr_bot
    ]
    
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
    num_hands_per_seed = 1000
    
    results = {}
    
    for opp in opponents:
        print(f"\n▶ 상대: {opp.name} (10개 시드 x 1,000판 = 총 10,000판 대결 중)...")
        student_chips_list = []
        bayesian_chips_list = []
        
        for s in seeds:
            sc = play_match(game, "pure_student", student_model, safe_bayesian, opp, num_hands_per_seed, seed=s)
            bc = play_match(game, "safe_bayesian", student_model, safe_bayesian, opp, num_hands_per_seed, seed=s)
            student_chips_list.append(sc)
            bayesian_chips_list.append(bc)
            
        student_chips_arr = np.array(student_chips_list)
        bayesian_chips_arr = np.array(bayesian_chips_list)
        
        # 시드당 1,000판의 bb/100 배열 (1 bb = 2 chips)
        student_bb100 = (student_chips_arr / num_hands_per_seed) * 100.0 / CHIPS_PER_BB
        bayesian_bb100 = (bayesian_chips_arr / num_hands_per_seed) * 100.0 / CHIPS_PER_BB
        
        mean_s_bb = np.mean(student_bb100)
        mean_b_bb = np.mean(bayesian_bb100)
        
        ci_s = 1.96 * np.std(student_bb100, ddof=1) / np.sqrt(len(seeds))
        ci_b = 1.96 * np.std(bayesian_bb100, ddof=1) / np.sqrt(len(seeds))
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(bayesian_bb100, student_bb100)
        
        results[opp.name] = {
            "student_mean_bb": mean_s_bb,
            "student_ci_bb": ci_s,
            "bayesian_mean_bb": mean_b_bb,
            "bayesian_ci_bb": ci_b,
            "delta_bb": mean_b_bb - mean_s_bb,
            "p_val": p_val,
            "student_tot_chips": np.sum(student_chips_arr),
            "bayesian_tot_chips": np.sum(bayesian_chips_arr),
        }
        
        print(f"  * Pure Student: {mean_s_bb:+.2f} ± {ci_s:.2f} bb/100 (총 {np.sum(student_chips_arr):+.1f} chips)")
        print(f"  * Safe Bayesian v2: {mean_b_bb:+.2f} ± {ci_b:.2f} bb/100 (총 {np.sum(bayesian_chips_arr):+.1f} chips)")
        print(f"  * 차이 (Bayesian 개선폭): {mean_b_bb - mean_s_bb:+.2f} bb/100 (p-value: {p_val:.4f})")
        
    print("\n" + "=" * 90)
    print("📊 [V4.5 최종 8대 상대 종합 벤치마크 비교표 (10개 독립 시드, 총 80,000판)]")
    print("=" * 90)
    print(f"{'상대 봇':<16} | {'Pure Student (bb/100)':<22} | {'Safe Bayesian v2 (bb/100)':<25} | {'Bayesian 개선폭':<14} | {'p-value'}")
    print("-" * 90)
    for name, r in results.items():
        s_str = f"{r['student_mean_bb']:+.2f} ± {r['student_ci_bb']:.2f}"
        b_str = f"{r['bayesian_mean_bb']:+.2f} ± {r['bayesian_ci_bb']:.2f}"
        diff_str = f"{r['delta_bb']:+.2f}"
        sig = "***" if r['p_val'] < 0.001 else ("**" if r['p_val'] < 0.01 else ("*" if r['p_val'] < 0.05 else "n.s."))
        print(f"{name:<16} | {s_str:<22} | {b_str:<25} | {diff_str:<14} | {r['p_val']:.4f} ({sig})")
    print("=" * 90)
    
    # 벤치마크 결과 저장
    with open("v45/benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("💾 벤치마크 결과 저장 완료: v45/benchmark_results.pkl")

if __name__ == "__main__":
    run_phase5_benchmark()
