"""
POKER AI V4.4 - PHASE 5: Leduc Poker 4대 모델 vs 상대 8종 10회 독립 시드 대결
- 4대 모델:
  A. CFR+ 2,000 (공식 최적 기준선)
  B. Leduc NFSP (150k Best Checkpoint)
  C. Leduc Bayesian Exploiter
  D. Leduc NFSP + Safe Bayesian
- 8대 상대군:
  1. Random, 2. Rock, 3. Maniac, 4. Calling Station,
  5. Balanced, 6. Tricky, 7. Randomized, 8. Unseen Counter (신규 적대적 상대)
- 10개 독립 시드 Paired Test (동일 덱 순열, 공정 포지션 교대)
- 평균 보상, 95% CI, 최대 낙폭(MDD), 적응 속도 측정
"""

import sys
import os
import pickle
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper
from v44.phase4_leduc_bayesian import LeducSafeBayesianPlayer


# =====================================================================
# 1. 8대 평가 상대 봇 정의
# =====================================================================

class LeducBot:
    def __init__(self, name: str):
        self.name = name
    def act(self, state, cur_player, legal_acts) -> int:
        raise NotImplementedError

class RandomBot(LeducBot):
    def __init__(self): super().__init__("Random")
    def act(self, state, cur_player, legal_acts) -> int:
        return random.choice(legal_acts)

class RockBot(LeducBot):
    def __init__(self): super().__init__("Rock")
    def act(self, state, cur_player, legal_acts) -> int:
        # K(4,5)나 페어면 레이즈/콜, 아니면 체크/폴드
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
            return 1 if 1 in legal_acts else 0 # 1라운드는 슬로우플레이(체크/콜)
        # 2라운드는 적극적 레이즈
        return 2 if 2 in legal_acts else 1

class RandomizedBot(LeducBot):
    def __init__(self): super().__init__("Randomized")
    def act(self, state, cur_player, legal_acts) -> int:
        # 콜 위주(60%), 레이즈(30%), 폴드(10%)
        p_dist = []
        for a in legal_acts:
            w = 0.6 if a == 1 else (0.3 if a == 2 else 0.1)
            p_dist.append(w)
        p_dist = np.array(p_dist) / sum(p_dist)
        return int(np.random.choice(legal_acts, p=p_dist))

class UnseenCounterBot(LeducBot):
    """학습에 포함되지 않은 신규 적대적 봇: 강한 패는 숨기고 약한 패로만 공격하는 역착취 봇"""
    def __init__(self): super().__init__("Unseen Counter")
    def act(self, state, cur_player, legal_acts) -> int:
        info_str = state.information_state_string(cur_player)
        has_j = "[Private: 0]" in info_str or "[Private: 1]" in info_str
        if has_j: # 최약체 카드로 블러핑 레이즈
            return 2 if 2 in legal_acts else 1
        # 강한 카드는 트랩 체크
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


# =====================================================================
# 2. 실전 대전 시뮬레이션 엔진
# =====================================================================

def play_leduc_hand(game, p0_bot, p1_bot, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    state = game.new_initial_state()

    # 궤적 및 베팅 이력 추적
    p0_card, p1_card = None, None
    board_card = None
    opp_history_for_p0 = [] # list of (round_idx, act)
    opp_history_for_p1 = []

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, prs = zip(*outcomes)
            ch_act = int(np.random.choice(acts, p=prs))
            if p0_card is None: p0_card = ch_act
            elif p1_card is None: p1_card = ch_act
            else: board_card = ch_act
            state.apply_action(ch_act)
            continue

        cp = state.current_player()
        legal = state.legal_actions(cp)
        info_vec = state.information_state_tensor(cp)
        info_str = state.information_state_string(cp)
        r_idx = 1 if "[Round 2]" in info_str else 0

        if cp == 0:
            if isinstance(p0_bot, LeducSafeBayesianPlayer):
                act = p0_bot.act(info_vec, legal, p0_card, board_card, opp_history_for_p0)
            else:
                act = p0_bot.act(state, 0, legal) if isinstance(p0_bot, LeducBot) else p0_bot.act(info_vec, legal, is_evaluation=True)[0]
            opp_history_for_p1.append((r_idx, act))
        else:
            if isinstance(p1_bot, LeducSafeBayesianPlayer):
                act = p1_bot.act(info_vec, legal, p1_card, board_card, opp_history_for_p1)
            else:
                act = p1_bot.act(state, 1, legal) if isinstance(p1_bot, LeducBot) else p1_bot.act(info_vec, legal, is_evaluation=True)[0]
            opp_history_for_p0.append((r_idx, act))

        state.apply_action(act)

    returns = state.returns()
    pay0, pay1 = returns[0], returns[1]

    # 쇼다운 여부 (마지막 상태에서 둘 다 폴드하지 않았으면 쇼다운)
    is_showdown = (board_card is not None and returns[0] != 0)
    p0_shown = p0_card if is_showdown else None
    p1_shown = p1_card if is_showdown else None

    # 베이지안 업데이트
    if isinstance(p0_bot, LeducSafeBayesianPlayer):
        p0_bot.model.update_hand(opp_history_for_p0, p0_card, board_card, p1_shown)
    if isinstance(p1_bot, LeducSafeBayesianPlayer):
        p1_bot.model.update_hand(opp_history_for_p1, p1_card, board_card, p0_shown)

    return pay0, pay1


def evaluate_matchup_paired(game, agent_eval, opp_bot, num_hands: int, seed: int) -> Tuple[float, List[float]]:
    """동일 난수 시드에서 P0, P1 공정 교대 대결"""
    p0_prof = 0.0
    history = []
    
    # 50%는 P0, 50%는 P1로 출전
    for h in range(num_hands):
        hand_seed = seed * 100000 + h
        if h % 2 == 0:
            r0, _ = play_leduc_hand(game, agent_eval, opp_bot, hand_seed)
            p0_prof += r0
        else:
            _, r1 = play_leduc_hand(game, opp_bot, agent_eval, hand_seed)
            p0_prof += r1
        history.append(p0_prof)
        
    return p0_prof, history


def run_phase5():
    print("=" * 80)
    print("🔬 [V4.4 PHASE 5] Leduc Poker 4대 모델 vs 상대 8종 10회 독립 시드 대결")
    print("=" * 80)

    game = pyspiel.load_game("leduc_poker")

    # 1. 모델 준비
    # A. CFR+ 2000
    with open("v44/models/leduc_cfr_plus_2000.pkl", "rb") as f:
        cfr_dict = pickle.load(f)
    cfr_bot = TabularCFRBot(cfr_dict)

    # B. Leduc Best NFSP
    agent_nfsp = LeducNFSPAgent(0)
    agent_nfsp.avg_net.load_state_dict(torch.load("v44/models/v44_best_leduc_nfsp_p0.pt"))

    # C. NFSP + Safe Bayesian
    agent_hybrid = LeducSafeBayesianPlayer(agent_nfsp, max_exploit_weight=0.70)

    # 8대 상대군
    opponents = [
        RandomBot(),
        RockBot(),
        ManiacBot(),
        CallingStationBot(),
        TabularCFRBot(cfr_dict), # Balanced
        TrickyBot(),
        RandomizedBot(),
        UnseenCounterBot()
    ]

    seeds = list(range(701, 711)) # 10개 독립 시드
    num_hands = 2000 # 시드당 2,000판 (총 20,000판)

    print(f"\n▶ 10개 독립 시드(701~710)로 상대 8종 대상 1:1 Paired Test 진행 중...")

    results = {}
    for opp in opponents:
        print(f"\n▶ vs {opp.name} (10 Seeds Paired Test)...")
        nfsp_profits = []
        hybrid_profits = []
        diffs = []
        mdds_nfsp = []
        mdds_hybrid = []

        for s in seeds:
            # NFSP
            p_nfsp, hist_n = evaluate_matchup_paired(game, agent_nfsp, opp, num_hands, seed=s)
            nfsp_profits.append(p_nfsp)

            # Hybrid
            # 각 시드마다 새로운 베이지안 모델로 출발
            hybrid_bot = LeducSafeBayesianPlayer(agent_nfsp, max_exploit_weight=0.70)
            p_hyb, hist_h = evaluate_matchup_paired(game, hybrid_bot, opp, num_hands, seed=s)
            hybrid_profits.append(p_hyb)

            diffs.append(p_hyb - p_nfsp)

            def get_mdd(h):
                peak = h[0]; mdd = 0.0
                for v in h:
                    if v > peak: peak = v
                    if peak - v > mdd: mdd = peak - v
                return mdd
            mdds_nfsp.append(get_mdd(hist_n))
            mdds_hybrid.append(get_mdd(hist_h))

        # 통계
        n_m = np.mean(nfsp_profits) / (num_hands / 100.0)
        h_m = np.mean(hybrid_profits) / (num_hands / 100.0)
        d_m = np.mean(diffs) / (num_hands / 100.0)
        d_sem = (np.std(diffs, ddof=1) / np.sqrt(len(seeds))) / (num_hands / 100.0)
        ci = 2.262 * d_sem

        results[opp.name] = {
            "nfsp_bb": n_m, "hybrid_bb": h_m, "diff_bb": d_m, "ci": ci,
            "mdd_nfsp": np.mean(mdds_nfsp), "mdd_hybrid": np.mean(mdds_hybrid)
        }
        print(f"  * vs {opp.name:<16}: Pure NFSP = {n_m:>+6.2f} bb/100 | Hybrid = {h_m:>+6.2f} bb/100 | Diff = {d_m:>+6.2f} ± {ci:.2f} bb/100")

    print("\n" + "=" * 85)
    print("📊 [Leduc Poker 8대 상대군 10개 독립 시드 종합 검증표 (Mean ± 95% CI)]")
    print("=" * 85)
    print(f"{'상대 유형':<20} | {'Pure NFSP (bb/100)':<20} | {'NFSP + Safe Bayesian':<22} | {'착취 개선폭 (Diff)':<18}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<20} | {r['nfsp_bb']:>+7.2f} bb/100       | {r['hybrid_bb']:>+7.2f} bb/100         | {r['diff_bb']:>+7.2f} ± {r['ci']:.2f} bb")
    print("=" * 85)


if __name__ == "__main__":
    run_phase5()
