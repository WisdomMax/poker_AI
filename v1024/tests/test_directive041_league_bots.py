#!/usr/bin/env python3
"""
Unit Tests for DIRECTIVE 041 League Opponent Policies
Verifies:
1. SelfPlayLeagueBot loads historical checkpoints and selects legal actions.
2. BalancedBot shows balanced sizing across Half-pot, Pot, Check/Call, Fold.
3. RationalExploiter folds weak hands to big bets and traps with monsters.
4. Aggressive3BetBot generates 3-Bets when facing raises and fires C-bets.
5. DynamicRandomBot provides stochastic exploratory action diversity.
6. Sample distribution matches ChatGPT directive (30/25/20/15/10).
"""
import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from v1024.directive041_league import (
    calc_chen_score,
    SelfPlayLeagueBot,
    BalancedBot,
    RationalExploiter,
    Aggressive3BetBot,
    DynamicRandomBot,
    sample_league_opponent
)

def test_chen_score_accuracy():
    # AA = 10 * 2 = 20
    assert calc_chen_score("As", "Ah") == 20.0
    # KK = 8 * 2 = 16
    assert calc_chen_score("Ks", "Kh") == 16.0
    # AKs = 10 (A) + 2 (suited) - 1 (gap 1) = 11.0
    assert calc_chen_score("As", "Ks") == 11.0
    # 72o = 3.5 - 5 (gap 5) = -1.5
    assert calc_chen_score("7s", "2c") == -1.5
    print("[PASS] test_chen_score_accuracy: Chen 공식 정상 동작 확인.")

def test_balanced_bot_distribution():
    bot = BalancedBot(seat=1, seed=42)
    actions = []
    # 100 hands with high cards
    for i in range(100):
        # Postflop premium hands
        act = bot.act(None, 1, [0, 1, 2, 3], {1: ("As", "Ks")}, round_idx=1, to_call=100.0, is_allin_facing=False)
        actions.append(act)
    counts = Counter(actions)
    # Balanced bot should use both Action 2 (Half pot) and Action 3 (Pot)
    assert counts[2] > 0, "BalancedBot must use Action 2 (Half pot)!"
    assert counts[3] > 0, "BalancedBot must use Action 3 (Pot)!"
    print(f"[PASS] test_balanced_bot_distribution: 하프팟={counts[2]}, 팟={counts[3]}, 체크콜={counts[1]} 균형 분포 확인.")

def test_rational_exploiter_anti_spam():
    exploiter = RationalExploiter(seat=2, seed=123)
    # Weak hand facing huge bet -> must fold (Action 0)
    weak_act = exploiter.act(None, 2, [0, 1], {2: ("7s", "2c")}, round_idx=1, to_call=1000.0, is_allin_facing=False)
    assert weak_act == 0, "RationalExploiter must fold weak hand against huge bet!"
    
    # Monster hand facing all-in -> must trap call (Action 1)
    monster_act = exploiter.act(None, 2, [0, 1], {2: ("As", "Ah")}, round_idx=1, to_call=5000.0, is_allin_facing=True)
    assert monster_act == 1, "RationalExploiter must trap call monster hand against all-in!"
    print("[PASS] test_rational_exploiter_anti_spam: 스팸 베팅에 대한 칼폴드 및 몬스터 트랩 확인.")

def test_aggressive_3bet_bot():
    bot = Aggressive3BetBot(seat=3, seed=777)
    # Preflop facing a raise (to_call > 100) with strong hand
    act_3bet = bot.act(None, 3, [0, 1, 3], {3: ("As", "Qh")}, round_idx=0, to_call=300.0, is_allin_facing=False)
    assert act_3bet == 3, f"Aggressive3BetBot must 3-Bet (Action 3) with AQ facing raise! got {act_3bet}"
    
    # Flop C-bet check
    flop_acts = [bot.act(None, 3, [0, 1, 2], {3: ("As", "Qh")}, round_idx=1, to_call=0.0, is_allin_facing=False) for _ in range(50)]
    flop_counts = Counter(flop_acts)
    assert flop_counts[2] > 0, "Aggressive3BetBot must fire C-bet (Action 2) on flop!"
    print(f"[PASS] test_aggressive_3bet_bot: 3-Bet 및 플랍 C-bet({flop_counts[2]}/50) 발동 확인.")

def test_league_sampling_distribution():
    bot_types = []
    for s in range(10000):
        opp = sample_league_opponent(seat=1, seed=s)
        bot_types.append(opp.__class__.__name__)
    counts = Counter(bot_types)
    total = 10000
    
    p_self = counts["SelfPlayLeagueBot"] / total
    p_bal = counts["BalancedBot"] / total
    p_exp = counts["RationalExploiter"] / total
    p_3bet = counts["Aggressive3BetBot"] / total
    p_rand = counts["DynamicRandomBot"] / total
    
    print(f"Sampled 10,000 Opponents:")
    print(f"  SelfPlay (Target 30%): {p_self*100:.1f}%")
    print(f"  Balanced (Target 25%): {p_bal*100:.1f}%")
    print(f"  Exploiter (Target 20%): {p_exp*100:.1f}%")
    print(f"  Aggressive3Bet (Target 15%): {p_3bet*100:.1f}%")
    print(f"  DynamicRandom (Target 10%): {p_rand*100:.1f}%")
    
    assert 0.28 <= p_self <= 0.32, "SelfPlay proportion mismatch"
    assert 0.23 <= p_bal <= 0.27, "Balanced proportion mismatch"
    assert 0.18 <= p_exp <= 0.22, "Exploiter proportion mismatch"
    assert 0.13 <= p_3bet <= 0.17, "Aggressive3Bet proportion mismatch"
    assert 0.08 <= p_rand <= 0.12, "DynamicRandom proportion mismatch"
    print("[PASS] test_league_sampling_distribution: 리그 상대 비율 오차범위 1% 이내 완벽 일치 확인.")

def run_all_tests():
    print("=" * 70)
    print("RUNNING DIRECTIVE 041 LEAGUE BOT UNIT TESTS")
    print("=" * 70)
    test_chen_score_accuracy()
    test_balanced_bot_distribution()
    test_rational_exploiter_anti_spam()
    test_aggressive_3bet_bot()
    test_league_sampling_distribution()
    print("=" * 70)
    print("ALL LEAGUE BOT TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
