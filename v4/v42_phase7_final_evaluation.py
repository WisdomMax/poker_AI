"""
POKER AI V4.2 - PHASE 7: 5대 핵심 모델 최종 종합 벤치마크
A. OpenSpiel CFR
B. V4.0 REINFORCE
C. V4.1 NFSP
D. V4.2 Optimized NFSP
E. V4.2 NFSP + Safe Bayesian
"""

import sys
import os
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability, cfr
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_exact_solver import compute_exact_exploitability, get_kuhn_analytical_nash
from v4.phase2b_exact_oracle import get_rock_strategy, get_maniac_strategy, get_station_strategy, get_random_strategy
from v4.phase3_safe_bayesian import play_match, FixedStrategyBot, SafeBayesianPlayer
from v4.phase4_self_play import KuhnPolicyNet, extract_full_policy_dict
from v4.v41_nfsp import NFSPAgent
from v4.v41_train_nfsp import combine_policies
from v4.v42_phase2_ablation import OptimizedNFSPAgent


def run_phase7_final_eval():
    print("=" * 85)
    print("🏆 [PHASE 7] 5대 핵심 모델 최종 종합 벤치마크")
    print("=" * 85)

    game = pyspiel.load_game("kuhn_poker")

    # 1. OpenSpiel CFR 모델
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(1000): cfr_solver.evaluate_and_update_policy()
    cfr_os_policy = cfr_solver.average_policy()
    nc_cfr = float(exploitability.nash_conv(game, cfr_os_policy))
    exp_cfr = nc_cfr / 2.0

    # 2. V4.0 REINFORCE 모델
    rf_net = KuhnPolicyNet(32)
    rf_net.load_state_dict(torch.load("v4/models/self_play_ckpt_final.pt"))
    rf_pol = extract_full_policy_dict(rf_net)
    nc_rf = float(exploitability.nash_conv(game, DictPolicyWrapper(game, rf_pol)))
    exp_rf = nc_rf / 2.0

    # 3. V4.1 NFSP 모델
    p0_41 = NFSPAgent(0); p1_41 = NFSPAgent(1)
    p0_41.avg_policy_net.load_state_dict(torch.load("v4/models/nfsp_p0_avg_ckpt_25000.pt"))
    p1_41.avg_policy_net.load_state_dict(torch.load("v4/models/nfsp_p1_avg_ckpt_25000.pt"))
    v41_pol = combine_policies(p0_41, p1_41, mode="avg")
    nc_v41 = float(exploitability.nash_conv(game, DictPolicyWrapper(game, v41_pol)))
    exp_v41 = nc_v41 / 2.0

    # 4. V4.2 Optimized NFSP 모델
    p0_42 = OptimizedNFSPAgent(0); p1_42 = OptimizedNFSPAgent(1)
    p0_42.avg_policy_net.load_state_dict(torch.load("v4/models/v42_best_nfsp_p0.pt"))
    p1_42.avg_policy_net.load_state_dict(torch.load("v4/models/v42_best_nfsp_p1.pt"))
    p0_d = p0_42.get_average_policy_dict(); p1_d = p1_42.get_average_policy_dict()
    v42_pol = {}
    for c in ['J', 'Q', 'K']:
        v42_pol[f"{c}:"] = p0_d[f"{c}:"]; v42_pol[f"{c}:pb"] = p0_d[f"{c}:pb"]
        v42_pol[f"{c}:p"] = p1_d[f"{c}:p"]; v42_pol[f"{c}:b"] = p1_d[f"{c}:b"]
    nc_v42 = float(exploitability.nash_conv(game, DictPolicyWrapper(game, v42_pol)))
    exp_v42 = nc_v42 / 2.0

    # 5. 상대 평가 (Rock, Maniac, Station 각 5,000판)
    opponents = {
        "Rock": get_rock_strategy(),
        "Maniac": get_maniac_strategy(),
        "Calling Station": get_station_strategy()
    }

    models = {
        "A. OpenSpiel CFR": (FixedStrategyBot("CFR", get_kuhn_analytical_nash(1/3)), nc_cfr, exp_cfr, "1,000 Iter", "0.5s"),
        "B. V4.0 REINFORCE": (FixedStrategyBot("REINFORCE", rf_pol), nc_rf, exp_rf, "10,000 Ep", "2.8s"),
        "C. V4.1 NFSP": (FixedStrategyBot("V4.1 NFSP", v41_pol), nc_v41, exp_v41, "25,000 Ep", "3.1s"),
        "D. V4.2 Opt NFSP": (FixedStrategyBot("V4.2 Opt", v42_pol), nc_v42, exp_v42, "60,000 Ep", "8.5s"),
        "E. V4.2 NFSP+Bayes": (SafeBayesianPlayer(gto_strategy=v42_pol, max_exploit_weight=0.75), nc_v42, exp_v42, "하이브리드", "실시간")
    }

    perf_table = {}
    for m_name, (bot, nc, exp, train_vol, train_time) in models.items():
        perf_table[m_name] = {"nc": nc, "exp": exp, "train_vol": train_vol, "time": train_time}
        for opp_name, opp_strat in opponents.items():
            opp_bot = FixedStrategyBot(opp_name, opp_strat)
            p0, _, _ = play_match(bot, opp_bot, 2500, seed=51)
            _, p1, _ = play_match(opp_bot, bot, 2500, seed=52)
            perf_table[m_name][opp_name] = ((p0 + p1) / 5000.0) * 100.0

    print(f"{'모델명':<20} | {'NashConv':<10} | {'Exploit (bb)':<12} | {'학습량':<11} | {'vs Rock':<10} | {'vs Maniac':<10} | {'vs Station':<10}")
    print("-" * 95)
    for m_name, d in perf_table.items():
        print(f"{m_name:<20} | {d['nc']:.6f}   | {d['exp']*100:>5.2f} bb/100  | {d['train_vol']:<11} | {d['Rock']:>+6.2f} bb   | {d['Maniac']:>+6.2f} bb   | {d['Calling Station']:>+6.2f} bb")
    print("=" * 85)


if __name__ == "__main__":
    run_phase7_final_eval()
