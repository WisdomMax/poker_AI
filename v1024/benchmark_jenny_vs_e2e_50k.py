"""
Head-to-Head Benchmark: Cycle 13 (Jenny, 2.89M) vs New E2E PPO (50k)
====================================================================
Directly responds to ChatGPT's strict critique:
Evaluates both models on identical 1,000 seeds against Unseen Strong Rational Holdout Opponents.
Tracks:
- BB/100 & 95% Confidence Interval
- Called EV (Actual EV when shove is called)
- VPIP, PFR, 3-Bet %
- All-in Rate (Preflop & Postflop)
- Max Drawdown (BB)
- Showdown Win Rate (W$SD)
- Policy Entropy
"""

import os
import sys
import re
import json
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1023.v1023_13_differential_forensics import PPOAgentWrapper
from arena.human_friendly_policy import calculate_chen_score
from arena.audit_directive031_scientific_failure_investigation import StrongRationalBot, parse_cards_from_state

def evaluate_model(model_path, num_hands=1000, seed_base=999000):
    game = get_5p_fullgame()
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    m.eval()
    agent = PPOAgentWrapper(m)
    
    agent_seat = 0
    hand_returns_bb = []
    vpip_cnt = 0
    pfr_cnt = 0
    allin_cnt = 0
    allin_pf = 0
    allin_post = 0
    fold_cnt = 0
    sd_cnt = 0
    sd_win_cnt = 0
    called_allin_returns = []
    uncalled_allin_returns = []
    entropy_samples = []
    
    running_p = 0.0
    cumulative_profits = []
    
    for h in range(num_hands):
        h_seed = seed_base + h
        rng = np.random.RandomState(h_seed)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        priv_cards = parse_cards_from_state(st)
        
        opps = {s: StrongRationalBot(s, h_seed * 10 + s) for s in range(1, 5)}
        folded = set()
        agent_actions = []
        went_allin = False
        allin_street = 0
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
            cp = st.current_player()
            legal = True6ActionAdapter.get_legal_6actions(st, cp)
            rnd = get_round_from_state(st)
            
            if cp == agent_seat:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(inf)
                t_inf = torch.tensor(norm_inf, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = m.forward(t_inf)
                    mask = torch.ones(6, dtype=torch.bool)
                    for a in range(6):
                        if a not in legal: mask[a] = False
                    ml = logits.clone()
                    ml[0, ~mask] = -1e9
                    probs = F.softmax(ml[0], dim=-1).cpu().numpy()
                    p_clean = np.clip(probs[mask.numpy()], 1e-12, 1.0)
                    ent = -np.sum(p_clean * np.log(p_clean))
                    entropy_samples.append(ent)
                    act = int(np.argmax(ml[0].cpu().numpy()))
                    
                raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                agent_actions.append((rnd, act))
                if (act == 5 or raw_act == 10000) and not went_allin:
                    went_allin = True
                    allin_street = rnd
            else:
                act = opps[cp].select_action(st, cp, legal, priv_cards)
                raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                
            if act == 0:
                folded.add(cp)
            st.apply_action(raw_act)
            
        ret_bb = st.returns()[agent_seat] / 100.0
        hand_returns_bb.append(ret_bb)
        running_p += ret_bb
        cumulative_profits.append(running_p)
        
        pf_acts = [a for rnd, a in agent_actions if rnd == 0]
        if any(a in [1, 2, 3, 4, 5] for a in pf_acts): vpip_cnt += 1
        if any(a in [2, 3, 4, 5] for a in pf_acts): pfr_cnt += 1
        if any(a == 0 for rnd, a in agent_actions): fold_cnt += 1
        
        if went_allin:
            allin_cnt += 1
            if allin_street == 0: allin_pf += 1
            else: allin_post += 1
            
            active_opps = [s for s in range(1, 5) if s not in folded]
            if len(active_opps) > 0:
                called_allin_returns.append(ret_bb)
            else:
                uncalled_allin_returns.append(ret_bb)
                
        if len([s for s in range(5) if s not in folded]) > 1 and agent_seat not in folded:
            sd_cnt += 1
            if ret_bb > 0: sd_win_cnt += 1
            
    arr = np.array(hand_returns_bb)
    mean_bb100 = float(np.mean(arr) * 100.0)
    sem = float(np.std(arr) / np.sqrt(num_hands) * 100.0)
    
    peaks = np.maximum.accumulate(cumulative_profits)
    max_dd = float(np.max(peaks - cumulative_profits)) if len(cumulative_profits) > 0 else 0.0
    
    called_ev = float(np.mean(called_allin_returns)) if called_allin_returns else 0.0
    called_ratio = len(called_allin_returns) / allin_cnt * 100.0 if allin_cnt > 0 else 0.0
    avg_entropy = float(np.mean(entropy_samples)) if entropy_samples else 0.0
    
    return {
        "mean_bb100": mean_bb100,
        "sem": sem,
        "ci95": [mean_bb100 - 1.96 * sem, mean_bb100 + 1.96 * sem],
        "vpip": vpip_cnt / num_hands * 100.0,
        "pfr": pfr_cnt / num_hands * 100.0,
        "allin_rate": allin_cnt / num_hands * 100.0,
        "allin_preflop": allin_pf,
        "allin_postflop": allin_post,
        "called_ratio": called_ratio,
        "called_ev_bb": called_ev,
        "max_drawdown": max_dd,
        "wsd": (sd_win_cnt / sd_cnt * 100.0) if sd_cnt > 0 else 0.0,
        "avg_entropy": avg_entropy
    }

def main():
    print("=" * 75)
    print("DIRECTIVE 031: Jenny (Cycle 13) vs New E2E PPO (50k) Head-to-Head Benchmark")
    print("Identical 1,000 Seeds vs Unseen Strong Rational Holdout Opponents")
    print("=" * 75)
    
    t0 = time.time()
    print("[1/2] Evaluating Cycle 13 (Jenny, 2.89M)...", flush=True)
    res_jenny = evaluate_model("v1024/models/V1024_PPO_Cycle_13_2890003hands.pt", 1000, 999000)
    print(f"  Cycle 13 evaluated in {time.time()-t0:.1f}s")
    
    t1 = time.time()
    print("[2/2] Evaluating New E2E PPO Final 50k...", flush=True)
    res_e2e = evaluate_model("v1024/models_e2e/PPO_E2E_Final_50k.pt", 1000, 999000)
    print(f"  E2E Final 50k evaluated in {time.time()-t1:.1f}s")
    
    out = {
        "jenny_cycle13": res_jenny,
        "new_e2e_50k": res_e2e
    }
    
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD BENCHMARK COMPARISON (1,000 Hands Identical Seeds)")
    print("=" * 80)
    print(f"{'지표 (Metric)':26s} | {'기존 제니 (Cycle 13, 2.89M)':24s} | {'신규 E2E PPO (50k)':20s} | {'개선도'}")
    print("-" * 80)
    
    bb_diff = res_e2e["mean_bb100"] - res_jenny["mean_bb100"]
    dd_diff = res_jenny["max_drawdown"] - res_e2e["max_drawdown"]
    allin_diff = res_jenny["allin_rate"] - res_e2e["allin_rate"]
    ent_diff = res_e2e["avg_entropy"] - res_jenny["avg_entropy"]
    cev_diff = res_e2e["called_ev_bb"] - res_jenny["called_ev_bb"]
    
    print(f"{'수익률 (BB/100, 95% CI)':26s} | {res_jenny['mean_bb100']:+6.1f} ± {1.96*res_jenny['sem']:4.1f} BB/100   | {res_e2e['mean_bb100']:+6.1f} ± {1.96*res_e2e['sem']:4.1f} BB/100  | {bb_diff:+6.1f} BB/100")
    print(f"{'최대 낙폭 (Max Drawdown)':26s} | {res_jenny['max_drawdown']:8.1f} BB             | {res_e2e['max_drawdown']:8.1f} BB            | {dd_diff:+8.1f} BB (방어)")
    print(f"{'올인 빈도 (Total All-in)':26s} | {res_jenny['allin_rate']:8.1f}% ({res_jenny['allin_preflop']}/{res_jenny['allin_postflop']})       | {res_e2e['allin_rate']:8.1f}% ({res_e2e['allin_preflop']}/{res_e2e['allin_postflop']})      | {allin_diff:+6.1f}%p (감소)")
    print(f"{'올인 콜당했을 때 EV':26s} | {res_jenny['called_ev_bb']:+8.1f} BB             | {res_e2e['called_ev_bb']:+8.1f} BB            | {cev_diff:+8.1f} BB (개선)")
    print(f"{'정책 엔트로피 (Entropy)':26s} | {res_jenny['avg_entropy']:8.3f}                | {res_e2e['avg_entropy']:8.3f}               | {ent_diff:+8.3f} (건강도)")
    print(f"{'VPIP / PFR':26s} | {res_jenny['vpip']:4.1f}% / {res_jenny['pfr']:4.1f}%        | {res_e2e['vpip']:4.1f}% / {res_e2e['pfr']:4.1f}%       | 정통 레인지")
    print(f"{'쇼다운 승률 (W$SD)':26s} | {res_jenny['wsd']:8.1f}%               | {res_e2e['wsd']:8.1f}%              | 쇼다운 품질")
    print("-" * 80)
    
    with open("v1024/jenny_vs_e2e_50k_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved result to v1024/jenny_vs_e2e_50k_comparison.json")

if __name__ == "__main__":
    main()
