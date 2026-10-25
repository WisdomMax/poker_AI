"""
DIRECTIVE 031: Reinforcement Learning Failure & Policy Collapse Forensic Investigation
=====================================================================================
Rigorous, independent scientific investigation into the poker AI reinforcement learning loop.
Hypothesis to verify:
1. Did PPO suffer Policy Degeneracy / Collapse from Cycle 1 -> 4 -> 7 -> 10 -> 13?
2. Did the training bots' over-folding flaw reward preflop all-in spamming (Reward Gaming)?
3. How do the models perform against a Strong Rational Holdout Opponent vs Training Bots?
4. Detailed Poker Metrics: VPIP, PFR, All-in Freq, Fold Rate, W$SD, Max Drawdown, 95% CI.
5. Exact EV of All-in lines (Overall EV vs Called EV).
"""

import os
import sys
import re
import json
import time
import math
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1023.v1023_13_differential_forensics import normalize_inf_state_v79, PPOAgentWrapper
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot
)
from v1021.run_v1021_anti_limper_league import LooseLimperBot
from arena.human_friendly_policy import calculate_chen_score

def parse_cards_from_state(st):
    s = st.to_string()
    priv = {}
    for line in s.splitlines():
        m = re.match(r'P(\d+) Cards:\s*([2-9TJQKA][shdc][2-9TJQKA][shdc])', line.strip())
        if m:
            p_idx = int(m.group(1))
            c_str = m.group(2)
            priv[p_idx] = [c_str[:2], c_str[2:]]
    return priv

# =============================================================================
# 1. 시계열 정책 붕괴 (Policy Collapse / Degeneracy) 측정
# =============================================================================
def audit_policy_collapse_timeseries(num_states=1000, seed=314159):
    """
    Measures Policy Entropy and Action Distribution across checkpoint cycles:
    Cycle 1 -> Cycle 4 -> Cycle 7 -> Cycle 10 -> Cycle 13 + Baseline
    """
    ckpts = {
        "Cycle 1 (0.34M)": "v1024/models/V1024_PPO_Cycle_1_344433hands.pt",
        "Cycle 4 (1.21M)": "v1024/models/V1024_PPO_Cycle_4_1211670hands.pt",
        "Cycle 7 (1.77M)": "v1024/models/V1024_PPO_Cycle_7_1771164hands.pt",
        "Cycle 10 (2.29M)": "v1024/models/V1024_PPO_Cycle_10_2289023hands.pt",
        "Cycle 13 (2.89M)": "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt",
        "Baseline (100K)": "v1022/models/StrictAntiLimper_R2_100K.pt"
    }
    
    game = get_5p_fullgame()
    rng = np.random.RandomState(seed)
    
    # Generate 1,000 distinct preflop information states
    states = []
    for i in range(num_states):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        cp = st.current_player()
        legal = True6ActionAdapter.get_legal_6actions(st, cp)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        states.append((inf, legal))
        
    results = {}
    
    for label, path in ckpts.items():
        if not os.path.exists(path):
            continue
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        m.eval()
        
        entropy_list = []
        action_counts = Counter()
        prob_sums = np.zeros(6, dtype=np.float64)
        
        with torch.no_grad():
            for inf, legal in states:
                norm_inf = normalize_inf_state_v79(inf)
                t_inf = torch.tensor(norm_inf, dtype=torch.float32).unsqueeze(0)
                logits, _ = m.forward(t_inf)
                
                mask = torch.ones(6, dtype=torch.bool)
                for a in range(6):
                    if a not in legal: mask[a] = False
                masked_logits = logits.clone()
                masked_logits[0, ~mask] = -1e9
                
                probs = F.softmax(masked_logits[0], dim=-1).cpu().numpy()
                prob_sums += probs
                
                # Policy entropy
                p_clean = np.clip(probs[mask.numpy()], 1e-12, 1.0)
                ent = -np.sum(p_clean * np.log(p_clean))
                entropy_list.append(ent)
                
                # Argmax action
                pred_act = int(np.argmax(masked_logits[0].cpu().numpy()))
                action_counts[pred_act] += 1
                
        avg_ent = float(np.mean(entropy_list))
        avg_probs = (prob_sums / num_states).tolist()
        argmax_dist = {a: action_counts[a] / num_states for a in range(6)}
        
        results[label] = {
            "avg_entropy": avg_ent,
            "avg_probabilities": avg_probs,
            "argmax_distribution": argmax_dist,
            "allin_argmax_pct": argmax_dist.get(5, 0.0) * 100.0,
            "fold_argmax_pct": argmax_dist.get(0, 0.0) * 100.0,
            "intermediate_argmax_pct": sum(argmax_dist.get(a, 0.0) for a in [1, 2, 3, 4]) * 100.0
        }
        
    return results

# =============================================================================
# 2. 강한 착취 상대 봇 (Strong Rational Holdout Bot) 구현
# =============================================================================
class StrongRationalBot:
    """
    Unseen holdout opponent that rationally punishes irrational all-ins:
    - Against All-in / Shove (facing >= 3000 chips):
      Calls ONLY with Strong/Premium range (Chen >= 9.0: AA, KK, QQ, JJ, TT, 99, AK, AQ, AJ).
      Folds all weaker hands to never give EV to trash bluffs.
    - Standard Streets: Tight-Aggressive ABC GTO-approximation poker.
    """
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, st, cp, legal, priv_cards):
        spent = extract_player_spent(st)
        curr_spent = spent[cp]
        max_spent = max(spent)
        facing = max_spent - curr_spent
        rnd = get_round_from_state(st)
        
        cards = priv_cards.get(cp, ["??", "??"])
        c_score = calculate_chen_score(cards[0], cards[1])
        
        # Facing All-in / heavy bet
        if facing >= 3000:
            if c_score >= 9.0: # Call with premium value
                return 1
            return 0 if 0 in legal else 1 # Fold everything else
            
        # Preflop standard
        if rnd == 0:
            if c_score >= 12.0: # Monster (AA, KK, QQ, AKs) -> Raise / Shove
                if 4 in legal: return 4
                if 5 in legal: return 5
                return 1
            elif c_score >= 8.5: # Strong (JJ, TT, AQs, AQo, KQs) -> 100% Pot Raise
                if 3 in legal: return 3
                return 1
            elif c_score >= 6.0: # Marginal -> Cheap call only
                if facing <= 150: return 1
                return 0 if 0 in legal else 1
            else: # Trash -> Fold
                if facing == 0: return 1
                return 0 if 0 in legal else 1
                
        # Postflop standard
        if c_score >= 8.0:
            if 3 in legal and self.rng.random() < 0.6: return 3
            return 1
        if facing == 0:
            return 1
        return 0 if 0 in legal else 1

# =============================================================================
# 3. 독립 홀드아웃 환경 1,000핸드 시뮬레이션 및 포커 지표 산출
# =============================================================================
def run_scientific_evaluation(target_model_key, opponent_type="strong_holdout", num_hands=1000, seed_base=123000):
    game = get_5p_fullgame()
    
    # Load target model
    model_paths = {
        "cycle4": "v1024/models/V1024_PPO_Cycle_4_1211670hands.pt",
        "cycle10": "v1024/models/V1024_PPO_Cycle_10_2289023hands.pt",
        "cycle13": "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt",
        "baseline": "v1022/models/StrictAntiLimper_R2_100K.pt"
    }
    
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    model.load_state_dict(torch.load(model_paths[target_model_key], map_location="cpu", weights_only=False))
    model.eval()
    agent = PPOAgentWrapper(model)
    
    # Opponents setup (Seat 0 is target agent, Seat 1~4 are opponents)
    agent_seat = 0
    
    # Tracking metrics
    hand_returns_bb = []
    vpip_count = 0
    pfr_count = 0
    allin_count = 0
    allin_preflop = 0
    allin_postflop = 0
    fold_count = 0
    showdown_count = 0
    showdown_win_count = 0
    
    # All-in forensics
    allin_chen_scores = []
    allin_hands_returns = []
    called_allin_returns = []
    uncalled_allin_returns = []
    
    # Hand by hand drawdown tracking
    cumulative_profits = []
    running_p = 0.0
    
    for h_idx in range(num_hands):
        h_seed = seed_base + h_idx
        chance_rng = np.random.RandomState(h_seed)
        
        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)
        priv_cards = parse_cards_from_state(st)
        
        # Instantiate opponents
        if opponent_type == "strong_holdout":
            opps = {s: StrongRationalBot(s, h_seed * 10 + s) for s in range(1, 5)}
        else: # training_bots (DeterministicTightPassive, LooseLimper, etc.)
            opps = {
                1: DeterministicTightPassiveBot(seed=h_seed * 11),
                2: LooseLimperBot(seed=h_seed * 13),
                3: DeterministicBalancedBot(rng_seed=h_seed * 17),
                4: DeterministicUltraAggressiveBot(seed=h_seed * 19)
            }
            
        folded_seats = set()
        agent_actions = []
        agent_went_allin = False
        allin_street = None
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue
                
            cp = st.current_player()
            legal = True6ActionAdapter.get_legal_6actions(st, cp)
            rnd = get_round_from_state(st)
            
            if cp == agent_seat:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = agent.act(inf, legal)
                raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                agent_actions.append((rnd, act, raw_act))
                
                is_shove = (act == 5 or raw_act == 10000)
                if is_shove and not agent_went_allin:
                    agent_went_allin = True
                    allin_street = rnd
                    c_pair = priv_cards.get(agent_seat, ["??", "??"])
                    allin_chen_scores.append(calculate_chen_score(c_pair[0], c_pair[1]))
            else:
                bot = opps[cp]
                if opponent_type == "strong_holdout":
                    act = bot.select_action(st, cp, legal, priv_cards)
                else:
                    act = bot.select_action(st, cp, legal)
                raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
                
            if act == 0:
                folded_seats.add(cp)
            st.apply_action(raw_act)
            
        # Hand settlement
        ret_chips = st.returns()[agent_seat]
        ret_bb = ret_chips / 100.0
        hand_returns_bb.append(ret_bb)
        running_p += ret_bb
        cumulative_profits.append(running_p)
        
        # Analyze agent's actions in this hand
        pf_actions = [a for rnd, a, raw in agent_actions if rnd == 0]
        # VPIP: Put money in preflop voluntarily (Call or Raise, act in [1, 2, 3, 4, 5])
        # Note: If big blind and checked (act=1 with facing=0), not strictly voluntary, but check/call counts
        if any(a in [1, 2, 3, 4, 5] for a in pf_actions):
            vpip_count += 1
        # PFR: Preflop raise (act in [2, 3, 4, 5])
        if any(a in [2, 3, 4, 5] for a in pf_actions):
            pfr_count += 1
        # Fold count
        if any(a == 0 for rnd, a, raw in agent_actions):
            fold_count += 1
            
        if agent_went_allin:
            allin_count += 1
            if allin_street == 0:
                allin_preflop += 1
            else:
                allin_postflop += 1
            allin_hands_returns.append(ret_bb)
            
            # Check if called by someone
            active_opps = [s for s in range(1, 5) if s not in folded_seats]
            if len(active_opps) > 0:
                called_allin_returns.append(ret_bb)
            else:
                uncalled_allin_returns.append(ret_bb)
                
        # Showdown stats
        if len([s for s in range(5) if s not in folded_seats]) > 1:
            if agent_seat not in folded_seats:
                showdown_count += 1
                if ret_bb > 0:
                    showdown_win_count += 1
                    
    # Metrics calculations
    arr = np.array(hand_returns_bb)
    mean_bb100 = float(np.mean(arr) * 100.0)
    std_bb100 = float(np.std(arr) * 100.0)
    sem = float(std_bb100 / np.sqrt(num_hands))
    ci95 = [mean_bb100 - 1.96 * sem, mean_bb100 + 1.96 * sem]
    
    # Max Drawdown
    peaks = np.maximum.accumulate(cumulative_profits)
    drawdowns = peaks - cumulative_profits
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # All-in EV
    allin_ev_bb = float(np.mean(allin_hands_returns)) if allin_hands_returns else 0.0
    called_allin_ev = float(np.mean(called_allin_returns)) if called_allin_returns else 0.0
    uncalled_allin_ev = float(np.mean(uncalled_allin_returns)) if uncalled_allin_returns else 0.0
    called_ratio = (len(called_allin_returns) / allin_count * 100.0) if allin_count > 0 else 0.0
    
    # Hand strength distribution of all-ins
    trash_allin_pct = (sum(1 for sc in allin_chen_scores if sc < 6.0) / len(allin_chen_scores) * 100.0) if allin_chen_scores else 0.0
    premium_allin_pct = (sum(1 for sc in allin_chen_scores if sc >= 11.0) / len(allin_chen_scores) * 100.0) if allin_chen_scores else 0.0
    avg_chen = float(np.mean(allin_chen_scores)) if allin_chen_scores else 0.0
    
    wsd = (showdown_win_count / showdown_count * 100.0) if showdown_count > 0 else 0.0
    
    return {
        "model": target_model_key,
        "opponent_type": opponent_type,
        "hands": num_hands,
        "mean_bb100": mean_bb100,
        "sem": sem,
        "ci95": ci95,
        "max_drawdown_bb": max_dd,
        "vpip_pct": vpip_count / num_hands * 100.0,
        "pfr_pct": pfr_count / num_hands * 100.0,
        "allin_pct": allin_count / num_hands * 100.0,
        "allin_preflop": allin_preflop,
        "allin_postflop": allin_postflop,
        "fold_pct": fold_count / num_hands * 100.0,
        "showdown_count": showdown_count,
        "wsd_pct": wsd,
        "allin_total_count": allin_count,
        "allin_chen_avg": avg_chen,
        "trash_allin_pct": trash_allin_pct,
        "premium_allin_pct": premium_allin_pct,
        "allin_overall_ev_bb": allin_ev_bb,
        "called_allin_ev_bb": called_allin_ev,
        "uncalled_allin_ev_bb": uncalled_allin_ev,
        "allin_called_ratio_pct": called_ratio
    }

def main():
    print("=" * 80)
    print("DIRECTIVE 031: REINFORCEMENT LEARNING FAILURE INVESTIGATION AUDIT")
    print("=" * 80)
    
    # Part 1: Policy Collapse Analysis across training cycles
    print("\n[Part 1] Analyzing Policy Entropy & Action Collapse Across Training Cycles...")
    t0 = time.time()
    collapse_data = audit_policy_collapse_timeseries(num_states=1000)
    print(f"  -> Analyzed in {time.time()-t0:.1f}s")
    
    print("\n" + "-" * 85)
    print(f"{'체크포인트 단계':22s} | {'엔트로피':8s} | {'Fold %':8s} | {'중간액션 %':10s} | {'All-in %':8s} | {'판정'}")
    print("-" * 85)
    for k, v in collapse_data.items():
        ent = v["avg_entropy"]
        f_pct = v["fold_argmax_pct"]
        mid_pct = v["intermediate_argmax_pct"]
        allin_pct = v["allin_argmax_pct"]
        status = "정상 분산" if ent > 0.8 else ("축퇴 진행" if ent > 0.2 else "완전 붕괴 (Push/Fold)")
        print(f"{k:22s} | {ent:8.3f} | {f_pct:7.1f}% | {mid_pct:9.1f}% | {allin_pct:7.1f}% | {status}")
        
    # Part 2: Controlled Benchmark vs Training Bots vs Strong Rational Holdout Opponents
    print("\n[Part 2] Running Independent Holdout Benchmark (1,000 hands per condition)...")
    models = ["cycle4", "cycle10", "cycle13", "baseline"]
    benchmark_results = {}
    
    for m in models:
        # 1) Training bots
        print(f"  -> Testing {m} against [Training Bots] (1,000 hands)...", flush=True)
        res_train = run_scientific_evaluation(m, opponent_type="training_bots", num_hands=1000, seed_base=500000)
        
        # 2) Strong Holdout
        print(f"  -> Testing {m} against [Strong Holdout Opponents] (1,000 hands)...", flush=True)
        res_holdout = run_scientific_evaluation(m, opponent_type="strong_holdout", num_hands=1000, seed_base=600000)
        
        benchmark_results[m] = {
            "training_bots": res_train,
            "strong_holdout": res_holdout
        }
        
    print("\n" + "=" * 95)
    print("DIRECTIVE 031: BENCHMARK RESULTS MATRIX")
    print("=" * 95)
    print(f"{'모델':12s} | {'상대 환경':18s} | {'수익률 (BB/100)':18s} | {'VPIP':6s} | {'PFR':6s} | {'Allin':6s} | {'Called EV':10s} | {'최대낙폭'}")
    print("-" * 95)
    for m in models:
        for env in ["training_bots", "strong_holdout"]:
            r = benchmark_results[m][env]
            env_name = "훈련 봇 (Weak)" if env == "training_bots" else "홀드아웃 (Strong)"
            ci_str = f"{r['mean_bb100']:+6.1f} ± {1.96*r['sem']:4.1f}"
            print(f"{m:12s} | {env_name:18s} | {ci_str:18s} | {r['vpip_pct']:5.1f}%| {r['pfr_pct']:5.1f}%| {r['allin_pct']:5.1f}%| {r['called_allin_ev_bb']:+8.1f}BB| {r['max_drawdown_bb']:6.1f}BB")
        print("-" * 95)
        
    # Save raw audit artifacts
    audit_payload = {
        "policy_collapse_timeseries": collapse_data,
        "benchmark_matrix": benchmark_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    out_file = Path("/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/directive031_scientific_investigation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[저장 완료] 결과 JSON: {out_file}")

if __name__ == "__main__":
    main()
