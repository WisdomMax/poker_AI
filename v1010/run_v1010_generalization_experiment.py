import os
import sys
import json
import time
import random
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v83.run_v83_robust_range_and_cfr_audit import RobustBayesianRangeModel
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v90.run_v90_adversarial_causal_audit_and_scaling import MultiwayHeroExploiter
from v91.run_v91_chip_flow_and_rl_ablation import TightAggressiveBot
from v103.run_v103_showdown_and_rl_validation import BlindStealerBot, HyperAggressiveBot
from v106.run_v106_deterministic_replay import DeterministicBalancedBot

# =====================================================================
# DETERMINISTIC INFRASTRUCTURE
# =====================================================================
class DeterministicTrueNFSP_Agent(TrueNFSP_6Action_Agent):
    def select_action_deterministic(self, info, legal_6, rng):
        norm = normalize_inf_state_v79(info)
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True
        self.sl_net.eval()
        with torch.no_grad():
            lg = self.sl_net(x).squeeze(0)
            lg[~mask] = -1e9
            probs = F.softmax(lg, dim=-1).numpy()
        return int(rng.choice(self.action_dim, p=probs))

def estimate_equity_det(hero, board, combos, probs, n_opps, sims, rng):
    known = set(hero) | set(board)
    deck = list(set(range(52)) - known)
    need_board = 5 - len(board)
    if len(deck) < (2 * n_opps + need_board): return 0.50
    wins = ties = 0
    nc = len(combos)
    if nc == 0: return 0.50
    for _ in range(sims):
        i1 = rng.choice(nc, p=probs)
        h1 = list(combos[i1])
        blocked = known | set(h1)
        rem = [c for c in deck if c not in blocked]
        extra = n_opps - 1
        needed = need_board + 2 * extra
        if len(rem) < needed: continue
        sampled = rng.choice(rem, size=needed, replace=False)
        sim_board = list(board) + list(sampled[:need_board])
        hs = eval_7cards(list(hero) + sim_board)
        won, tied = True, False
        os1 = eval_7cards(h1 + sim_board)
        if os1 > hs: won = False
        elif os1 == hs: tied = True
        if won or tied:
            for i in range(extra):
                oh = list(sampled[need_board+2*i:need_board+2*i+2])
                os2 = eval_7cards(oh + sim_board)
                if os2 > hs: won = tied = False; break
                elif os2 == hs: tied = True
        if won and not tied: wins += 1
        elif tied: ties += 1
    return float(wins + 0.5 * ties) / sims

class DeterministicTightPassiveBot:
    def __init__(self, seed): self.rng = random.Random(seed)
    def select_action(self, st, cp, l6):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        if tc > 150 and 0 in l6 and self.rng.random() < 0.55: return 0
        return 1

class DeterministicUltraAggressiveBot:
    def __init__(self, seed): self.rng = random.Random(seed)
    def select_action(self, st, cp, l6):
        sp = extract_player_spent(st)
        tc = max(sp) - sp[cp]
        if tc > 0:
            r = [a for a in l6 if a in [2,3,5]]
            if r and self.rng.random() < 0.80: return r[-1]
            return 1 if 1 in l6 else (0 if 0 in l6 else 1)
        else:
            b = [a for a in l6 if a in [2,3]]
            if b and self.rng.random() < 0.85: return b[-1]
            return 1

def get_standard_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        DeterministicTightPassiveBot(seed=seed_h*100+2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(seed=seed_h*100+4),
    ]

def get_non_cs_opponents(seed_h):
    return [
        TightAggressiveBot(),
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        BlindStealerBot(),
        HyperAggressiveBot(),
    ]

class ModelA:
    def __init__(self, bp, sims=300):
        self.bp = bp; self.sims = sims; self.temp = 2.0
    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        sp = extract_player_spent(st); tc = max(sp)-sp[cp]; pot = sum(sp)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        if tc > 0 and 0 in l6:
            hc = list(np.where(inf[5:57]==1.0)[0])
            bc = list(np.where(inf[57:109]==1.0)[0])
            ao = sum(1 for p in range(5) if p!=cp and (folded is None or p not in folded))
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hc, bc, mode='temperature', temperature=self.temp,
                opp_aggression_level=1.5, is_raised=(tc>=100))
            po = tc/float(pot+tc)
            eq_rng = np.random.RandomState((hand_seed*10007+cp*101+dec_idx*17)&0xFFFFFFFF)
            eq = estimate_equity_det(hc, bc, combos, probs, ao, self.sims, eq_rng)
            if eq > po+0.05:
                r = [a for a in l6 if a in [2,3]]
                return r[0] if (r and eq>0.65) else (1 if 1 in l6 else l6[0])
            elif eq < po-0.05: return 0
            else: return 1 if 1 in l6 else l6[0]
        bp_rng = np.random.RandomState((hand_seed*20011+cp*307+dec_idx*31)&0xFFFFFFFF)
        return self.bp.select_action_deterministic(inf, l6, bp_rng)

class ModelB:
    def __init__(self, bp, q, sims=300):
        self.bp = bp; self.q = q; self.sims = sims; self.temp = 2.0
    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        sp = extract_player_spent(st); tc = max(sp)-sp[cp]; pot = sum(sp)
        if tc > 0 and 0 in l6:
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            hc = list(np.where(inf[5:57]==1.0)[0])
            bc = list(np.where(inf[57:109]==1.0)[0])
            ao = sum(1 for p in range(5) if p!=cp and (folded is None or p not in folded))
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hc, bc, mode='temperature', temperature=self.temp,
                opp_aggression_level=1.5, is_raised=(tc>=100))
            po = tc/float(pot+tc)
            eq_rng = np.random.RandomState((hand_seed*10007+cp*101+dec_idx*17)&0xFFFFFFFF)
            eq = estimate_equity_det(hc, bc, combos, probs, ao, self.sims, eq_rng)
            if eq > po+0.15:
                r = [a for a in l6 if a in [2,3]]
                return r[0] if r else 1
            elif eq >= po-0.05: return 1
            else: return 0
        state = self.q.get_abstract_state(st, cp)
        qv = self.q.q_table[state]
        if np.any(qv != 0.0):
            m = np.full(6, -1e9, dtype=np.float32)
            for a in l6: m[a] = qv[a]
            return int(np.argmax(m))
        return 1 if 1 in l6 else l6[0]

def main():
    print("="*80)
    print("POKER AI V10.10: MULTI-SEED GENERALIZATION & FINAL POLICY COMPARISON")
    print("="*80)

    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Q-table loaded: {len(q.q_table)} states")

    # ================================================================
    # PHASE 1: 기존 V10.9 12만 판 재분석 (시드 수준 보수적 통계)
    # ================================================================
    print("\n[*] Phase 1: Re-analyzing V10.9 Multi-Seed Variance...")
    # 시드별 실측 데이터:
    # MC=30:  [+14.3, +68.5, +22.9] -> mean=+35.23, se_seed=16.85, t=2.09, p_two=0.1712, p_one=0.0856
    # MC=100: [-2.0, +44.6, +13.9]  -> mean=+18.83, se_seed=13.62, t=1.38, p_two=0.3024
    # MC=300: [+21.8, +20.8, -9.2]  -> mean=+11.13, se_seed=10.17, t=1.09, p_two=0.3879
    # MC=1000:[-3.8, +31.4, +3.9]   -> mean=+10.50, se_seed=10.66, t=0.98, p_two=0.4293
    # 4개 다중 검정 Bonferroni 임계 p: 0.05 / 4 = 0.0125

    # ================================================================
    # PHASE 2: 5개 신규 독립 시드 대규모 재현 실험 (MC=300 표준, 각 2,500핸드)
    # ================================================================
    print("\n[*] Phase 2: Independent Replication (5 New Seeds × 2,500 hands, MC=300)...")
    NEW_SEEDS = [20300001, 20310001, 20320001, 20330001, 20340001]
    HANDS_PER_SEED = 2500

    ma = ModelA(bp, sims=300)
    mb = ModelB(bp, q, sims=300)

    seed_diffs = []
    seed_a_rets = []
    seed_b_rets = []
    all_paired_diffs = []

    # 행동 변경 통계 추적 (Phase 4)
    action_transitions = {
        "check_to_bet": 0,       # 1 -> 2 or 3
        "check_to_overbet": 0,   # 1 -> 4
        "check_to_allin": 0,     # 1 -> 5
        "bet_to_check": 0,       # 2,3,4,5 -> 1
        "other": 0
    }
    action_diff_payoffs = {k: [] for k in action_transitions}

    t0_all = time.time()
    for si, s_val in enumerate(NEW_SEEDS):
        print(f"  Running Seed {si+1}/5 ({s_val}) ...", end=" ", flush=True)
        t_s = time.time()
        r_a, r_b = [], []
        for h in range(HANDS_PER_SEED):
            seed_h = s_val + h
            seat = h % 5

            # Model A
            opps_a = get_standard_opponents(seed_h)
            st_a = game.new_initial_state()
            drng_a = np.random.RandomState(seed_h)
            advance_chance_nodes(st_a, drng_a)
            f_a = set(); dec_a = 0
            while not st_a.is_terminal():
                if st_a.is_chance_node(): advance_chance_nodes(st_a, drng_a); continue
                cp = st_a.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                a = ma.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
                if cp == seat: dec_a += 1
                if a == 0: f_a.add(cp)
                st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
            ret_a = st_a.returns()[seat] / 100.0
            r_a.append(ret_a)

            # Model B (행동 전이 추적)
            opps_b = get_standard_opponents(seed_h)
            st_b = game.new_initial_state()
            drng_b = np.random.RandomState(seed_h)
            advance_chance_nodes(st_b, drng_b)
            f_b = set(); dec_b = 0
            hand_trans_type = None

            while not st_b.is_terminal():
                if st_b.is_chance_node(): advance_chance_nodes(st_b, drng_b); continue
                cp = st_b.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                if cp == seat:
                    act_a_hypo = ma.act(st_b, cp, l6, f_b, seed_h, dec_b)
                    act_b = mb.act(st_b, cp, l6, f_b, seed_h, dec_b)
                    dec_b += 1
                    if act_b != act_a_hypo and hand_trans_type is None:
                        if act_a_hypo == 1 and act_b in [2, 3]: hand_trans_type = "check_to_bet"
                        elif act_a_hypo == 1 and act_b == 4: hand_trans_type = "check_to_overbet"
                        elif act_a_hypo == 1 and act_b == 5: hand_trans_type = "check_to_allin"
                        elif act_a_hypo in [2,3,4,5] and act_b == 1: hand_trans_type = "bet_to_check"
                        else: hand_trans_type = "other"
                    a = act_b
                else:
                    a = opps_b[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
                if a == 0: f_b.add(cp)
                st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
            ret_b = st_b.returns()[seat] / 100.0
            r_b.append(ret_b)

            all_paired_diffs.append((ret_b - ret_a) * 100.0)
            if hand_trans_type:
                action_transitions[hand_trans_type] += 1
                action_diff_payoffs[hand_trans_type].append(ret_b - ret_a)

        m_a = float(np.mean(r_a)) * 100.0
        m_b = float(np.mean(r_b)) * 100.0
        diff = m_b - m_a
        seed_a_rets.append(m_a)
        seed_b_rets.append(m_b)
        seed_diffs.append(diff)
        print(f"A={m_a:+.1f}, B={m_b:+.1f} -> Diff={diff:+.2f} bb/100 [{time.time()-t_s:.1f}s]")

    # ================================================================
    # 통계 계산: 시드 수준 vs 핸드 수준
    # ================================================================
    # 1. 시드 수준 (N=5, 자유도 4) - 가장 보수적
    mean_seed_diff = float(np.mean(seed_diffs))
    se_seed_diff = float(stats.sem(seed_diffs))
    t_seed = mean_seed_diff / se_seed_diff if se_seed_diff > 0 else 0.0
    p_seed_two = float(stats.t.sf(abs(t_seed), df=len(seed_diffs)-1) * 2)
    p_seed_one = float(stats.t.sf(t_seed, df=len(seed_diffs)-1))
    ci95_seed = (mean_seed_diff - 2.776 * se_seed_diff, mean_seed_diff + 2.776 * se_seed_diff)

    # 2. 핸드 수준 (N=12,500)
    arr_h = np.array(all_paired_diffs)
    mean_h = float(np.mean(arr_h))
    se_h = float(stats.sem(arr_h))
    t_h = mean_h / se_h if se_h > 0 else 0.0
    p_h_two = float(stats.t.sf(abs(t_h), df=len(arr_h)-1) * 2)
    p_h_one = float(stats.t.sf(t_h, df=len(arr_h)-1))
    ci95_h = (mean_h - 1.96 * se_h, mean_h + 1.96 * se_h)

    print("\n[Phase 2: 5개 신규 독립 시드 재현 결과]")
    print(f"  시드 수준 (N=5)   : Mean={mean_seed_diff:+.2f} ± {se_seed_diff:.2f} bb/100 | 95% CI=[{ci95_seed[0]:+.2f}, {ci95_seed[1]:+.2f}] | t={t_seed:+.4f} | p(two)={p_seed_two:.4f} | p(one)={p_seed_one:.4f}")
    print(f"  핸드 수준 (N=12.5k): Mean={mean_h:+.2f} ± {se_h:.2f} bb/100 | 95% CI=[{ci95_h[0]:+.2f}, {ci95_h[1]:+.2f}] | t={t_h:+.4f} | p(two)={p_h_two:.4f} | p(one)={p_h_one:.4f}")

    # ================================================================
    # PHASE 4: 행동 변경 분석
    # ================================================================
    print("\n[Phase 4: Q-Learning 행동 변경 세부 분석]")
    action_summary = {}
    for k, count in action_transitions.items():
        pays = action_diff_payoffs[k]
        m_pay = float(np.mean(pays)) if pays else 0.0
        tot_pay = float(np.sum(pays)) if pays else 0.0
        se_pay = float(stats.sem(pays)) if len(pays) > 1 else 0.0
        action_summary[k] = {
            "count": count,
            "mean_diff_bb": m_pay,
            "total_diff_bb": tot_pay,
            "se_diff_bb": se_pay
        }
        print(f"  * {k:18s}: {count:4d}회 | 건당 EV={m_pay:+.2f} BB | 누적 손익={tot_pay:+.1f} BB")

    # ================================================================
    # PHASE 5: 비-콜링스테이션 풀 스트레스 테스트 (일반화 검증, 2,500핸드)
    # ================================================================
    print("\n[*] Phase 5: Non-CallingStation Pool Generalization (2,500 hands)...")
    non_cs_seed = 20350001
    r_ncs_a, r_ncs_b = [], []
    for h in range(2500):
        s_h = non_cs_seed + h
        seat = h % 5

        # A
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(s_h))
        opps_a = get_non_cs_opponents(s_h)
        f_a = set(); dec_a = 0
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(s_h)); continue
            cp = st_a.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
            a = ma.act(st_a, cp, l6, f_a, s_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
            if cp == seat: dec_a += 1
            if a == 0: f_a.add(cp)
            st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
        r_ncs_a.append(st_a.returns()[seat] / 100.0)

        # B
        st_b = game.new_initial_state()
        advance_chance_nodes(st_b, np.random.RandomState(s_h))
        opps_b = get_non_cs_opponents(s_h)
        f_b = set(); dec_b = 0
        while not st_b.is_terminal():
            if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(s_h)); continue
            cp = st_b.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
            a = mb.act(st_b, cp, l6, f_b, s_h, dec_b) if cp == seat else opps_b[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
            if cp == seat: dec_b += 1
            if a == 0: f_b.add(cp)
            st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
        r_ncs_b.append(st_b.returns()[seat] / 100.0)

    m_ncsa = float(np.mean(r_ncs_a)) * 100.0
    m_ncsb = float(np.mean(r_ncs_b)) * 100.0
    ncs_diff = m_ncsb - m_ncsa
    ncs_diffs = [(b - a) * 100.0 for a, b in zip(r_ncs_a, r_ncs_b)]
    se_ncs = float(stats.sem(ncs_diffs))
    p_ncs = float(stats.t.sf(abs(ncs_diff / se_ncs), df=2499) * 2)

    print(f"  Non-CS Arena: Model A={m_ncsa:+.2f}, Model B={m_ncsb:+.2f} -> Diff={ncs_diff:+.2f} ± {se_ncs:.2f} bb/100 (p={p_ncs:.4f})")

    # ================================================================
    # 결과 JSON 저장
    # ================================================================
    output = {
        "seeds": NEW_SEEDS,
        "seed_a_rets": seed_a_rets,
        "seed_b_rets": seed_b_rets,
        "seed_diffs": seed_diffs,
        "seed_level_stats": {
            "mean": mean_seed_diff,
            "se": se_seed_diff,
            "ci95": ci95_seed,
            "t": t_seed,
            "p_two": p_seed_two,
            "p_one": p_seed_one
        },
        "hand_level_stats": {
            "n": len(all_paired_diffs),
            "mean": mean_h,
            "se": se_h,
            "ci95": ci95_h,
            "t": t_h,
            "p_two": p_h_two,
            "p_one": p_h_one
        },
        "action_analysis": action_summary,
        "non_cs_arena": {
            "model_a": m_ncsa,
            "model_b": m_ncsb,
            "diff": ncs_diff,
            "se": se_ncs,
            "p_two": p_ncs
        }
    }

    out_file = "v1010/data/v1010_generalization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] V10.10 실험 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
