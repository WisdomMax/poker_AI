"""
POKER AI V10.9: EQUITY PRECISION CAUSAL TEST & LARGE-SAMPLE CHAMPIONSHIP
ChatGPT 명령서 기반 이중차분(DID) 정밀도 검정 및 반사실적 인과 효과 분석

핵심 검정:
  (B-A)@1000 - (B-A)@30 의 대응 신뢰구간 산출
  단측 vs 양측 p값 명확 구분
  3개 독립 시드 × 5,000핸드 × 4 MC 조건
"""

import os, sys, json, time, random
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
from v106.run_v106_deterministic_replay import DeterministicBalancedBot

# =====================================================================
# 결정론적 인프라 (V10.8에서 검증 완료)
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

def get_opps(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        DeterministicTightPassiveBot(seed=seed_h*100+2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(seed=seed_h*100+4),
    ]

class ModelA:
    """Temperature Hybrid (결정론적)"""
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
    """Q-Learning Offense + Temp Defense (결정론적)"""
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

def run_hands(game, model, opps_fn, n_hands, base_seed, seat_fn=None):
    """model의 .act(st, cp, l6, folded, hand_seed, dec_idx) 인터페이스 사용"""
    results = []
    for h in range(n_hands):
        seed_h = base_seed + h
        seat = h % 5 if seat_fn is None else seat_fn(h)
        opps = opps_fn(seed_h)
        st = game.new_initial_state()
        drng = np.random.RandomState(seed_h)
        advance_chance_nodes(st, drng)
        f = set(); dec = 0
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, drng); continue
            cp = st.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st, cp)
            if cp == seat:
                a = model.act(st, cp, l6, f, seed_h, dec); dec += 1
            else:
                opp = opps[cp if cp < seat else cp-1]
                a = opp.select_action(st, cp, l6)
            if a == 0: f.add(cp)
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, a))
        results.append(st.returns()[seat] / 100.0)
    return results

def main():
    print("="*80)
    print("POKER AI V10.9: EQUITY PRECISION CAUSAL TEST & LARGE-SAMPLE CHAMPIONSHIP")
    print("="*80)

    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Q-table loaded: {len(q.q_table)} states")

    # ================================================================
    # PHASE 1: 이중차분(DID) 검정 - 3 독립 시드 × 5,000핸드 × 4 MC 조건
    # ================================================================
    print("\n[*] Phase 1: DID Causal Test (3 seeds × 5,000 hands × 4 MC settings)...")
    SEEDS   = [20270001, 20280001, 20290001]
    MC_SIMS = [30, 100, 300, 1000]
    N_HANDS = 5000

    # seed × mc → (results_a, results_b) 저장
    all_results = {}   # key: (seed_idx, mc) → (list_a, list_b)

    total_start = time.time()
    for si, base_seed in enumerate(SEEDS):
        for mc in MC_SIMS:
            print(f"  Seed {si+1}/3, MC={mc:4d} → {N_HANDS} hands ...", end=" ", flush=True)
            t0 = time.time()
            ma = ModelA(bp, sims=mc)
            mb = ModelB(bp, q, sims=mc)
            ra = run_hands(game, ma, get_opps, N_HANDS, base_seed)
            rb = run_hands(game, mb, get_opps, N_HANDS, base_seed)
            elapsed = time.time()-t0
            mean_a = np.mean(ra)*100; mean_b = np.mean(rb)*100
            diff = mean_b - mean_a
            print(f"A={mean_a:+.1f} B={mean_b:+.1f} Diff={diff:+.1f} bb/100  [{elapsed:.0f}s]")
            all_results[(si, mc)] = (ra, rb)

    print(f"\n  총 소요시간: {time.time()-total_start:.0f}s")

    # ================================================================
    # PHASE 2: 통계 집계 - 대응 차이 및 DID 신뢰구간
    # ================================================================
    print("\n[*] Phase 2: Statistical Aggregation & DID Analysis...")

    mc_stats = {}   # mc → {mean_diff, se_diff, ci95, t, p_one, p_two}
    for mc in MC_SIMS:
        diffs_all = []
        for si in range(len(SEEDS)):
            ra, rb = all_results[(si, mc)]
            for a, b in zip(ra, rb):
                diffs_all.append(b - a)
        arr = np.array(diffs_all) * 100.0
        mean_d = float(np.mean(arr))
        se_d   = float(stats.sem(arr))
        t_val  = mean_d / se_d if se_d > 0 else 0.0
        p_one  = float(stats.t.sf(t_val, df=len(arr)-1))
        p_two  = float(stats.t.sf(abs(t_val), df=len(arr)-1) * 2)
        ci95   = (mean_d - 1.96*se_d, mean_d + 1.96*se_d)
        mc_stats[mc] = dict(mean=mean_d, se=se_d, t=t_val, p_one=p_one, p_two=p_two, ci95=ci95, n=len(arr))

    # DID: (B-A)@1000 - (B-A)@30 (paired per hand)
    did_diffs = []
    for si in range(len(SEEDS)):
        r30_a, r30_b = all_results[(si, 30)]
        r1k_a, r1k_b = all_results[(si, 1000)]
        for a30, b30, a1k, b1k in zip(r30_a, r30_b, r1k_a, r1k_b):
            did_diffs.append(((b1k-a1k) - (b30-a30)) * 100.0)

    did_arr  = np.array(did_diffs)
    did_mean = float(np.mean(did_arr))
    did_se   = float(stats.sem(did_arr))
    did_t    = did_mean / did_se if did_se > 0 else 0.0
    did_p    = float(stats.t.sf(abs(did_t), df=len(did_arr)-1) * 2)
    did_ci   = (did_mean - 1.96*did_se, did_mean + 1.96*did_se)

    # ================================================================
    # PHASE 3: 결과 출력
    # ================================================================
    print("\n[Phase 1 & 2: MC 조건별 대응 차이 (B-A) 통계 요약]")
    print(f"{'MC':>6} | {'Mean B-A':>10} | {'SE':>8} | {'95% CI':>22} | {'t':>7} | {'p(one)':>8} | {'p(two)':>8}")
    print("-"*80)
    for mc in MC_SIMS:
        s = mc_stats[mc]
        print(f"{mc:6d} | {s['mean']:+10.2f} | {s['se']:8.2f} | [{s['ci95'][0]:+8.2f}, {s['ci95'][1]:+8.2f}] | {s['t']:+7.4f} | {s['p_one']:8.4f} | {s['p_two']:8.4f}")
    print("-"*80)
    print(f"\n[DID: (B-A)@1000 - (B-A)@30]")
    print(f"  Mean DID  : {did_mean:+.4f} bb/100")
    print(f"  SE        : {did_se:.4f}")
    print(f"  95% CI    : [{did_ci[0]:+.4f}, {did_ci[1]:+.4f}]")
    print(f"  t         : {did_t:+.4f}")
    print(f"  p (two)   : {did_p:.4f}  => {'유의 (p<0.05)' if did_p < 0.05 else '경향성 (p<0.10)' if did_p < 0.10 else '미달'}")

    # ================================================================
    # PHASE 4: 반사실적 인과 효과 (단측/양측 명확 구분)
    # ================================================================
    print("\n[*] Phase 4: Causal Rollout - 분기 상태 반사실적 검정 (단측/양측 명확 구분)...")
    causal_model_a = ModelA(bp, sims=300)
    causal_model_b = ModelB(bp, q, sims=300)
    causal_diffs = []
    n_causal_hands = 5000

    for h in range(n_causal_hands):
        seed_h = 20261010 + h
        seat   = h % 5
        opps_b = get_opps(seed_h)
        st_b   = game.new_initial_state()
        drng_b = np.random.RandomState(seed_h)
        advance_chance_nodes(st_b, drng_b)
        f_b  = set(); dec_b = 0; diverged = False

        while not st_b.is_terminal():
            if st_b.is_chance_node(): advance_chance_nodes(st_b, drng_b); continue
            cp = st_b.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
            if cp == seat:
                aa = causal_model_a.act(st_b, cp, l6, f_b, seed_h, dec_b)
                ab = causal_model_b.act(st_b, cp, l6, f_b, seed_h, dec_b)
                dec_b += 1
                if ab != aa and not diverged:
                    diverged = True
                    # Branch A
                    cl_a = st_b.clone(); f_ca = set(f_b)
                    cl_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(cl_a, cp, aa))
                    if aa == 0: f_ca.add(cp)
                    rng_a = np.random.RandomState(seed_h+900000); dc_a = dec_b
                    opps_ca = get_opps(seed_h+500000)
                    while not cl_a.is_terminal():
                        if cl_a.is_chance_node(): advance_chance_nodes(cl_a, rng_a); continue
                        p2 = cl_a.current_player()
                        l2 = True6ActionAdapter.get_legal_6actions(cl_a, p2)
                        if p2 == seat:
                            a2 = causal_model_a.act(cl_a, p2, l2, f_ca, seed_h, dc_a); dc_a += 1
                        else:
                            a2 = opps_ca[p2 if p2<seat else p2-1].select_action(cl_a, p2, l2)
                        if a2 == 0: f_ca.add(p2)
                        cl_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(cl_a, p2, a2))
                    pay_a = cl_a.returns()[seat]/100.0

                    # Branch B
                    cl_b = st_b.clone(); f_cb = set(f_b)
                    cl_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(cl_b, cp, ab))
                    if ab == 0: f_cb.add(cp)
                    rng_b2 = np.random.RandomState(seed_h+900000); dc_b2 = dec_b
                    opps_cb = get_opps(seed_h+500000)
                    while not cl_b.is_terminal():
                        if cl_b.is_chance_node(): advance_chance_nodes(cl_b, rng_b2); continue
                        p2 = cl_b.current_player()
                        l2 = True6ActionAdapter.get_legal_6actions(cl_b, p2)
                        if p2 == seat:
                            a2 = causal_model_b.act(cl_b, p2, l2, f_cb, seed_h, dc_b2); dc_b2 += 1
                        else:
                            a2 = opps_cb[p2 if p2<seat else p2-1].select_action(cl_b, p2, l2)
                        if a2 == 0: f_cb.add(p2)
                        cl_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(cl_b, p2, a2))
                    pay_b = cl_b.returns()[seat]/100.0
                    causal_diffs.append(pay_b - pay_a)
                a = ab
            else:
                opp = opps_b[cp if cp<seat else cp-1]
                a = opp.select_action(st_b, cp, l6)
            if a == 0: f_b.add(cp)
            st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))

    N = len(causal_diffs)
    cd = np.array(causal_diffs)
    cm = float(np.mean(cd)); cse = float(stats.sem(cd))
    ct = cm/cse if cse > 0 else 0.0
    cp1 = float(stats.t.sf(ct, df=N-1))        # 단측
    cp2 = float(stats.t.sf(abs(ct), df=N-1)*2)  # 양측
    cci = (cm-1.96*cse, cm+1.96*cse)

    print(f"\n[Phase 4: 반사실적 인과 효과 (N={N})]")
    print(f"  평균 인과 가치   : {cm:+.4f} ± {cse:.4f} BB/decision")
    print(f"  누적 인과 손익   : {float(np.sum(cd)):+.2f} BB")
    print(f"  95% CI          : [{cci[0]:+.4f}, {cci[1]:+.4f}] BB")
    print(f"  t 통계량         : {ct:+.4f}")
    print(f"  p (단측, H:>0)  : {cp1:.4f}  => {'유의' if cp1<0.05 else '경향성' if cp1<0.10 else '미달'}")
    print(f"  p (양측, H:≠0)  : {cp2:.4f}  => {'유의' if cp2<0.05 else '경향성' if cp2<0.10 else '미달'}")

    # ================================================================
    # 최종 저장
    # ================================================================
    output = {
        "mc_stats": {str(mc): mc_stats[mc] for mc in MC_SIMS},
        "did": {"mean": did_mean, "se": did_se, "t": did_t, "p_two": did_p, "ci95": did_ci},
        "causal": {"n": N, "mean": cm, "se": cse, "t": ct, "p_one": cp1, "p_two": cp2, "ci95": cci}
    }
    with open("v109/data/v109_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] V10.9 결과 저장 완료: v109/data/v109_results.json")

if __name__ == "__main__":
    main()
