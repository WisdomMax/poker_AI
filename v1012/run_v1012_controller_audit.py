import os
import sys
import json
import time
import random
from collections import defaultdict, deque
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

class DeterministicDynamicSwitchingBot:
    def __init__(self, seed, switch_interval=500):
        self.agg = DeterministicUltraAggressiveBot(seed=seed)
        self.tight = DeterministicTightPassiveBot(seed=seed+1)
        self.switch_interval = switch_interval
        self.hand_counter = 0

    def select_action(self, st, cp, legal_6: list):
        self.hand_counter += 1
        if (self.hand_counter // self.switch_interval) % 2 == 0:
            return self.agg.select_action(st, cp, legal_6)
        return self.tight.select_action(st, cp, legal_6)

# 상대 풀 정의
def get_pool_standard(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        DeterministicTightPassiveBot(seed=seed_h*100+2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(seed=seed_h*100+4),
    ]

def get_pool_aggressive(seed_h):
    return [
        TightAggressiveBot(),
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        BlindStealerBot(),
        HyperAggressiveBot(),
    ]

def get_pool_dynamic(seed_h):
    return [
        DeterministicDynamicSwitchingBot(seed=seed_h*100+1, switch_interval=500),
        TightAggressiveBot(),
        DeterministicTightPassiveBot(seed=seed_h*100+3),
        DeterministicBalancedBot(rng_seed=seed_h*100+4),
    ]

# =====================================================================
# POLICIES: Model A, Model B, Model C, Model D
# =====================================================================
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

class ModelC_OldController:
    """기존 V10.11 Aggression Threshold Controller (단순 누적)"""
    def __init__(self, model_a: ModelA, model_b: ModelB):
        self.model_a = model_a
        self.model_b = model_b
        self.opp_calls = defaultdict(int)
        self.opp_bets = defaultdict(int)
        self.choices_a = 0
        self.choices_b = 0

    def update_public_history(self, cp, action):
        if action == 1: self.opp_calls[cp] += 1
        elif action in [2, 3, 4, 5]: self.opp_bets[cp] += 1

    def get_agg_index(self, hero_cp):
        tot_b = sum(self.opp_bets[p] for p in range(5) if p != hero_cp)
        tot_c = sum(self.opp_calls[p] for p in range(5) if p != hero_cp)
        if tot_c == 0: return 1.0 # 초기값 버그: 1.0 < 1.25 -> 초기에 무조건 Model B 선택!
        return tot_b / float(tot_c)

    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        agg = self.get_agg_index(cp)
        if agg > 1.25:
            self.choices_a += 1
            return self.model_a.act(st, cp, l6, folded, hand_seed, dec_idx)
        else:
            self.choices_b += 1
            return self.model_b.act(st, cp, l6, folded, hand_seed, dec_idx)

class ModelD_RobustController:
    """
    Model D: 안전장치 탑재 Robust Controller (Fail-Safe + Seat-Level Exploitation Gate)
    - 원칙 1: 기본값은 항상 최강 기준선 Model A (Default-to-A)
    - 원칙 2: 상대 좌석별(Seat-level) 최근 50회 액션 슬라이딩 윈도우 추적
    - 원칙 3: 테이블 내 활성 상대 중 '순수 콜링스테이션(Call 비율 > 65%, Bet < 15%, Fold < 20%)'이
             확인될 때만 Model B 착취 가동, 그 외에는 전원 Model A 방어.
    """
    def __init__(self, model_a: ModelA, model_b: ModelB, window_size=50):
        self.model_a = model_a
        self.model_b = model_b
        self.player_history = defaultdict(lambda: deque(maxlen=window_size))
        self.choices_a = 0
        self.choices_b = 0

    def update_public_history(self, cp, action):
        self.player_history[cp].append(action)

    def should_exploit(self, hero_cp):
        # 테이블 내에 확실한 Calling Station이 존재하는가?
        for p, acts in self.player_history.items():
            if p == hero_cp: continue
            if len(acts) >= 15:
                tot = len(acts)
                n_calls = sum(1 for a in acts if a == 1)
                n_bets = sum(1 for a in acts if a in [2, 3, 4, 5])
                n_folds = sum(1 for a in acts if a == 0)
                if (n_calls / float(tot) > 0.65) and (n_bets / float(tot) < 0.15) and (n_folds / float(tot) < 0.20):
                    return True
        return False

    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        if self.should_exploit(cp):
            self.choices_b += 1
            return self.model_b.act(st, cp, l6, folded, hand_seed, dec_idx)
        else:
            self.choices_a += 1
            return self.model_a.act(st, cp, l6, folded, hand_seed, dec_idx)

# =====================================================================
# MAIN EXPERIMENT RUNNER
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.12: ADAPTIVE CONTROLLER FAILURE AUDIT & ROBUST SELECTION")
    print("=" * 80)

    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Q-table loaded: {len(q.q_table)} states")

    ma = ModelA(bp, sims=300)
    mb = ModelB(bp, q, sims=300)

    # =================================================================
    # PHASE 1: Pool 2 성적 불일치 포렌식 전수 감사 (2,500핸드)
    # =================================================================
    print("\n[*] Phase 1: Forensic Audit of Pool 2 Discrepancy (2,500 hands)...")
    base_seed_p2 = 20370001
    mc_old = ModelC_OldController(ma, mb)

    p2_diverged_hands = []
    p2_a_rets = []
    p2_c_rets = []

    for h in range(2500):
        seed_h = base_seed_p2 + h
        seat = h % 5

        # Run Model A
        opps_a = get_pool_aggressive(seed_h)
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        f_a = set(); dec_a = 0
        path_a = []
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
            cp = st_a.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
            a = ma.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
            if cp == seat: dec_a += 1
            if a == 0: f_a.add(cp)
            eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a)
            path_a.append((cp, a, eng_amt))
            st_a.apply_action(eng_amt)
        ret_a = st_a.returns()[seat] / 100.0
        p2_a_rets.append(ret_a)

        # Run Model C
        opps_c = get_pool_aggressive(seed_h)
        st_c = game.new_initial_state()
        advance_chance_nodes(st_c, np.random.RandomState(seed_h))
        f_c = set(); dec_c = 0
        path_c = []
        hand_used_b = False
        while not st_c.is_terminal():
            if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h)); continue
            cp = st_c.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
            if cp == seat:
                agg_val = mc_old.get_agg_index(cp)
                chose_b = (agg_val <= 1.25)
                if chose_b: hand_used_b = True
                a = mc_old.act(st_c, cp, l6, f_c, seed_h, dec_c)
                dec_c += 1
            else:
                a = opps_c[cp if cp<seat else cp-1].select_action(st_c, cp, l6)
                mc_old.update_public_history(cp, a)
            if a == 0: f_c.add(cp)
            eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, a)
            path_c.append((cp, a, eng_amt))
            st_c.apply_action(eng_amt)
        ret_c = st_c.returns()[seat] / 100.0
        p2_c_rets.append(ret_c)

        if path_a != path_c:
            p2_diverged_hands.append({
                "hand_id": h,
                "used_b": hand_used_b,
                "ret_a": ret_a,
                "ret_c": ret_c,
                "diff": ret_c - ret_a
            })

    tot_p2_diff = sum(p2_c_rets) - sum(p2_a_rets)
    print(f"  * Pool 2 전체 손익 차이: {tot_p2_diff:+.2f} BB ({tot_p2_diff/25:+.2f} bb/100)")
    print(f"  * 경로 불일치 핸드 수  : {len(p2_diverged_hands)}개 (전체 2,500핸드 중)")
    for dh in p2_diverged_hands[:10]:
        print(f"    - Hand #{dh['hand_id']:4d}: Model B 사용={dh['used_b']} | Ret A={dh['ret_a']:+.2f}, Ret C={dh['ret_c']:+.2f} (Diff={dh['diff']:+.2f} BB)")

    # =================================================================
    # PHASE 2: Pool 3 동적 전략 실패 원인 500핸드 구간별 정밀 분석
    # =================================================================
    print("\n[*] Phase 2: Analyzing Pool 3 Dynamic Failure Across 500-Hand Intervals...")
    base_seed_p3 = 20380001
    mc_diag = ModelC_OldController(ma, mb)

    interval_stats = defaultdict(lambda: {"a_ret": 0.0, "b_ret": 0.0, "c_ret": 0.0, "c_chose_a": 0, "c_chose_b": 0, "agg_indices": []})

    for h in range(2500):
        interval_idx = h // 500
        seed_h = base_seed_p3 + h
        seat = h % 5

        # A
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        opps_a = get_pool_dynamic(seed_h)
        f_a = set(); dec_a = 0
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
            cp = st_a.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
            a = ma.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
            if cp == seat: dec_a += 1
            if a == 0: f_a.add(cp)
            st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
        ret_a = st_a.returns()[seat] / 100.0

        # B
        st_b = game.new_initial_state()
        advance_chance_nodes(st_b, np.random.RandomState(seed_h))
        opps_b = get_pool_dynamic(seed_h)
        f_b = set(); dec_b = 0
        while not st_b.is_terminal():
            if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(seed_h)); continue
            cp = st_b.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
            a = mb.act(st_b, cp, l6, f_b, seed_h, dec_b) if cp == seat else opps_b[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
            if cp == seat: dec_b += 1
            if a == 0: f_b.add(cp)
            st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
        ret_b = st_b.returns()[seat] / 100.0

        # C
        st_c = game.new_initial_state()
        advance_chance_nodes(st_c, np.random.RandomState(seed_h))
        opps_c = get_pool_dynamic(seed_h)
        f_c = set(); dec_c = 0
        while not st_c.is_terminal():
            if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h)); continue
            cp = st_c.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
            if cp == seat:
                agg = mc_diag.get_agg_index(cp)
                interval_stats[interval_idx]["agg_indices"].append(agg)
                if agg > 1.25: interval_stats[interval_idx]["c_chose_a"] += 1
                else: interval_stats[interval_idx]["c_chose_b"] += 1
                a = mc_diag.act(st_c, cp, l6, f_c, seed_h, dec_c)
                dec_c += 1
            else:
                a = opps_c[cp if cp<seat else cp-1].select_action(st_c, cp, l6)
                mc_diag.update_public_history(cp, a)
            if a == 0: f_c.add(cp)
            st_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, a))
        ret_c = st_c.returns()[seat] / 100.0

        interval_stats[interval_idx]["a_ret"] += ret_a
        interval_stats[interval_idx]["b_ret"] += ret_b
        interval_stats[interval_idx]["c_ret"] += ret_c

    print(f"  {'구간 (500H)':>12} | {'상대 1 실제모드':>14} | {'관측 AggIndex':>12} | {'Model A (bb/100)':>16} | {'Model B (bb/100)':>16} | {'Model C (bb/100)':>16} | {'C의 A/B 선택 비율'}")
    print("-" * 110)
    for i in range(5):
        st_item = interval_stats[i]
        real_mode = "UltraAggressive" if i % 2 == 0 else "TightPassive"
        avg_agg = np.mean(st_item["agg_indices"]) if st_item["agg_indices"] else 1.0
        bb_a = (st_item["a_ret"] / 500) * 100
        bb_b = (st_item["b_ret"] / 500) * 100
        bb_c = (st_item["c_ret"] / 500) * 100
        tot_choices = st_item["c_chose_a"] + st_item["c_chose_b"]
        ratio_b = (st_item["c_chose_b"] / tot_choices * 100) if tot_choices > 0 else 0
        print(f"  {f'{i*500}~{(i+1)*500}H':>12} | {real_mode:>14} | {avg_agg:12.2f} | {bb_a:+16.2f} | {bb_b:+16.2f} | {bb_c:+16.2f} | A={st_item['c_chose_a']} / B={st_item['c_chose_b']} (B:{ratio_b:.1f}%)")

    # =================================================================
    # PHASE 4: 4자 챔피언십 (Model A vs B vs C vs D)
    # =================================================================
    print("\n[*] Phase 4: 4-Way Championship (Model A vs B vs C vs D across 3 Pools)...")
    pools = [
        ("Pool 1: Standard (CallingStation 포함)", get_pool_standard, 20390001),
        ("Pool 2: Aggressive Red-Team (CS 배제)", get_pool_aggressive, 20400001),
        ("Pool 3: Dynamic Switching (적응형 상대)", get_pool_dynamic, 20410001),
    ]

    champ_results = {}
    for pool_title, pool_gen, seed_base in pools:
        print(f"\n--- Testing 4 Models on {pool_title} (2,500 hands) ---")
        t_start_pool = time.time()
        md_test = ModelD_RobustController(ma, mb, window_size=100)
        mc_test = ModelC_OldController(ma, mb)

        rets = {"A": [], "B": [], "C": [], "D": []}

        for h in range(2500):
            seed_h = seed_base + h
            seat = h % 5

            # A
            st_a = game.new_initial_state()
            advance_chance_nodes(st_a, np.random.RandomState(seed_h))
            opps = pool_gen(seed_h)
            f = set(); dec = 0
            while not st_a.is_terminal():
                if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
                cp = st_a.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                a = ma.act(st_a, cp, l6, f, seed_h, dec) if cp == seat else opps[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
                if cp == seat: dec += 1
                if a == 0: f.add(cp)
                st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
            rets["A"].append(st_a.returns()[seat] / 100.0)

            # B
            st_b = game.new_initial_state()
            advance_chance_nodes(st_b, np.random.RandomState(seed_h))
            opps = pool_gen(seed_h)
            f = set(); dec = 0
            while not st_b.is_terminal():
                if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(seed_h)); continue
                cp = st_b.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                a = mb.act(st_b, cp, l6, f, seed_h, dec) if cp == seat else opps[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
                if cp == seat: dec += 1
                if a == 0: f.add(cp)
                st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
            rets["B"].append(st_b.returns()[seat] / 100.0)

            # C
            st_c = game.new_initial_state()
            advance_chance_nodes(st_c, np.random.RandomState(seed_h))
            opps = pool_gen(seed_h)
            f = set(); dec = 0
            while not st_c.is_terminal():
                if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h)); continue
                cp = st_c.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
                if cp == seat:
                    a = mc_test.act(st_c, cp, l6, f, seed_h, dec); dec += 1
                else:
                    a = opps[cp if cp<seat else cp-1].select_action(st_c, cp, l6)
                    mc_test.update_public_history(cp, a)
                if a == 0: f.add(cp)
                st_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, a))
            rets["C"].append(st_c.returns()[seat] / 100.0)

            # D
            st_d = game.new_initial_state()
            advance_chance_nodes(st_d, np.random.RandomState(seed_h))
            opps = pool_gen(seed_h)
            f = set(); dec = 0
            while not st_d.is_terminal():
                if st_d.is_chance_node(): advance_chance_nodes(st_d, np.random.RandomState(seed_h)); continue
                cp = st_d.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_d, cp)
                if cp == seat:
                    a = md_test.act(st_d, cp, l6, f, seed_h, dec); dec += 1
                else:
                    a = opps[cp if cp<seat else cp-1].select_action(st_d, cp, l6)
                    md_test.update_public_history(cp, a)
                if a == 0: f.add(cp)
                st_d.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_d, cp, a))
            rets["D"].append(st_d.returns()[seat] / 100.0)

        m_a = np.mean(rets["A"])*100; se_a = stats.sem(rets["A"])*100
        m_b = np.mean(rets["B"])*100; se_b = stats.sem(rets["B"])*100
        m_c = np.mean(rets["C"])*100; se_c = stats.sem(rets["C"])*100
        m_d = np.mean(rets["D"])*100; se_d = stats.sem(rets["D"])*100

        diff_da = [(d - a)*100 for a, d in zip(rets["A"], rets["D"])]
        m_da = np.mean(diff_da); se_da = stats.sem(diff_da)
        p_da = stats.t.sf(abs(m_da/se_da), df=2499)*2 if se_da>0 else 1.0

        print(f"  * Model A (Always Temp)    : {m_a:+7.2f} ± {se_a:5.2f} bb/100")
        print(f"  * Model B (Always Q-Learn) : {m_b:+7.2f} ± {se_b:5.2f} bb/100")
        print(f"  * Model C (Old Aggress-Ctrl): {m_c:+7.2f} ± {se_c:5.2f} bb/100")
        print(f"  * Model D (Robust FailSafe): {m_d:+7.2f} ± {se_d:5.2f} bb/100 (D - A: {m_da:+.2f} ± {se_da:.2f}, p={p_da:.4f})")
        print(f"  * Model D 선택: A={md_test.choices_a}회, B={md_test.choices_b}회 [소요: {time.time()-t_start_pool:.1f}s]")

        champ_results[pool_title] = {
            "model_a": {"mean": float(m_a), "se": float(se_a)},
            "model_b": {"mean": float(m_b), "se": float(se_b)},
            "model_c": {"mean": float(m_c), "se": float(se_c)},
            "model_d": {"mean": float(m_d), "se": float(se_d)},
            "diff_da": {"mean": float(m_da), "se": float(se_da), "p": float(p_da)},
            "choices_d": {"a": md_test.choices_a, "b": md_test.choices_b}
        }
        if "Dynamic" in pool_title:
            pool3_d_intervals = {}
            for i in range(5):
                sub_d = rets["D"][i*500:(i+1)*500]
                pool3_d_intervals[f"{i*500}~{(i+1)*500}"] = float(np.mean(sub_d)*100)
            champ_results[pool_title]["model_d_intervals"] = pool3_d_intervals

    out_file = "v1012/data/v1012_audit_results.json"
    final_output = {
        "pool2_forensic": {
            "diverged_count": len(p2_diverged_hands),
            "diverged_samples": p2_diverged_hands[:10],
            "total_diff_bb": tot_p2_diff
        },
        "pool3_intervals": {str(k): {
            "a_ret_bb": v["a_ret"], "b_ret_bb": v["b_ret"], "c_ret_bb": v["c_ret"],
            "chose_a": v["c_chose_a"], "chose_b": v["c_chose_b"],
            "avg_agg": float(np.mean(v["agg_indices"])) if v["agg_indices"] else 1.0
        } for k, v in interval_stats.items()},
        "championship": champ_results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] V10.12 포렌식 감사 및 챔피언십 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
