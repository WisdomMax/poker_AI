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
from v91.run_v91_chip_flow_and_rl_ablation import get_hand_bucket
from v106.run_v106_deterministic_replay import DeterministicBalancedBot

# =====================================================================
# 100% DETERMINISTIC OPPONENTS WITH ISOLATED SEEDED RNG
# =====================================================================
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

class DeterministicTightAggressiveBot:
    def __init__(self, seed=None):
        pass
    def select_action(self, st, cp, legal_6: list):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        h_cards = list(np.where(inf[5:57] == 1.0)[0])
        b_cards = list(np.where(inf[57:109] == 1.0)[0])
        bucket = get_hand_bucket(h_cards, b_cards)
        if to_call > 0:
            if bucket in ["Monster", "Medium"]:
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if raises else 1
            return 0 if 0 in legal_6 else 1
        else:
            if bucket in ["Monster", "Medium"]:
                bets = [a for a in legal_6 if a in [2, 3]]
                return bets[0] if bets else 1
            return 1

class DeterministicBlindStealerBot:
    def __init__(self, seed):
        self.rng = random.Random(seed)
    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        if n_b == 0:
            if cp in [2, 3, 4] and to_call <= 100:
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if raises else 1
            if to_call == 0: return 1
            return 1 if to_call <= 100 else 0
        if to_call == 0:
            return 2 if 2 in legal_6 and self.rng.random() < 0.40 else 1
        return 1 if to_call <= 200 else 0

class DeterministicHyperAggressiveBot:
    def __init__(self, seed):
        self.rng = random.Random(seed)
    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        if n_b == 0:
            if to_call > 0 and 3 in legal_6 and self.rng.random() < 0.60:
                return 3
            raises = [a for a in legal_6 if a in [2, 3, 4]]
            return raises[-1] if raises and self.rng.random() < 0.70 else 1
        if to_call == 0:
            bets = [a for a in legal_6 if a in [2, 3, 4]]
            return bets[-1] if bets and self.rng.random() < 0.60 else 1
        raises = [a for a in legal_6 if a in [2, 3, 4]]
        return raises[-1] if raises and self.rng.random() < 0.50 else (1 if to_call <= 300 else 0)

# 상대 풀 생성기 (각 봇에 hand_seed 기반 독립 난수 주입)
def get_pool_standard(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        DeterministicTightPassiveBot(seed=seed_h*100+2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(seed=seed_h*100+4),
    ]

def get_pool_aggressive(seed_h):
    return [
        DeterministicTightAggressiveBot(seed=seed_h*100+1),
        DeterministicBalancedBot(rng_seed=seed_h*100+2),
        DeterministicBlindStealerBot(seed=seed_h*100+3),
        DeterministicHyperAggressiveBot(seed=seed_h*100+4),
    ]

def get_pool_dynamic(seed_h):
    return [
        DeterministicDynamicSwitchingBot(seed=seed_h*100+1, switch_interval=500),
        DeterministicTightAggressiveBot(seed=seed_h*100+2),
        DeterministicTightPassiveBot(seed=seed_h*100+3),
        DeterministicBalancedBot(rng_seed=seed_h*100+4),
    ]

# =====================================================================
# DETERMINISTIC AGENTS & ESTIMATOR
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

# =====================================================================
# SEMANTIC OPPONENT TRACKER & MODEL D (SEMANTIC FAIL-SAFE CONTROLLER)
# =====================================================================
class SemanticOpponentTracker:
    """
    True 6-Action 환경에서 무료 Check와 직면 베팅 Call을 정확히 구별하는 시맨틱 추적기
    - to_call == 0 일 때 action == 1 -> 무료 Check
    - to_call > 0 일 때 action == 1 -> 직면 Call
    - to_call > 0 일 때 action == 0 -> 직면 Fold
    - to_call > 0 일 때 action in [2,3,4,5] -> 직면 Raise
    """
    def __init__(self, window_size=50):
        self.history = defaultdict(lambda: deque(maxlen=window_size))

    def record_action(self, cp, to_call, action):
        if to_call > 0:
            act_type = 'call' if action == 1 else ('fold' if action == 0 else 'raise')
            self.history[cp].append({'context': 'facing_bet', 'action': act_type})
        else:
            act_type = 'check' if action == 1 else 'bet'
            self.history[cp].append({'context': 'unopened', 'action': act_type})

    def get_player_stats(self, cp):
        events = self.history[cp]
        facing = [e for e in events if e['context'] == 'facing_bet']
        unopened = [e for e in events if e['context'] == 'unopened']
        
        n_facing = len(facing)
        facing_call_rate = sum(1 for e in facing if e['action'] == 'call') / float(n_facing) if n_facing > 0 else 0.0
        facing_fold_rate = sum(1 for e in facing if e['action'] == 'fold') / float(n_facing) if n_facing > 0 else 0.0
        facing_raise_rate = sum(1 for e in facing if e['action'] == 'raise') / float(n_facing) if n_facing > 0 else 0.0
        
        n_unopened = len(unopened)
        vol_bet_rate = sum(1 for e in unopened if e['action'] == 'bet') / float(n_unopened) if n_unopened > 0 else 0.0
        
        return {
            "n_facing": n_facing,
            "facing_call_rate": facing_call_rate,
            "facing_fold_rate": facing_fold_rate,
            "facing_raise_rate": facing_raise_rate,
            "vol_bet_rate": vol_bet_rate
        }

    def has_calling_station(self, hero_cp, min_samples=10):
        for p in self.history:
            if p == hero_cp: continue
            stats = self.get_player_stats(p)
            if stats["n_facing"] >= min_samples:
                # 순수 콜링스테이션 판별 기준:
                # 1. 벳 직면 시 콜 비율 >= 70%
                # 2. 벳 직면 시 폴드 비율 <= 20%
                # 3. 벳 직면 시 레이즈 비율 <= 15%
                if (stats["facing_call_rate"] >= 0.70 and 
                    stats["facing_fold_rate"] <= 0.20 and 
                    stats["facing_raise_rate"] <= 0.15):
                    return True
        return False

class ModelD_SemanticController:
    """
    Model D: 시맨틱 콜링스테이션 게이트 + Default-to-A 안전장치 탑재 컨트롤러
    """
    def __init__(self, model_a: ModelA, model_b: ModelB, window_size=50):
        self.model_a = model_a
        self.model_b = model_b
        self.tracker = SemanticOpponentTracker(window_size=window_size)
        self.choices_a = 0
        self.choices_b = 0

    def update_action(self, cp, to_call, action):
        self.tracker.record_action(cp, to_call, action)

    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        if self.tracker.has_calling_station(cp, min_samples=10):
            self.choices_b += 1
            return self.model_b.act(st, cp, l6, folded, hand_seed, dec_idx)
        else:
            self.choices_a += 1
            return self.model_a.act(st, cp, l6, folded, hand_seed, dec_idx)

# =====================================================================
# MAIN EXPERIMENT EXECUTION (PHASE 1 ~ 4)
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.13: DETERMINISTIC REPLAY AUDIT & ROBUST CONTROLLER VALIDATION")
    print("=" * 80)

    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Loaded Q-table: {len(q.q_table)} states")

    ma = ModelA(bp, sims=300)
    mb = ModelB(bp, q, sims=300)

    # =================================================================
    # PHASE 1: 결정론적 재현성 감사 (Zero-Drift Replay Integrity, 1,000 hands)
    # =================================================================
    print("\n[*] Phase 1: Deterministic Replay Integrity Audit (Pool 2, 1,000 hands)...")
    seed_p1 = 20500001
    md_p1 = ModelD_SemanticController(ma, mb, window_size=50)

    diff_hands = 0
    tot_a_ret = 0.0
    tot_d_ret = 0.0

    for h in range(1000):
        seed_h = seed_p1 + h
        seat = h % 5

        # 1. Run Model A
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

        # 2. Run Model D
        opps_d = get_pool_aggressive(seed_h)
        st_d = game.new_initial_state()
        advance_chance_nodes(st_d, np.random.RandomState(seed_h))
        f_d = set(); dec_d = 0
        path_d = []
        while not st_d.is_terminal():
            if st_d.is_chance_node(): advance_chance_nodes(st_d, np.random.RandomState(seed_h)); continue
            cp = st_d.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_d, cp)
            sp = extract_player_spent(st_d); tc = max(sp) - sp[cp]
            if cp == seat:
                a = md_p1.act(st_d, cp, l6, f_d, seed_h, dec_d)
                dec_d += 1
            else:
                a = opps_d[cp if cp<seat else cp-1].select_action(st_d, cp, l6)
                md_p1.update_action(cp, tc, a)
            if a == 0: f_d.add(cp)
            eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_d, cp, a)
            path_d.append((cp, a, eng_amt))
            st_d.apply_action(eng_amt)
        ret_d = st_d.returns()[seat] / 100.0

        tot_a_ret += ret_a
        tot_d_ret += ret_d
        if path_a != path_d:
            diff_hands += 1

    print(f"  * Model A 총 손익: {tot_a_ret:+.2f} BB ({tot_a_ret/10:+.2f} bb/100)")
    print(f"  * Model D 총 손익: {tot_d_ret:+.2f} BB ({tot_d_ret/10:+.2f} bb/100)")
    print(f"  * 두 모델 간 손익 차이: {tot_d_ret - tot_a_ret:+.4f} BB (bb/100 차이: {(tot_d_ret - tot_a_ret)/10:+.4f})")
    print(f"  * 의사결정 경로 불일치 핸드 수: {diff_hands} / 1000개")
    print(f"  * Model D 정책 선택: A={md_p1.choices_a}회, B={md_p1.choices_b}회")
    if diff_hands == 0:
        print("  -> [성공] 결정론적 무손실 완벽 재현(Zero-Drift Replay) 달성 확인!")
    else:
        print("  -> [경고] 여전히 불일치가 발생하고 있습니다.")

    # =================================================================
    # PHASE 3: CallingStation 시맨틱 판별기 실측 감사
    # =================================================================
    print("\n[*] Phase 3: Semantic CallingStation Metrics Audit...")
    pools_for_metric = [
        ("Pool 1: Standard", get_pool_standard),
        ("Pool 2: Aggressive Red-Team", get_pool_aggressive),
        ("Pool 3: Dynamic Switching", get_pool_dynamic),
    ]
    semantic_audit_results = {}
    for p_name, p_gen in pools_for_metric:
        tracker = SemanticOpponentTracker(window_size=50)
        for h in range(500):
            seed_h = 20505001 + h
            seat = h % 5
            opps = p_gen(seed_h)
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, np.random.RandomState(seed_h)); continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                sp = extract_player_spent(st); tc = max(sp) - sp[cp]
                a = ma.act(st, cp, l6, None, seed_h, 0) if cp == seat else opps[cp if cp<seat else cp-1].select_action(st, cp, l6)
                if cp != seat:
                    tracker.record_action(cp, tc, a)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, a))
        
        pool_stats = {}
        detected_cs = False
        for p in range(5):
            if p == seat: continue
            st_p = tracker.get_player_stats(p)
            pool_stats[f"Player_{p}"] = st_p
            if tracker.has_calling_station(seat, min_samples=10):
                detected_cs = True
        semantic_audit_results[p_name] = {
            "stats": pool_stats,
            "detected_calling_station": detected_cs
        }
        print(f"  * {p_name}: CallingStation 감지 여부 = {detected_cs}")

    # =================================================================
    # PHASE 4: 5개 신규 독립 시드 대규모 챔피언십 (총 45,000 모델-핸드)
    # =================================================================
    print("\n[*] Phase 4: 5-Seed Large-Sample Championship (Model A vs B vs D)...")
    seeds = [20510001, 20520001, 20530001, 20540001, 20550001]
    pools = [
        ("Pool 1: Standard (CS 포함)", get_pool_standard),
        ("Pool 2: Aggressive Red-Team (CS 배제)", get_pool_aggressive),
        ("Pool 3: Dynamic Switching (적응형 상대)", get_pool_dynamic),
    ]

    championship_data = defaultdict(lambda: {"A": [], "B": [], "D": [], "D_choices_a": 0, "D_choices_b": 0})
    seed_pool_details = defaultdict(dict)

    total_start_time = time.time()
    for s_idx, s_base in enumerate(seeds, 1):
        print(f"\n--- [Seed {s_idx}/5: Base Seed {s_base}] ---")
        t_seed_start = time.time()
        for pool_title, pool_gen in pools:
            md = ModelD_SemanticController(ma, mb, window_size=50)
            rets_a = []; rets_b = []; rets_d = []
            for h in range(1000):
                seed_h = s_base + h
                seat = h % 5

                # A
                opps_a = pool_gen(seed_h)
                st_a = game.new_initial_state()
                advance_chance_nodes(st_a, np.random.RandomState(seed_h))
                f_a = set(); dec_a = 0
                while not st_a.is_terminal():
                    if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
                    cp = st_a.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                    a = ma.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
                    if cp == seat: dec_a += 1
                    if a == 0: f_a.add(cp)
                    st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
                rets_a.append(st_a.returns()[seat] / 100.0)

                # B
                opps_b = pool_gen(seed_h)
                st_b = game.new_initial_state()
                advance_chance_nodes(st_b, np.random.RandomState(seed_h))
                f_b = set(); dec_b = 0
                while not st_b.is_terminal():
                    if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(seed_h)); continue
                    cp = st_b.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                    a = mb.act(st_b, cp, l6, f_b, seed_h, dec_b) if cp == seat else opps_b[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
                    if cp == seat: dec_b += 1
                    if a == 0: f_b.add(cp)
                    st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
                rets_b.append(st_b.returns()[seat] / 100.0)

                # D
                opps_d = pool_gen(seed_h)
                st_d = game.new_initial_state()
                advance_chance_nodes(st_d, np.random.RandomState(seed_h))
                f_d = set(); dec_d = 0
                while not st_d.is_terminal():
                    if st_d.is_chance_node(): advance_chance_nodes(st_d, np.random.RandomState(seed_h)); continue
                    cp = st_d.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_d, cp)
                    sp = extract_player_spent(st_d); tc = max(sp) - sp[cp]
                    if cp == seat:
                        a = md.act(st_d, cp, l6, f_d, seed_h, dec_d)
                        dec_d += 1
                    else:
                        a = opps_d[cp if cp<seat else cp-1].select_action(st_d, cp, l6)
                        md.update_action(cp, tc, a)
                    if a == 0: f_d.add(cp)
                    st_d.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_d, cp, a))
                rets_d.append(st_d.returns()[seat] / 100.0)

            championship_data[pool_title]["A"].extend(rets_a)
            championship_data[pool_title]["B"].extend(rets_b)
            championship_data[pool_title]["D"].extend(rets_d)
            championship_data[pool_title]["D_choices_a"] += md.choices_a
            championship_data[pool_title]["D_choices_b"] += md.choices_b

            m_a_seed = np.mean(rets_a) * 100
            m_b_seed = np.mean(rets_b) * 100
            m_d_seed = np.mean(rets_d) * 100
            seed_pool_details[f"Seed_{s_idx}"][pool_title] = {
                "Model_A": m_a_seed, "Model_B": m_b_seed, "Model_D": m_d_seed,
                "D_A_diff": m_d_seed - m_a_seed
            }
            print(f"  * {pool_title:<38}: A={m_a_seed:+7.2f} | B={m_b_seed:+7.2f} | D={m_d_seed:+7.2f} (D-A: {m_d_seed-m_a_seed:+6.2f}) [D 선택: A={md.choices_a}, B={md.choices_b}]")
        print(f"  [Seed {s_idx} 완료 소요 시간: {time.time()-t_seed_start:.1f}s]")

    # =================================================================
    # STATISTICAL SYNTHESIS
    # =================================================================
    print("\n" + "=" * 80)
    print("V10.13 5-SEED CHAMPIONSHIP FINAL STATISTICAL SYNTHESIS (5,000 Hands per Pool)")
    print("=" * 80)

    final_pool_results = {}
    tot_rets_a = []; tot_rets_b = []; tot_rets_d = []

    for pool_title in pools:
        p_name = pool_title[0]
        data = championship_data[p_name]
        ra = data["A"]; rb = data["B"]; rd = data["D"]
        tot_rets_a.extend(ra); tot_rets_b.extend(rb); tot_rets_d.extend(rd)

        m_a = np.mean(ra)*100; se_a = stats.sem(ra)*100
        m_b = np.mean(rb)*100; se_b = stats.sem(rb)*100
        m_d = np.mean(rd)*100; se_d = stats.sem(rd)*100

        diff_da = [(d - a)*100 for a, d in zip(ra, rd)]
        m_da = np.mean(diff_da); se_da = stats.sem(diff_da)
        ci_95_da = (m_da - 1.96*se_da, m_da + 1.96*se_da)
        p_da = stats.t.sf(abs(m_da/se_da), df=len(diff_da)-1)*2 if se_da>0 else 1.0

        diff_db = [(d - b)*100 for b, d in zip(rb, rd)]
        m_db = np.mean(diff_db); se_db = stats.sem(diff_db)
        ci_95_db = (m_db - 1.96*se_db, m_db + 1.96*se_db)
        p_db = stats.t.sf(abs(m_db/se_db), df=len(diff_db)-1)*2 if se_db>0 else 1.0

        print(f"\n[{p_name}] (5,000 Hands)")
        print(f"  - Model A (Always Temp)    : {m_a:+7.2f} ± {se_a:5.2f} bb/100")
        print(f"  - Model B (Always Q-Learn) : {m_b:+7.2f} ± {se_b:5.2f} bb/100")
        print(f"  - Model D (Robust Semantic): {m_d:+7.2f} ± {se_d:5.2f} bb/100")
        print(f"  - D vs A 차이: {m_da:+6.2f} ± {se_da:5.2f} bb/100 (95% CI: [{ci_95_da[0]:+.2f}, {ci_95_da[1]:+.2f}], p={p_da:.4f})")
        print(f"  - D vs B 차이: {m_db:+6.2f} ± {se_db:5.2f} bb/100 (95% CI: [{ci_95_db[0]:+.2f}, {ci_95_db[1]:+.2f}], p={p_db:.4f})")
        print(f"  - Model D 선택 비율: A={data['D_choices_a']}회, B={data['D_choices_b']}회")

        final_pool_results[p_name] = {
            "model_a": {"mean": float(m_a), "se": float(se_a)},
            "model_b": {"mean": float(m_b), "se": float(se_b)},
            "model_d": {"mean": float(m_d), "se": float(se_d)},
            "diff_da": {"mean": float(m_da), "se": float(se_da), "ci95": [float(ci_95_da[0]), float(ci_95_da[1])], "p": float(p_da)},
            "diff_db": {"mean": float(m_db), "se": float(se_db), "ci95": [float(ci_95_db[0]), float(ci_95_db[1])], "p": float(p_db)},
            "choices_d": {"a": data["D_choices_a"], "b": data["D_choices_b"]}
        }

    # 전체 풀 통합 평균
    m_a_tot = np.mean(tot_rets_a)*100; se_a_tot = stats.sem(tot_rets_a)*100
    m_b_tot = np.mean(tot_rets_b)*100; se_b_tot = stats.sem(tot_rets_b)*100
    m_d_tot = np.mean(tot_rets_d)*100; se_d_tot = stats.sem(tot_rets_d)*100
    diff_tot_da = [(d - a)*100 for a, d in zip(tot_rets_a, tot_rets_d)]
    m_tot_da = np.mean(diff_tot_da); se_tot_da = stats.sem(diff_tot_da)
    p_tot_da = stats.t.sf(abs(m_tot_da/se_tot_da), df=len(diff_tot_da)-1)*2 if se_tot_da>0 else 1.0

    print("\n" + "=" * 80)
    print("3대 풀 15,000 모델-핸드 전체 통합 최종 결과")
    print(f"  * Model A (Always Temp)    : {m_a_tot:+7.2f} ± {se_a_tot:5.2f} bb/100")
    print(f"  * Model B (Always Q-Learn) : {m_b_tot:+7.2f} ± {se_b_tot:5.2f} bb/100")
    print(f"  * Model D (Robust Semantic): {m_d_tot:+7.2f} ± {se_d_tot:5.2f} bb/100")
    print(f"  * D vs A 통합 격차: {m_tot_da:+6.2f} ± {se_tot_da:5.2f} bb/100 (p={p_tot_da:.4f})")
    print("=" * 80)

    out_file = "v1013/data/v1013_validation_results.json"
    final_output = {
        "phase1_replay_audit": {
            "total_hands": 1000,
            "diverged_hands": diff_hands,
            "tot_a_ret_bb": tot_a_ret,
            "tot_d_ret_bb": tot_d_ret,
            "diff_bb": tot_d_ret - tot_a_ret,
            "d_choices": {"a": md_p1.choices_a, "b": md_p1.choices_b}
        },
        "phase3_semantic_metrics": semantic_audit_results,
        "seed_pool_details": seed_pool_details,
        "championship_pools": final_pool_results,
        "championship_total": {
            "model_a": {"mean": float(m_a_tot), "se": float(se_a_tot)},
            "model_b": {"mean": float(m_b_tot), "se": float(se_b_tot)},
            "model_d": {"mean": float(m_d_tot), "se": float(se_d_tot)},
            "diff_da": {"mean": float(m_tot_da), "se": float(se_tot_da), "p": float(p_tot_da)}
        },
        "total_runtime_seconds": time.time() - total_start_time
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] V10.13 검증 및 챔피언십 전체 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
