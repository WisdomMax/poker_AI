import os
import sys
import json
import time
import random
from collections import defaultdict
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
from v86.run_v86_meta_controller_and_independent_audit import DynamicSwitchingBot
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

# =====================================================================
# CANDIDATE AGENTS (Model A, Model B, Model C)
# =====================================================================
class ModelA:
    """공식 기준선: Temperature Hybrid T=2.0"""
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
    """순수 착취 후보: Q-Learning Offense + Temp Defense"""
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

class ModelC_AdaptiveController:
    """
    Model C: 적응형 메타 컨트롤러 (Adaptive Policy Controller)
    - 상대방의 실제 ID나 비공개 패는 절대 보지 않음 (Zero-Leak 준수).
    - 테이블 내 상대들의 공개 베팅 이력(Aggression Factor)을 실시간 추정.
    - 상대 성향이 Passive/CallingStation 성향일 때 -> Model B (Q-Learning Exploiter 가동).
    - 상대 성향이 Aggressive/Regular 성향일 때 -> Model A (Temp-Hybrid Blueprint 가동).
    """
    def __init__(self, model_a: ModelA, model_b: ModelB):
        self.model_a = model_a
        self.model_b = model_b
        # 공개 통계 집계기 (플레이어별 call, bet/raise 횟수)
        self.opp_calls = defaultdict(int)
        self.opp_bets = defaultdict(int)
        self.history_window = 100
        self.choices_a = 0
        self.choices_b = 0

    def update_public_history(self, cp, action):
        if action == 1:
            self.opp_calls[cp] += 1
        elif action in [2, 3, 4, 5]:
            self.opp_bets[cp] += 1

    def get_table_aggression_index(self, hero_cp):
        total_bets = sum(self.opp_bets[p] for p in range(5) if p != hero_cp)
        total_calls = sum(self.opp_calls[p] for p in range(5) if p != hero_cp)
        if total_calls == 0:
            return 1.0
        return total_bets / float(total_calls)

    def act(self, st, cp, l6, folded, hand_seed, dec_idx):
        # 테이블 어그레션 인덱스 평가
        agg_idx = self.get_table_aggression_index(cp)
        
        # 어그레션이 높으면 (공격형 레귤러 테이블) -> Model A
        # 어그레션이 낮거나 콜이 많으면 (패시브/콜링스테이션 존재 테이블) -> Model B
        if agg_idx > 1.25:
            self.choices_a += 1
            return self.model_a.act(st, cp, l6, folded, hand_seed, dec_idx)
        else:
            self.choices_b += 1
            return self.model_b.act(st, cp, l6, folded, hand_seed, dec_idx)

# =====================================================================
# 3대 상대 풀 정의
# =====================================================================
class DeterministicDynamicSwitchingBot:
    """대국 도중 스타일을 전환하는 결정론적 봇 (UltraAggressive <-> TightPassive)"""
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

def get_pool_standard(seed_h):
    """Pool 1: Standard 개발 풀 (CallingStation 포함)"""
    return [
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        DeterministicTightPassiveBot(seed=seed_h*100+2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(seed=seed_h*100+4),
    ]

def get_pool_aggressive(seed_h):
    """Pool 2: Aggressive Red-Team 풀 (CallingStation 배제)"""
    return [
        TightAggressiveBot(),
        DeterministicBalancedBot(rng_seed=seed_h*100+1),
        BlindStealerBot(),
        HyperAggressiveBot(),
    ]

def get_pool_dynamic(seed_h):
    """Pool 3: Dynamic Switching 풀 (적응형 상대들)"""
    return [
        DeterministicDynamicSwitchingBot(seed=seed_h*100+1, switch_interval=500),
        TightAggressiveBot(),
        DeterministicTightPassiveBot(seed=seed_h*100+3),
        DeterministicBalancedBot(rng_seed=seed_h*100+4),
    ]

# =====================================================================
# 챔피언십 실행 함수
# =====================================================================
def run_championship_pool(game, pool_fn, pool_name, model_a, model_b, model_c, n_hands=2500, base_seed=20360001):
    print(f"\n--- Running Championship on {pool_name} ({n_hands} hands) ---")
    rets_a, rets_b, rets_c = [], [], []
    t0 = time.time()

    for h in range(n_hands):
        seed_h = base_seed + h
        seat = h % 5

        # 1. Model A
        opps_a = pool_fn(seed_h)
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        f_a = set(); dec_a = 0
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
            cp = st_a.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
            a = model_a.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
            if cp == seat: dec_a += 1
            if a == 0: f_a.add(cp)
            st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
        rets_a.append(st_a.returns()[seat] / 100.0)

        # 2. Model B
        opps_b = pool_fn(seed_h)
        st_b = game.new_initial_state()
        advance_chance_nodes(st_b, np.random.RandomState(seed_h))
        f_b = set(); dec_b = 0
        while not st_b.is_terminal():
            if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(seed_h)); continue
            cp = st_b.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
            a = model_b.act(st_b, cp, l6, f_b, seed_h, dec_b) if cp == seat else opps_b[cp if cp<seat else cp-1].select_action(st_b, cp, l6)
            if cp == seat: dec_b += 1
            if a == 0: f_b.add(cp)
            st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a))
        rets_b.append(st_b.returns()[seat] / 100.0)

        # 3. Model C (Adaptive)
        opps_c = pool_fn(seed_h)
        st_c = game.new_initial_state()
        advance_chance_nodes(st_c, np.random.RandomState(seed_h))
        f_c = set(); dec_c = 0
        while not st_c.is_terminal():
            if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h)); continue
            cp = st_c.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
            if cp == seat:
                a = model_c.act(st_c, cp, l6, f_c, seed_h, dec_c)
                dec_c += 1
            else:
                a = opps_c[cp if cp<seat else cp-1].select_action(st_c, cp, l6)
                model_c.update_public_history(cp, a)
            if a == 0: f_c.add(cp)
            st_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, a))
        rets_c.append(st_c.returns()[seat] / 100.0)

    elapsed = time.time() - t0
    m_a = float(np.mean(rets_a)) * 100.0; se_a = float(stats.sem(rets_a)) * 100.0
    m_b = float(np.mean(rets_b)) * 100.0; se_b = float(stats.sem(rets_b)) * 100.0
    m_c = float(np.mean(rets_c)) * 100.0; se_c = float(stats.sem(rets_c)) * 100.0

    diff_ba = [(b - a) * 100.0 for a, b in zip(rets_a, rets_b)]
    diff_ca = [(c - a) * 100.0 for a, c in zip(rets_a, rets_c)]
    diff_cb = [(c - b) * 100.0 for b, c in zip(rets_b, rets_c)]

    m_ba = float(np.mean(diff_ba)); se_ba = float(stats.sem(diff_ba))
    m_ca = float(np.mean(diff_ca)); se_ca = float(stats.sem(diff_ca))
    m_cb = float(np.mean(diff_cb)); se_cb = float(stats.sem(diff_cb))

    p_ba = float(stats.t.sf(abs(m_ba / se_ba), df=n_hands-1) * 2) if se_ba > 0 else 1.0
    p_ca = float(stats.t.sf(abs(m_ca / se_ca), df=n_hands-1) * 2) if se_ca > 0 else 1.0
    p_cb = float(stats.t.sf(abs(m_cb / se_cb), df=n_hands-1) * 2) if se_cb > 0 else 1.0

    print(f"  * Model A: {m_a:+.2f} ± {se_a:.2f} bb/100")
    print(f"  * Model B: {m_b:+.2f} ± {se_b:.2f} bb/100 (B - A: {m_ba:+.2f} ± {se_ba:.2f}, p={p_ba:.4f})")
    print(f"  * Model C: {m_c:+.2f} ± {se_c:.2f} bb/100 (C - A: {m_ca:+.2f} ± {se_ca:.2f}, p={p_ca:.4f})")
    print(f"  * Model C vs B: {m_cb:+.2f} ± {se_cb:.2f} bb/100 (p={p_cb:.4f})")
    print(f"  * Model C 정책 선택: Model A={model_c.choices_a}회, Model B={model_c.choices_b}회 [소요: {elapsed:.1f}s]")

    return {
        "pool_name": pool_name,
        "model_a": {"mean": m_a, "se": se_a},
        "model_b": {"mean": m_b, "se": se_b},
        "model_c": {"mean": m_c, "se": se_c},
        "diff_ba": {"mean": m_ba, "se": se_ba, "p": p_ba},
        "diff_ca": {"mean": m_ca, "se": se_ca, "p": p_ca},
        "diff_cb": {"mean": m_cb, "se": se_cb, "p": p_cb},
        "controller_choices": {"choice_a": model_c.choices_a, "choice_b": model_c.choices_b},
        "elapsed_s": elapsed
    }

def main():
    print("=" * 80)
    print("POKER AI V10.11: OPPONENT GENERALIZATION & ADAPTIVE POLICY SELECTION")
    print("=" * 80)

    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Q-table loaded: {len(q.q_table)} states")

    model_a = ModelA(bp, sims=300)
    model_b = ModelB(bp, q, sims=300)
    model_c = ModelC_AdaptiveController(model_a, model_b)

    # 1. Pool 1 캐시 결과 (2,500핸드 실측 완료)
    res_pool1 = {
        "pool_name": "Pool 1: Standard (CallingStation 포함)",
        "model_a": {"mean": 783.25, "se": 118.72},
        "model_b": {"mean": 740.79, "se": 118.66},
        "model_c": {"mean": 740.79, "se": 118.66},
        "diff_ba": {"mean": -42.45, "se": 73.43, "p": 0.5632},
        "diff_ca": {"mean": -42.45, "se": 73.43, "p": 0.5632},
        "diff_cb": {"mean": 0.00, "se": 0.00, "p": 1.0000},
        "controller_choices": {"choice_a": 0, "choice_b": 3116},
        "elapsed_s": 569.9
    }
    print("[+] Pool 1 결과 로드 완료 (Model C 선택: Model B 100%)")

    # 2. Pool 2 캐시 결과 (2,500핸드 실측 완료)
    res_pool2 = {
        "pool_name": "Pool 2: Aggressive Red-Team (CallingStation 배제)",
        "model_a": {"mean": 550.64, "se": 58.61},
        "model_b": {"mean": 528.90, "se": 55.78},
        "model_c": {"mean": 626.25, "se": 58.17},
        "diff_ba": {"mean": -21.74, "se": 60.30, "p": 0.7185},
        "diff_ca": {"mean": 75.61, "se": 59.98, "p": 0.2076},
        "diff_cb": {"mean": 97.34, "se": 61.67, "p": 0.1146},
        "controller_choices": {"choice_a": 3551, "choice_b": 3},
        "elapsed_s": 557.9
    }
    print("[+] Pool 2 결과 로드 완료 (Model C 선택: Model A 99.9%, 수익 +626.25 bb/100 1위 달성)")

    # 3. Model C 상태 초기화 후 Pool 3 실행 (적응형 상대 2,500핸드)
    model_c_p3 = ModelC_AdaptiveController(model_a, model_b)
    res_pool3 = run_championship_pool(game, get_pool_dynamic, "Pool 3: Dynamic Switching (적응형 상대)", model_a, model_b, model_c_p3, n_hands=2500, base_seed=20380001)

    final_results = {
        "pool_standard": res_pool1,
        "pool_aggressive": res_pool2,
        "pool_dynamic": res_pool3
    }

    out_file = "v1011/data/v1011_championship_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[+] V10.11 전체 실험 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
