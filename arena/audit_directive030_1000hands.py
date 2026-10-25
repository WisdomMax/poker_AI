"""
DIRECTIVE 030: 1,000-Hand Independent AI Audit Script
=====================================================
Evaluates 4 AI models (Cycle 4, Cycle 13, Cycle 10+MC, Baseline)
across 1,000 independent hands.
Tracks:
- Preflop & Postflop All-in frequency
- Hand strength (Chen Formula & Cards) on All-in
- Pot size, facing bet, effective stack, position
- Opponent fold rate when All-in occurs
- Win rate & EV when All-in is called
- Exploitability & Net profitability of all-in lines
- Model policy action probability distribution (Action 0~5)
"""

import os
import sys
import re
import json
import time
import random
import numpy as np
import torch
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1023.v1023_13_differential_forensics import normalize_inf_state_v79
from arena.models_manager import ArenaModelManager

def chen_score(card1: str, card2: str) -> float:
    ranks = '23456789TJQKA'
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    v1, v2 = ranks.index(r1), ranks.index(r2)
    if v1 < v2:
        v1, v2 = v2, v1
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    
    score_map = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
                 '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
    score = score_map[r1]
    is_pair = (r1 == r2)
    if is_pair:
        score = max(5.0, score * 2)
    
    if s1 == s2:
        score += 2.0
        
    gap = v1 - v2 - 1
    if not is_pair:
        if gap == 0:
            pass
        elif gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        else:
            score -= 5.0
            
    if not is_pair and gap <= 1 and v1 < ranks.index('Q'):
        score += 1.0
        
    return round(score, 1)

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

def get_position_name(seat, btn_seat, num_players=5):
    # 5-player positions: BTN, SB, BB, UTG, CO
    offset = (seat - btn_seat) % num_players
    pos_map = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "CO"}
    return pos_map.get(offset, f"P{offset}")

def run_audit(num_hands=1000, seed_base=777000):
    print("=" * 70)
    print(f"DIRECTIVE 030: {num_hands}핸드 독립 AI 올인 및 정책 심층 감사 시작")
    print("=" * 70)
    
    mgr = ArenaModelManager.get_instance()
    game = get_5p_fullgame()
    
    # 4개 AI 모델 매핑 (Seat 1: Cycle 4, Seat 2: Cycle 13, Seat 3: Cycle 10+MC, Seat 4: Baseline)
    # Seat 0도 공정한 비교를 위해 순환 또는 표준 베이스라인 배치 (여기서는 5인 풀게임 시뮬레이션)
    # Seat 0: Baseline, Seat 1: Cycle 4, Seat 2: Cycle 13, Seat 3: Cycle 10+MC, Seat 4: Baseline
    # 또는 Seat 0~4에 4개 모델을 고르게 분배
    seat_models = {
        0: "cycle4",
        1: "cycle4",
        2: "cycle13",
        3: "cycle10_mc",
        4: "baseline"
    }
    
    model_names = {
        "cycle4": "Cycle 4 (1.2M)",
        "cycle13": "Cycle 13 (2.9M)",
        "cycle10_mc": "Cycle 10+MC (2.3M)",
        "baseline": "Baseline (100K)"
    }
    
    # 통계 저장 구조
    stats = {
        m: {
            "total_decisions": 0,
            "actions_chosen": Counter(),
            "probs_sum": np.zeros(6, dtype=np.float64),
            "allin_total": 0,
            "allin_preflop": 0,
            "allin_postflop": 0,
            "allin_chen_scores": [],
            "allin_cards": [],
            "allin_positions": Counter(),
            "allin_pot_sizes": [],
            "allin_facing_bets": [],
            "allin_called_count": 0,
            "allin_folded_count": 0,
            "allin_called_wins": 0,
            "allin_called_net_chips": 0.0,
            "allin_folded_net_chips": 0.0,
            "total_net_chips": 0.0,
            "allin_hand_net_chips": 0.0
        } for m in model_names.keys()
    }
    
    t_start = time.time()
    
    for h_idx in range(num_hands):
        h_seed = seed_base + h_idx
        chance_rng = np.random.RandomState(h_seed)
        btn_seat = h_idx % 5
        
        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)
        priv_cards = parse_cards_from_state(st)
        
        folded_seats = set()
        dec_idx = 0
        
        # 이번 핸드에서 올인한 플레이어 추적
        hand_allin_events = [] # dict per allin
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue
                
            cp = st.current_player()
            rnd = get_round_from_state(st)
            m_key = seat_models[cp]
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            
            spent = extract_player_spent(st)
            pot = sum(spent)
            curr_spent = spent[cp]
            max_spent = max(spent)
            facing_bet = max_spent - curr_spent
            pos = get_position_name(cp, btn_seat)
            cards = priv_cards.get(cp, ["??", "??"])
            c_score = chen_score(cards[0], cards[1]) if cards[0] != "??" else 0.0
            
            # 모델 정책 및 확률 계산
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            wrapper = mgr.wrappers.get(m_key)
            probs = wrapper.get_action_probs(inf, legal_6)
            
            # 액션 선택
            mc_used = False
            if m_key == "cycle10_mc" and rnd > 0:
                act = mgr.adaptive_guard.act(st, cp, legal_6, folded_seats, h_seed, dec_idx)
                mc_used = True
            else:
                act = wrapper.act(inf, legal_6)
                
            # 통계 기록
            st_data = stats[m_key]
            st_data["total_decisions"] += 1
            st_data["actions_chosen"][act] += 1
            st_data["probs_sum"] += np.array(probs, dtype=np.float64)
            
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            is_allin_action = (act == 5 or raw_act == 10000)
            
            if is_allin_action:
                st_data["allin_total"] += 1
                if rnd == 0:
                    st_data["allin_preflop"] += 1
                else:
                    st_data["allin_postflop"] += 1
                st_data["allin_chen_scores"].append(c_score)
                st_data["allin_cards"].append(f"{cards[0]}{cards[1]}")
                st_data["allin_positions"][pos] += 1
                st_data["allin_pot_sizes"].append(pot)
                st_data["allin_facing_bets"].append(facing_bet)
                
                hand_allin_events.append({
                    "seat": cp,
                    "model": m_key,
                    "street": rnd,
                    "pot_at_allin": pot,
                    "facing_bet": facing_bet,
                    "chen_score": c_score,
                    "cards": cards,
                    "active_opponents": [s for s in range(5) if s != cp and s not in folded_seats]
                })
                
            if act == 0:
                folded_seats.add(cp)
                
            st.apply_action(raw_act)
            dec_idx += 1
            
        # 핸드 종료 후 정산
        returns = st.returns()
        for s in range(5):
            m_key = seat_models[s]
            stats[m_key]["total_net_chips"] += returns[s]
            
        # 올인 이벤트 결과 추적
        for ev in hand_allin_events:
            cp = ev["seat"]
            m_key = ev["model"]
            opps = ev["active_opponents"]
            # 올인 이후 남은 상대들이 콜했는지, 전원 폴드했는지 확인
            # 쇼다운에 참여한 상대가 있는지
            showdown_opps = [o for o in opps if o not in folded_seats]
            net = returns[cp]
            stats[m_key]["allin_hand_net_chips"] += net
            
            if len(showdown_opps) == 0:
                # 전원 폴드 승리
                stats[m_key]["allin_folded_count"] += 1
                stats[m_key]["allin_folded_net_chips"] += net
            else:
                # 1명 이상 콜해서 쇼다운 진행
                stats[m_key]["allin_called_count"] += 1
                stats[m_key]["allin_called_net_chips"] += net
                if net > 0:
                    stats[m_key]["allin_called_wins"] += 1
                    
        if (h_idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{h_idx+1:4d}/{num_hands}] 완료 ({elapsed:.1f}s 경과)...", flush=True)
            
    elapsed_total = time.time() - t_start
    print(f"\n[1,000핸드 시뮬레이션 완료] 총 소요시간: {elapsed_total:.1f}초\n")
    
    # 결과 보고서 출력
    print("=" * 80)
    print("DIRECTIVE 030: 1,000-Hand All-in Forensic Audit Results")
    print("=" * 80)
    
    results = {}
    for m_key, m_name in model_names.items():
        s = stats[m_key]
        n_dec = s["total_decisions"]
        avg_probs = (s["probs_sum"] / n_dec) if n_dec > 0 else np.zeros(6)
        
        allin_tot = s["allin_total"]
        pf_allin = s["allin_preflop"]
        post_allin = s["allin_postflop"]
        allin_freq = (allin_tot / n_dec * 100) if n_dec > 0 else 0
        
        scores = s["allin_chen_scores"]
        avg_chen = np.mean(scores) if scores else 0.0
        trash_count = sum(1 for sc in scores if sc < 6.0) # Chen < 6 (Weak/Trash)
        trash_pct = (trash_count / len(scores) * 100) if scores else 0.0
        premium_count = sum(1 for sc in scores if sc >= 12.0)
        premium_pct = (premium_count / len(scores) * 100) if scores else 0.0
        
        called_tot = s["allin_called_count"]
        fold_tot = s["allin_folded_count"]
        opp_fold_pct = (fold_tot / (called_tot + fold_tot) * 100) if (called_tot + fold_tot) > 0 else 0
        
        win_when_called = (s["allin_called_wins"] / called_tot * 100) if called_tot > 0 else 0
        called_ev_bb = (s["allin_called_net_chips"] / 100.0 / called_tot) if called_tot > 0 else 0.0
        total_allin_net_bb = s["allin_hand_net_chips"] / 100.0
        total_ev_bb100 = (s["total_net_chips"] / 100.0 / num_hands * 100.0)
        
        results[m_key] = {
            "name": m_name,
            "decisions": n_dec,
            "avg_probs": avg_probs.tolist(),
            "actions_chosen": dict(s["actions_chosen"]),
            "allin_total": allin_tot,
            "allin_freq_pct": allin_freq,
            "allin_preflop": pf_allin,
            "allin_postflop": post_allin,
            "avg_chen": avg_chen,
            "trash_allin_pct": trash_pct,
            "premium_allin_pct": premium_pct,
            "opp_fold_pct": opp_fold_pct,
            "win_when_called_pct": win_when_called,
            "called_ev_bb": called_ev_bb,
            "total_allin_net_bb": total_allin_net_bb,
            "total_ev_bb100": total_ev_bb100,
            "positions": dict(s["allin_positions"]),
            "sample_cards": s["allin_cards"][:15]
        }
        
        print(f"\n--- [{m_name}] ---")
        print(f"  * 총 의사결정 수: {n_dec}회")
        print(f"  * 평균 정책 확률: Fold={avg_probs[0]*100:.1f}%, Call={avg_probs[1]*100:.1f}%, 50%={avg_probs[2]*100:.1f}%, 100%={avg_probs[3]*100:.1f}%, 200%={avg_probs[4]*100:.1f}%, All-in={avg_probs[5]*100:.1f}%")
        print(f"  * 실제 선택 액션: Fold={s['actions_chosen'][0]}, Call={s['actions_chosen'][1]}, 50%={s['actions_chosen'][2]}, 100%={s['actions_chosen'][3]}, 200%={s['actions_chosen'][4]}, All-in={s['actions_chosen'][5]}")
        print(f"  * 올인 빈도: 총 {allin_tot}회 ({allin_freq:.1f}%) | Preflop: {pf_allin}회, Postflop: {post_allin}회")
        print(f"  * 올인 당시 Chen 점수: 평균 {avg_chen:.1f}점 | 약한 패(Trash, Chen<6): {trash_pct:.1f}% | 프리미엄(>=12): {premium_pct:.1f}%")
        print(f"  * 상대 폴드율 (Fold Equity): {opp_fold_pct:.1f}% ({fold_tot}회 폴드 / {called_tot}회 콜)")
        print(f"  * 올인이 콜당했을 때 승률: {win_when_called:.1f}% (콜당했을 때 핸드당 EV: {called_ev_bb:+.1f} BB)")
        print(f"  * 올인 핸드 누적 손익: {total_allin_net_bb:+.1f} BB")
        print(f"  * 전체 세션 순수익: {total_ev_bb100:+.1f} BB/100")
        print(f"  * 올인 포지션 분포: {dict(s['allin_positions'])}")
        print(f"  * 올인 샘플 핸드: {s['allin_cards'][:10]}")
        
    out_json = Path("/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/directive030_1000hands_audit.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장 완료] 결과 JSON: {out_json}")

if __name__ == "__main__":
    run_audit(1000)
