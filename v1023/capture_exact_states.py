"""
실제 대국 재생을 통해 Seed 180158 및 Seed 180408의
정확한 Flop 결정 시점의 정보집합(Hero 카드, 보드 카드, 활성 상대 수, combos, probs, pot_odds)을
추출하고, 기존 에쿼티(19.8333%, 44.8333%)가 비트 일치로 재현되는지 확인하는 스크립트.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v75.run_v75_true_6action_and_fold_recovery import (
    get_5p_fullgame, extract_player_spent, True6ActionAdapter
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v79.run_v79_controlled_causal_study import normalize_inf_state_v79
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA, estimate_equity_det,
    RobustBayesianRangeModel, eval_7cards
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

def advance_chance_nodes_persistent(state, rng):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = rng.choice(actions, p=probs)
        state.apply_action(action)

def capture_exact_decision_state(seed_h, target_street="Flop"):
    seat = seed_h % 5
    game = get_5p_fullgame()
    chance_rng = np.random.RandomState(seed_h)
    st = game.new_initial_state()
    advance_chance_nodes_persistent(st, chance_rng)
    
    # Load PPO R2 model
    m_r2 = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    m_r2.load_state_dict(torch.load("v1022/models/StrictAntiLimper_R2_100K.pt", map_location="cpu", weights_only=False))
    m_r2.eval()
    
    # Opponents
    opp_seeds = [(seed_h * 1000 + i * 97) & 0xFFFFFF for i in range(5)]
    from v1023.v1023_13_differential_forensics import RealisticAggressiveBot
    opps = [RealisticAggressiveBot(opp_seeds[i]) for i in range(5)]
    
    bp_agent = DeterministicTrueNFSP_Agent()
    
    f_set = set()
    dec = 0
    captured = None
    
    while not st.is_terminal():
        if st.is_chance_node():
            advance_chance_nodes_persistent(st, chance_rng)
            continue
        cp = st.current_player()
        l6 = True6ActionAdapter.get_legal_6actions(st, cp)
        rnd = get_round_from_state(st)
        street_name = ["Preflop", "Flop", "Turn", "River"][min(rnd, 3)]
        
        if cp == seat:
            sp = extract_player_spent(st)
            tc = max(sp) - sp[cp]
            pot = sum(sp)
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            
            if rnd == 0:
                # Preflop: PPO R2
                norm_inf = normalize_inf_state_v79(inf)
                t_inf = torch.tensor(norm_inf, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = m_r2.forward(t_inf)
                    mask = torch.ones(6, dtype=torch.bool)
                    for a in range(6):
                        if a not in l6:
                            mask[a] = False
                    masked_logits = logits.clone()
                    masked_logits[0, ~mask] = -1e9
                    act = int(torch.argmax(masked_logits[0]).item())
            else:
                # Postflop: Model A
                if tc > 0 and 0 in l6:
                    hc = list(np.where(inf[5:57] == 1.0)[0])
                    bc = list(np.where(inf[57:109] == 1.0)[0])
                    ao = sum(1 for p in range(5) if p != cp and p not in f_set)
                    combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                        hc, bc, mode='temperature', temperature=2.0,
                        opp_aggression_level=1.5, is_raised=(tc >= 100))
                    po = tc / float(pot + tc)
                    eq_rng = np.random.RandomState((seed_h * 10007 + cp * 101 + dec * 17) & 0xFFFFFFFF)
                    orig_eq = estimate_equity_det(hc, bc, combos, probs, ao, 300, eq_rng)
                    
                    if street_name == target_street and captured is None:
                        captured = {
                            "seed": seed_h,
                            "seat": cp,
                            "dec_idx": dec,
                            "street": street_name,
                            "hero_cards": hc,
                            "board_cards": bc,
                            "tc": tc,
                            "pot": pot,
                            "pot_odds": po,
                            "active_opps": ao,
                            "original_equity_300": orig_eq,
                            "combos_count": len(combos),
                            "combos": [list(c) for c in combos],
                            "probs": probs.tolist(),
                            "eq_rng_seed": (seed_h * 10007 + cp * 101 + dec * 17) & 0xFFFFFFFF
                        }
                    
                    if orig_eq > po + 0.05:
                        r = [a for a in l6 if a in [2, 3]]
                        act = r[0] if (r and orig_eq > 0.65) else (1 if 1 in l6 else l6[0])
                    elif orig_eq < po - 0.05:
                        act = 0
                    else:
                        act = 1 if 1 in l6 else l6[0]
                else:
                    bp_rng = np.random.RandomState((seed_h * 20011 + cp * 307 + dec * 31) & 0xFFFFFFFF)
                    act = bp_agent.select_action_deterministic(inf, l6, bp_rng)
            
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            dec += 1
        else:
            opp_idx = cp if cp < seat else cp - 1
            bot = opps[opp_idx]
            act = bot.select_action(st, cp, l6)
            raw_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            
        if act == 0:
            f_set.add(cp)
        st.apply_action(raw_act)
        
    return captured

if __name__ == "__main__":
    c158 = capture_exact_decision_state(180158, "Flop")
    print("Seed 180158 Flop State:")
    print(f"  Hero Cards: {c158['hero_cards']}, Board: {c158['board_cards']}")
    print(f"  TC: {c158['tc']}, Pot: {c158['pot']}, Pot Odds: {c158['pot_odds']:.4%}")
    print(f"  Active Opps: {c158['active_opps']}")
    print(f"  Original Equity (300 sims): {c158['original_equity_300']:.6f} ({c158['original_equity_300']:.4%})")
    
    c408 = capture_exact_decision_state(180408, "Flop")
    print("\nSeed 180408 Flop State:")
    print(f"  Hero Cards: {c408['hero_cards']}, Board: {c408['board_cards']}")
    print(f"  TC: {c408['tc']}, Pot: {c408['pot']}, Pot Odds: {c408['pot_odds']:.4%}")
    print(f"  Active Opps: {c408['active_opps']}")
    print(f"  Original Equity (300 sims): {c408['original_equity_300']:.6f} ({c408['original_equity_300']:.4%})")
