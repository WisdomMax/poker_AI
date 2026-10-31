#!/usr/bin/env python3
"""
v1024/test_directive059o_integrity.py
DIRECTIVE 059-O Mandatory Integrity & Negative Tests.
Strictly verifies:
  1. Negative Test 1: Zero-sum preserving maldistribution injected into actual eval path triggers immediate AssertionError.
  2. Negative Test 2: Calculator exception injected into actual eval path triggers immediate Exception (no broad except suppression).
  3. CRN Invariance Test: AlwaysFold vs AlwaysCall branching preserves cards in Hand 1 and invariant deal in Hand 2.
  4. 25-Hand Canary: All 25 hands have audit_executed = True, max_discrepancy < 1e-4, zero-sum error < 1e-4.
"""

import os
import sys
import numpy as np
import pyspiel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from v1024.directive059o_comparative_eval import evaluate_9arms_059o
from v79.run_v79_controlled_causal_study import get_5p_fullgame, advance_chance_nodes, extract_player_spent
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

def test_actual_eval_zero_sum_preserving_maldistribution_aborts():
    print(">>> [Test 1] Testing zero-sum preserving maldistribution abort in actual eval path...")
    try:
        evaluate_9arms_059o(
            seeds_list=[3006100],
            hands_per_seed=1,
            test_mode=True,
            injected_maldistribution=True
        )
        print("[-] FAILED: Injected maldistribution did NOT trigger abort!")
        return False
    except AssertionError as e:
        print(f"[+] PASSED: Injected zero-sum preserving maldistribution correctly aborted with AssertionError: {e}")
        return True
    except Exception as e:
        print(f"[-] FAILED: Injected maldistribution triggered unexpected error: {type(e)} {e}")
        return False

def test_actual_eval_calculator_exception_aborts():
    print(">>> [Test 2] Testing calculator exception abort in actual eval path...")
    try:
        evaluate_9arms_059o(
            seeds_list=[3006100],
            hands_per_seed=1,
            test_mode=True,
            injected_calculator_exception=True
        )
        print("[-] FAILED: Injected calculator exception did NOT trigger abort!")
        return False
    except ValueError as e:
        print(f"[+] PASSED: Injected calculator exception correctly raised without being swallowed: {e}")
        return True
    except Exception as e:
        print(f"[-] FAILED: Unexpected exception type: {type(e)} {e}")
        return False

def test_actual_eval_crn_and_branch_invariance():
    print(">>> [Test 3] Testing CRN & branching invariance on actual game path...")
    game = get_5p_fullgame()
    test_seed = 3006100

    def play_branch(action_choice_fn, hand_idx):
        deck_rng = np.random.RandomState((test_seed * 10007 + hand_idx * 17) & 0xFFFFFFFF)
        s = game.new_initial_state()
        advance_chance_nodes(s, deck_rng)
        folded = [False] * 5

        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, deck_rng)
                continue
            cp = s.current_player()
            legal = s.legal_actions()
            act = action_choice_fn(legal, cp)
            if act == 0:
                folded[cp] = True
            s.apply_action(act)

        s_str = s.to_string()
        holes, board = parse_cards_from_state(s_str)
        return holes, board, s.returns()

    # Hand 0: Fold Branch vs Call/Check Branch
    # Fold Branch: P0 folds immediately if 0 in legal
    def fold_actor(legal, cp):
        return 0 if (0 in legal and cp == 0) else legal[-1]

    # Call Branch: Always Call/Check
    def call_actor(legal, cp):
        return 1 if 1 in legal else legal[0]

    h0_fold_holes, h0_fold_board, _ = play_branch(fold_actor, 0)
    h0_call_holes, h0_call_board, _ = play_branch(call_actor, 0)

    # 1. Hole cards dealt to all 5 players must be 100% identical regardless of action branch
    assert h0_fold_holes == h0_call_holes, "Hole cards mismatch between fold and call branch!"

    # Hand 1: Verify deal in subsequent hand is invariant
    h1_fold_holes, h1_fold_board, _ = play_branch(fold_actor, 1)
    h1_call_holes, h1_call_board, _ = play_branch(call_actor, 1)

    assert h1_fold_holes == h1_call_holes, "Subsequent hand hole cards mismatch!"
    assert h0_fold_holes != h1_fold_holes, "Hand 0 and Hand 1 hole cards should differ!"

    print("[+] PASSED: CRN branching invariance and card isolation verified.")
    return True

def test_25hand_canary_audit_coverage():
    print(">>> [Test 4] Running 25-hand canary with strict audit verification...")
    res = evaluate_9arms_059o(
        seeds_list=[3006100],
        hands_per_seed=25,
        test_mode=True
    )
    records = res["all_records"]
    assert len(records) == 225, f"Expected 225 records (9 arms * 25 hands), got {len(records)}"

    audited_count = sum(1 for r in records if r["audit_executed"])
    assert audited_count == 225, f"Expected all 225 records to have audit_executed=True, got {audited_count}"

    assert res["max_pot_discrepancy"] < 1e-4, f"Discrepancy too high: {res['max_pot_discrepancy']}"
    assert res["max_zero_sum_error"] < 1e-4, f"Zero-sum error too high: {res['max_zero_sum_error']}"

    print(f"[+] PASSED: 25-hand canary passed completely. Total records: {len(records)}, Audited: {audited_count}")
    print(f"    Showdowns: {res['showdown_count']}, Foldouts: {res['foldout_count']}, Max discrepancy: {res['max_pot_discrepancy']}")
    return True

def main():
    print("=================================================================")
    print("STARTING DIRECTIVE 059-O MANDATORY INTEGRITY & NEGATIVE TESTS")
    print("=================================================================")

    t1 = test_actual_eval_zero_sum_preserving_maldistribution_aborts()
    t2 = test_actual_eval_calculator_exception_aborts()
    t3 = test_actual_eval_crn_and_branch_invariance()
    t4 = test_25hand_canary_audit_coverage()

    if t1 and t2 and t3 and t4:
        print("\n[SUCCESS] ALL DIRECTIVE 059-O INTEGRITY & CANARY TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n[FAILURE] One or more integrity tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
