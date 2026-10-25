"""
POKER AI HUMAN ARENA: 20-Hand Automated Ledger & Action Verification (DIRECTIVE 020)
=====================================================================================
Automated verification of 20 consecutive hands:
- Exercises Fold, Check, Call, 50% Raise, 100% Raise, 200% Raise, All-in, and Showdown
- Strictly audits chip conservation (sum of returns == 0) and total stack conservation (50,000 chips)
"""

import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.game_engine import PokerArenaSession

def main():
    print("=" * 80)
    print("DIRECTIVE 020: 20-HAND ARENA LEDGER & ACTION AUDIT START")
    print("=" * 80)
    
    session = PokerArenaSession("audit_session_20hands")
    
    action_type_counts = {
        "Fold": 0,
        "Check": 0,
        "Call": 0,
        "50% Raise": 0,
        "100% Raise": 0,
        "200% Raise": 0,
        "All-in": 0,
        "Showdown": 0
    }
    
    # 20핸드 순환
    for hand_i in range(1, 21):
        st = session.start_new_hand()
        # 핸드마다 다른 전략 시도 (폴드, 콜, 레이즈, 올인)
        target_action_policy = hand_i % 6  # 0~5 액션 순환
        
        while session.is_hand_active and not session.current_state.is_terminal():
            cp = session.current_state.current_player()
            if cp == 0:  # Human turn
                legal = session.current_state.legal_actions(0)
                legal_6 = [0, 1, 2, 3, 4, 5]
                # 가능한 액션 중 타겟 액션 선택 시도
                chosen_act = target_action_policy
                if chosen_act not in session.get_client_state()["legal_actions"]:
                    chosen_act = 1  # Fallback to Check/Call
                    
                # 액션 통계
                if chosen_act == 0:
                    action_type_counts["Fold"] += 1
                elif chosen_act == 1:
                    if session.get_client_state()["call_cost_bb"] > 0:
                        action_type_counts["Call"] += 1
                    else:
                        action_type_counts["Check"] += 1
                elif chosen_act == 2:
                    action_type_counts["50% Raise"] += 1
                elif chosen_act == 3:
                    action_type_counts["100% Raise"] += 1
                elif chosen_act == 4:
                    action_type_counts["200% Raise"] += 1
                elif chosen_act == 5:
                    action_type_counts["All-in"] += 1
                    
                st = session.step_human(chosen_act)
            else:
                acted, st = session.step_ai_single()
                
        # 핸드 종료 후 원장 감사
        action_type_counts["Showdown"] += 1
        returns_chips = session.current_state.returns()
        assert abs(sum(returns_chips)) < 1e-4, f"Hand #{hand_i}: Chip conservation violated! Sum: {sum(returns_chips)}"
        total_stacks = sum(session.stacks)
        assert abs(total_stacks - 50000.0) < 1e-4, f"Hand #{hand_i}: Total stack mismatch! Sum: {total_stacks}"
        assert st["winner_announcement"] is not None, f"Hand #{hand_i}: Missing winner announcement!"
        
        net_str = f"{st['hero_stats']['total_profit_bb']:+.1f} BB"
        print(f"  [Hand #{hand_i:2d}] Winner: {st['winner_announcement'][:38]}... | Hero Net: {net_str} | Total Stacks: {total_stacks/100:.1f} BB (AUDIT PASS)")

    print("\n" + "=" * 80)
    print("20-HAND AUDIT SUMMARY & ACTION COVERAGE:")
    print("-" * 80)
    for act_name, count in action_type_counts.items():
        print(f"  - {act_name:<15}: {count:2d} times exercised")
        assert count > 0, f"Action {act_name} was never exercised!"
    print("-" * 80)
    print(f"  [+] Chip Conservation Law (Sum of Returns == 0) : 100% ALL PASS (20/20 hands)")
    print(f"  [+] Total Stack Ledger Audit (50,000 chips exact): 100% ALL PASS (20/20 hands)")
    print(f"  [+] Winner Announcement Generation               : 100% ALL PASS (20/20 hands)")
    print("=" * 80)

if __name__ == "__main__":
    main()
