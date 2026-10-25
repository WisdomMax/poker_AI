"""
POKER AI HUMAN ARENA: Enhanced Game Engine (DIRECTIVE 020)
==========================================================
Upgraded engine providing:
- Real-time exact raise amounts (chips & BB) for actions 2, 3, 4, 5
- Showdown winner announcement with profit details
- Strict card security & zero leakage
- Real-time session ledger audit
"""

import os
import sys
import re
import time
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from arena.models_manager import ArenaModelManager

ACTION_NAMES = {
    0: "폴드 (Fold)",
    1: "체크/콜 (Check/Call)",
    2: "50% 팟 레이즈",
    3: "100% 팟 레이즈",
    4: "200% 팟 레이즈",
    5: "올인 (All-in)"
}

STREET_NAMES_KO = {
    0: "프리플랍 (Preflop)",
    1: "플랍 (Flop)",
    2: "턴 (Turn)",
    3: "리버 (River)",
    4: "쇼다운 (Showdown)"
}

class PokerArenaSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.game = get_5p_fullgame()
        self.model_mgr = ArenaModelManager.get_instance()
        
        self.player_profiles = [
            {"seat": 0, "name": "Hero (나)", "type": "human", "role": "인간 도전자", "avatar": "👤"},
            {"seat": 1, "name": "AI Cycle 4", "type": "ai", "key": "cycle4", "role": "기본 주력 챔피언 (121만)", "avatar": "🏆"},
            {"seat": 2, "name": "AI Cycle 13", "type": "ai", "key": "cycle13", "role": "신규 풀 1위 도전자 (289만)", "avatar": "⚡"},
            {"seat": 3, "name": "AI Cycle 10+MC", "type": "ai", "key": "cycle10_mc", "role": "적응형 하이브리드 (228만)", "avatar": "🧠"},
            {"seat": 4, "name": "AI Baseline", "type": "ai", "key": "baseline", "role": "V10.22 사전학습 기준선 (10만)", "avatar": "🤖"}
        ]
        
        self.stacks = [10000.0] * 5  # 100 BB each
        self.cumulative_returns_bb = [0.0] * 5
        self.hands_played = 0
        self.hero_wins = 0
        
        self.current_state = None
        self.chance_rng = None
        self.hand_seed = 0
        self.button_seat = 0
        self.folded_seats = set()
        self.all_private_cards = {}
        self.board_cards = []
        self.hand_action_log = []
        self.hand_dec_idx = 0
        self.is_hand_active = False
        self.winner_announcement = None

    def reset_stacks(self):
        self.stacks = [10000.0] * 5
        self.cumulative_returns_bb = [0.0] * 5
        self.hands_played = 0
        self.hero_wins = 0
        self.winner_announcement = None

    def start_new_hand(self):
        self.hands_played += 1
        self.hand_seed = random.randint(100000, 999999)
        self.chance_rng = np.random.RandomState(self.hand_seed)
        self.button_seat = (self.hands_played - 1) % 5
        self.folded_seats = set()
        self.hand_action_log = []
        self.hand_dec_idx = 0
        self.winner_announcement = None
        
        self.current_state = self.game.new_initial_state()
        advance_chance_nodes(self.current_state, self.chance_rng)
        
        self._parse_private_cards()
        self._update_board_cards()
        self.is_hand_active = True
        
        self.hand_action_log.append(f"--- #{self.hands_played}번째 핸드 시작 (딜러 버튼: 좌석 {self.button_seat}) ---")
        return self.get_client_state()

    def _parse_private_cards(self):
        s = self.current_state.to_string()
        self.all_private_cards = {}
        for line in s.splitlines():
            m = re.match(r'P(\d+) Cards:\s*([2-9TJQKA][shdc][2-9TJQKA][shdc])', line.strip())
            if m:
                p_idx = int(m.group(1))
                c_str = m.group(2)
                self.all_private_cards[p_idx] = [c_str[:2], c_str[2:]]

    def _update_board_cards(self):
        s = self.current_state.to_string()
        self.board_cards = []
        for line in s.splitlines():
            if 'BoardCards' in line:
                b_str = line.replace('BoardCards', '').strip()
                if b_str:
                    self.board_cards = [b_str[i:i+2] for i in range(0, len(b_str), 2)]
                break

    def step_human(self, action_id: int):
        if not self.is_hand_active or self.current_state.is_terminal():
            return self.get_client_state()
            
        cp = self.current_state.current_player()
        if cp != 0:
            raise ValueError(f"인간 플레이어의 턴이 아닙니다! (현재 플레이어: {cp})")
            
        legal_6 = True6ActionAdapter.get_legal_6actions(self.current_state, cp)
        if action_id not in legal_6:
            action_id = 1
            
        rnd = get_round_from_state(self.current_state)
        act_name = ACTION_NAMES.get(action_id, f"Action_{action_id}")
        self.hand_action_log.append(f"[{STREET_NAMES_KO.get(rnd, '')}] Hero (나): {act_name}")
        
        if action_id == 0:
            self.folded_seats.add(cp)
            
        raw_act = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, cp, action_id)
        self.current_state.apply_action(raw_act)
        self.hand_dec_idx += 1
        
        self._advance_chance_if_needed()
        return self.get_client_state()

    def step_ai_single(self):
        if not self.is_hand_active or self.current_state.is_terminal():
            return False, self.get_client_state()
            
        if self.current_state.is_chance_node():
            self._advance_chance_if_needed()
            return True, self.get_client_state()
            
        cp = self.current_state.current_player()
        if cp == 0:
            return False, self.get_client_state()
            
        profile = self.player_profiles[cp]
        ai_key = profile["key"]
        legal_6 = True6ActionAdapter.get_legal_6actions(self.current_state, cp)
        rnd = get_round_from_state(self.current_state)
        
        act, lat_ms, mc_used = self.model_mgr.select_action(
            ai_key, self.current_state, cp, legal_6,
            self.folded_seats, self.hand_seed, self.hand_dec_idx, rnd
        )
        
        act_name = ACTION_NAMES.get(act, f"Action_{act}")
        tag = " [MC수정]" if mc_used else ""
        self.hand_action_log.append(f"[{STREET_NAMES_KO.get(rnd, '')}] {profile['name']}: {act_name}{tag} ({lat_ms:.1f}ms)")
        
        if act == 0:
            self.folded_seats.add(cp)
            
        raw_act = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, cp, act)
        self.current_state.apply_action(raw_act)
        self.hand_dec_idx += 1
        
        self._advance_chance_if_needed()
        
        if self.current_state.is_terminal():
            self._handle_terminal()
            
        return True, self.get_client_state()

    def _advance_chance_if_needed(self):
        while not self.current_state.is_terminal() and self.current_state.is_chance_node():
            advance_chance_nodes(self.current_state, self.chance_rng)
        self._update_board_cards()
        if self.current_state.is_terminal():
            self._handle_terminal()

    def _handle_terminal(self):
        self.is_hand_active = False
        returns_chips = self.current_state.returns()
        returns_bb = [r / 100.0 for r in returns_chips]
        
        winners = []
        for i in range(5):
            self.stacks[i] += returns_chips[i]
            self.cumulative_returns_bb[i] += returns_bb[i]
            if returns_bb[i] > 0:
                winners.append((i, self.player_profiles[i]["name"], returns_bb[i]))
                
        if returns_bb[0] > 0:
            self.hero_wins += 1
            
        # 승자 안내 배너 작성
        if winners:
            w_names = ", ".join([f"{w[1]} (+{w[2]:.1f} BB)" for w in winners])
            if returns_bb[0] > 0:
                self.winner_announcement = f"🎉 축하합니다! 당신이 승리했습니다! (+{returns_bb[0]:.1f} BB)"
            else:
                self.winner_announcement = f"🏆 핸드 종료! 승자: {w_names}"
        else:
            self.winner_announcement = "핸드가 무승부(Chop)로 종료되었습니다."

        self.hand_action_log.append("=== 쇼다운 및 최종 정산 ===")
        for i, prof in enumerate(self.player_profiles):
            net = returns_bb[i]
            sign = "+" if net > 0 else ""
            self.hand_action_log.append(f"  좌석 {i} ({prof['name']}): {sign}{net:.1f} BB (현재 스택: {self.stacks[i]/100.0:.1f} BB)")
            
        self._save_hand_log(returns_chips, returns_bb)

    def _save_hand_log(self, returns_chips, returns_bb):
        log_dir = "arena/data"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "arena_hands.jsonl")
        
        record = {
            "session_id": self.session_id,
            "hand_number": self.hands_played,
            "seed": self.hand_seed,
            "button_seat": self.button_seat,
            "board_cards": self.board_cards,
            "private_cards": self.all_private_cards,
            "returns_chips": [float(r) for r in returns_chips],
            "returns_bb": [float(r) for r in returns_bb],
            "action_log": self.hand_action_log,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_client_state(self):
        is_term = self.current_state.is_terminal() if self.current_state else True
        spent = extract_player_spent(self.current_state) if self.current_state else [0]*5
        pot_chips = sum(spent)
        curr_round = get_round_from_state(self.current_state) if self.current_state else 0
        curr_player = self.current_state.current_player() if (self.current_state and not is_term) else -1
        
        legal_actions = []
        call_cost_bb = 0.0
        raise_amounts = {}
        
        if curr_player == 0 and not is_term:
            legal_actions = True6ActionAdapter.get_legal_6actions(self.current_state, 0)
            max_spent = max(spent)
            call_cost_chips = max_spent - spent[0]
            call_cost_bb = call_cost_chips / 100.0
            
            # 레이즈 금액 산출
            for a_id in [2, 3, 4, 5]:
                if a_id in legal_actions:
                    raw = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, 0, a_id)
                    raise_amounts[str(a_id)] = round(raw / 100.0, 1)
            
        players_view = []
        for i, prof in enumerate(self.player_profiles):
            p_dict = {
                "seat": i,
                "name": prof["name"],
                "role": prof["role"],
                "avatar": prof["avatar"],
                "stack_bb": round(self.stacks[i] / 100.0, 1),
                "current_bet_bb": round(spent[i] / 100.0, 1),
                "is_folded": i in self.folded_seats,
                "is_active": (i == curr_player),
                "is_dealer": (i == self.button_seat),
                "cards": []
            }
            
            if i == 0:
                p_dict["cards"] = self.all_private_cards.get(0, [])
            elif is_term and i not in self.folded_seats:
                p_dict["cards"] = self.all_private_cards.get(i, ["??", "??"])
            else:
                p_dict["cards"] = ["??", "??"] if i not in self.folded_seats else []
                
            players_view.append(p_dict)
            
        hero_bb100 = (self.cumulative_returns_bb[0] / self.hands_played * 100.0) if self.hands_played > 0 else 0.0
        hero_win_pct = (self.hero_wins / self.hands_played * 100.0) if self.hands_played > 0 else 0.0
        
        return {
            "session_id": self.session_id,
            "hand_number": self.hands_played,
            "is_terminal": is_term,
            "round_name": STREET_NAMES_KO.get(curr_round, "쇼다운 (Showdown)") if not is_term else "쇼다운 (Showdown)",
            "current_player": curr_player,
            "pot_bb": round(pot_chips / 100.0, 1),
            "board_cards": self.board_cards,
            "players": players_view,
            "legal_actions": legal_actions,
            "call_cost_bb": call_cost_bb,
            "raise_amounts": raise_amounts,
            "winner_announcement": self.winner_announcement,
            "hero_stats": {
                "hands": self.hands_played,
                "total_profit_bb": round(self.cumulative_returns_bb[0], 1),
                "bb100": round(hero_bb100, 1),
                "win_rate_pct": round(hero_win_pct, 1)
            },
            "recent_logs": self.hand_action_log[-12:]
        }
