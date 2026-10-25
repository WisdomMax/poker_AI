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

def format_dollar(amount: float, show_sign: bool = False) -> str:
    """가상 달러 포맷팅 (예: +$1,500, -$500, $10,000)"""
    amt_int = int(round(amount))
    sign = "+" if amt_int > 0 and show_sign else ("-" if amt_int < 0 else "")
    abs_str = f"{abs(amt_int):,}"
    if show_sign:
        return f"{sign}${abs_str}"
    return f"-${abs_str}" if amt_int < 0 else f"${abs_str}"

class PokerArenaSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.game = get_5p_fullgame()
        self.model_mgr = ArenaModelManager.get_instance()
        
        self.player_profiles = [
            {"seat": 0, "name": "Hero (나)", "type": "human", "role": "인간 도전자", "avatar": "👤", "char_id": "hero"},
            {"seat": 1, "name": "김프로 (Cycle 4)", "type": "ai", "key": "cycle4", "role": "중후한 챔피언 (121만)", "avatar": "🏆", "char_id": "kim"},
            {"seat": 2, "name": "제니 (Cycle 13)", "type": "ai", "key": "cycle13", "role": "차분한 도전자 (289만)", "avatar": "⚡", "char_id": "jenny"},
            {"seat": 3, "name": "잭 (Cycle 10+MC)", "type": "ai", "key": "cycle10_mc", "role": "자신감 넘치는 전략가 (228만)", "avatar": "🧠", "char_id": "jack"},
            {"seat": 4, "name": "미스터 K (Baseline)", "type": "ai", "key": "baseline", "role": "냉철한 기준선 (10만)", "avatar": "🤖", "char_id": "mr_k"}
        ]
        
        self.initial_stack = 10000.0  # 가상 $10,000 (100 BB)
        self.stacks = [self.initial_stack] * 5
        self.cumulative_returns_chips = [0.0] * 5
        self.cumulative_returns_bb = [0.0] * 5
        self.hands_played = 0
        self.hero_wins = 0
        self.last_hand_profit_dollars = 0.0
        self.last_action_event = None
        self.last_street = 0
        
        # AI 상대별 전적 기록 (Seat 1, 2, 3, 4)
        self.ai_head_to_head = {
            i: {"wins": 0, "losses": 0, "ties": 0} for i in range(1, 5)
        }
        
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
        self.stacks = [self.initial_stack] * 5
        self.cumulative_returns_chips = [0.0] * 5
        self.cumulative_returns_bb = [0.0] * 5
        self.hands_played = 0
        self.hero_wins = 0
        self.last_hand_profit_dollars = 0.0
        self.ai_head_to_head = {
            i: {"wins": 0, "losses": 0, "ties": 0} for i in range(1, 5)
        }
        self.winner_announcement = None

    def start_new_hand(self, seed: int = None):
        self.hands_played += 1
        self.hand_seed = seed if seed is not None else random.randint(100000, 999999)
        self.chance_rng = np.random.RandomState(self.hand_seed)
        self.button_seat = (self.hands_played - 1) % 5
        self.folded_seats = set()
        self.hand_action_log = []
        self.hand_dec_idx = 0
        self.winner_announcement = None
        self.last_action_event = {"type": "new_hand", "hand": self.hands_played}
        self.last_street = 0
        
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
        previous = self.board_cards
        self.board_cards = []
        for line in s.splitlines():
            if 'BoardCards' in line:
                b_str = line.replace('BoardCards', '').strip()
                if b_str:
                    parsed = [b_str[i:i+2] for i in range(0, len(b_str), 2)]
                    # The engine serializes a sorted board. Keep revealed slots stable.
                    self.board_cards = [c for c in previous if c in parsed]
                    self.board_cards.extend(c for c in parsed if c not in self.board_cards)
                break

    def _format_action_desc(self, cp: int, act_id: int, raw_act: int) -> str:
        if act_id == 0:
            return "다이 (폴드)"
        spent = extract_player_spent(self.current_state)
        max_spent = max(spent)
        if act_id == 1:
            diff = max_spent - spent[cp]
            if diff <= 0:
                return "체크 (Check)"
            return f"콜 (${int(diff):,})"
        base_name = ACTION_NAMES.get(act_id, f"Action_{act_id}")
        return f"{base_name} (${int(raw_act):,})"

    def _get_action_key(self, cp: int, act_id: int) -> str:
        if act_id == 0:
            return "fold"
        if act_id == 1:
            spent = extract_player_spent(self.current_state)
            diff = max(spent) - spent[cp]
            return "check" if diff <= 0 else "call"
        if act_id in [2, 3, 4]:
            return "raise"
        if act_id == 5:
            return "allin"
        return "call"

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
        raw_act = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, cp, action_id)
        act_desc = self._format_action_desc(cp, action_id, raw_act)
        act_key = self._get_action_key(cp, action_id)
        
        self.hand_action_log.append(f"[{STREET_NAMES_KO.get(rnd, '')}] Hero (나): {act_desc}")
        
        self.last_action_event = {
            "type": "action",
            "seat": cp,
            "char_id": "hero",
            "name": "Hero (나)",
            "action_id": action_id,
            "action_key": act_key,
            "amount": int(raw_act),
            "desc": act_desc
        }
        
        if action_id == 0:
            self.folded_seats.add(cp)
            
        self.current_state.apply_action(raw_act)
        self.hand_dec_idx += 1
        
        # 챈스 노드는 app.py의 process_game_flow() -> advance_one_chance_street()에서 단계별로 진행
        if self.current_state.is_terminal():
            self._handle_terminal()
        return self.get_client_state()

    def step_ai_single(self):
        if not self.is_hand_active or self.current_state.is_terminal():
            return False, self.get_client_state()
            
        if self.current_state.is_chance_node():
            return False, self.get_client_state()
            
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
        
        raw_act = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, cp, act)
        act_desc = self._format_action_desc(cp, act, raw_act)
        act_key = self._get_action_key(cp, act)
        tag = " [MC수정]" if mc_used else ""
        self.hand_action_log.append(f"[{STREET_NAMES_KO.get(rnd, '')}] {profile['name']}: {act_desc}{tag} ({lat_ms:.1f}ms)")
        
        self.last_action_event = {
            "type": "action",
            "seat": cp,
            "char_id": profile.get("char_id", f"seat{cp}"),
            "name": profile["name"],
            "action_id": act,
            "action_key": act_key,
            "amount": int(raw_act),
            "desc": act_desc
        }
        
        if act == 0:
            self.folded_seats.add(cp)
            
        self.current_state.apply_action(raw_act)
        self.hand_dec_idx += 1
        
        # 챈스 노드는 app.py의 process_game_flow() -> advance_one_chance_street()에서 단계별로 진행
        if self.current_state.is_terminal():
            self._handle_terminal()
            
        return True, self.get_client_state()

    def advance_one_chance_street(self):
        """챈스 노드가 있을 때 다음 1개 스트리트(플랍 3장, 턴 1장, 리버 1장)까지만 전진"""
        if not self.is_hand_active or self.current_state.is_terminal():
            return False, self.get_client_state()
            
        if not self.current_state.is_chance_node():
            return False, self.get_client_state()
            
        self._update_board_cards()
        initial_board_count = len(self.board_cards)
        target_count = 3 if initial_board_count == 0 else (4 if initial_board_count == 3 else 5)
        
        while not self.current_state.is_terminal() and self.current_state.is_chance_node():
            from v48.poker_environment import sample_chance_action
            act = sample_chance_action(self.current_state, self.chance_rng)
            self.current_state.apply_action(act)
            self._update_board_cards()
            if len(self.board_cards) >= target_count:
                break
                
        curr_round_name = STREET_NAMES_KO.get(get_round_from_state(self.current_state), "쇼다운")
        
        self.last_action_event = {
            "type": "street_deal",
            "round_name": curr_round_name,
            "board_cards": list(self.board_cards)
        }
        
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
        if not self.is_hand_active:
            return
        self.is_hand_active = False
        returns_chips = self.current_state.returns()
        returns_bb = [r / 100.0 for r in returns_chips]
        hero_net_chips = returns_chips[0]
        self.last_hand_profit_dollars = float(hero_net_chips)
        
        winners = []
        for i in range(5):
            self.stacks[i] += returns_chips[i]
            self.cumulative_returns_chips[i] += returns_chips[i]
            self.cumulative_returns_bb[i] += returns_bb[i]
            if returns_chips[i] > 0:
                winners.append((i, self.player_profiles[i]["name"], returns_chips[i]))
                
        if returns_chips[0] > 0:
            self.hero_wins += 1
            
        # AI 상대별 전적 집계 (Seat 1~4와 Hero의 상대적 핸드 손익 비교)
        for ai_seat in range(1, 5):
            ai_net = returns_chips[ai_seat]
            if hero_net_chips > ai_net:
                self.ai_head_to_head[ai_seat]["wins"] += 1
            elif hero_net_chips < ai_net:
                self.ai_head_to_head[ai_seat]["losses"] += 1
            else:
                self.ai_head_to_head[ai_seat]["ties"] += 1
            
        # 승자 안내 배너 작성 (가상 달러 $ 단위)
        if winners:
            w_names = ", ".join([f"{w[1]} (+${int(w[2]):,})" for w in winners])
            if hero_net_chips > 0:
                self.winner_announcement = f"🎉 축하합니다! 당신이 승리했습니다! (+${int(hero_net_chips):,})"
            else:
                loss_str = f"이번 판 -${abs(int(hero_net_chips)):,}" if hero_net_chips < 0 else "이번 판 $0"
                self.winner_announcement = f"🏆 승자: {w_names} ({loss_str})"
        else:
            self.winner_announcement = "핸드가 무승부(Chop)로 종료되었습니다."

        self.last_action_event = {
            "type": "terminal",
            "winners": [w[0] for w in winners],
            "winner_names": [w[1] for w in winners],
            "hero_win": hero_net_chips > 0,
            "hero_profit": int(hero_net_chips)
        }

        self.hand_action_log.append("=== 쇼다운 및 최종 정산 ===")
        for i, prof in enumerate(self.player_profiles):
            net_c = int(returns_chips[i])
            sign = "+" if net_c > 0 else ""
            self.hand_action_log.append(
                f"  좌석 {i} ({prof['name']}): {sign}${abs(net_c):,} (현재 보유: ${int(self.stacks[i]):,})"
            )
            
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
        call_cost_chips = 0
        raise_amounts = {}
        raise_amounts_dollars = {}
        
        if curr_player == 0 and not is_term:
            legal_actions = True6ActionAdapter.get_legal_6actions(self.current_state, 0)
            max_spent = max(spent)
            call_cost_chips = int(max_spent - spent[0])
            call_cost_bb = call_cost_chips / 100.0
            
            # 레이즈 금액 산출
            for a_id in [2, 3, 4, 5]:
                if a_id in legal_actions:
                    raw = True6ActionAdapter.map_6action_to_engine_amount(self.current_state, 0, a_id)
                    raise_amounts[str(a_id)] = round(raw / 100.0, 1)
                    raise_amounts_dollars[str(a_id)] = int(raw)
            
        players_view = []
        for i, prof in enumerate(self.player_profiles):
            p_dict = {
                "seat": i,
                "name": prof["name"],
                "role": prof["role"],
                "avatar": prof["avatar"],
                "char_id": prof.get("char_id", f"seat{i}"),
                "stack_bb": round(self.stacks[i] / 100.0, 1),
                "stack_dollars": int(self.stacks[i]),
                "current_bet_bb": round(spent[i] / 100.0, 1),
                "current_bet_dollars": int(spent[i]),
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
        hero_net_dollars = int(round(self.stacks[0] - self.initial_stack))
        
        ai_records_view = [
            {
                "seat": i,
                "name": self.player_profiles[i]["name"],
                "role": self.player_profiles[i]["role"],
                "avatar": self.player_profiles[i]["avatar"],
                "wins": self.ai_head_to_head[i]["wins"],
                "losses": self.ai_head_to_head[i]["losses"],
                "ties": self.ai_head_to_head[i]["ties"]
            }
            for i in range(1, 5)
        ]
        
        return {
            "session_id": self.session_id,
            "hand_number": self.hands_played,
            "is_terminal": is_term,
            "round_name": STREET_NAMES_KO.get(curr_round, "쇼다운 (Showdown)") if not is_term else "쇼다운 (Showdown)",
            "current_player": curr_player,
            "pot_bb": round(pot_chips / 100.0, 1),
            "pot_dollars": int(pot_chips),
            "board_cards": self.board_cards,
            "players": players_view,
            "legal_actions": legal_actions,
            "call_cost_bb": call_cost_bb,
            "call_cost_dollars": call_cost_chips,
            "raise_amounts": raise_amounts,
            "raise_amounts_dollars": raise_amounts_dollars,
            "winner_announcement": self.winner_announcement,
            "last_event": self.last_action_event,
            "event_sequence": self.hand_dec_idx,
            "hero_stats": {
                "hands": self.hands_played,
                "initial_bankroll": int(self.initial_stack),
                "current_balance": int(self.stacks[0]),
                "total_profit_dollars": hero_net_dollars,
                "last_hand_profit_dollars": int(self.last_hand_profit_dollars),
                "win_rate_pct": round(hero_win_pct, 1),
                "ai_records": ai_records_view,
                # 내부 연구용 지표
                "total_profit_bb": round(self.cumulative_returns_bb[0], 1),
                "bb100": round(hero_bb100, 1)
            },
            "recent_logs": self.hand_action_log[-12:]
        }
