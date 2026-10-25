// POKER AI HUMAN ARENA - CLIENT CONTROLLER WITH PIN AUTH (DIRECTIVE 022)

const SUIT_SYMBOLS = {
    's': '♠',
    'h': '♥',
    'd': '♦',
    'c': '♣'
};

const SUIT_COLORS = {
    's': 'black',
    'h': 'red',
    'd': 'red',
    'c': 'black'
};

class PokerArenaClient {
    constructor() {
        this.sessionId = this.getOrCreateSessionId();
        this.pin = this.getInitialPin();
        this.ws = null;
        this.state = null;
        
        this.initDOMElements();
        this.attachEventListeners();
        
        if (this.pin) {
            this.verifyAndConnect(this.pin);
        } else {
            this.showPinModal();
        }
    }

    getOrCreateSessionId() {
        let sid = localStorage.getItem('arena_session_id');
        if (!sid) {
            sid = 'user_' + Math.random().toString(36).substring(2, 10);
            localStorage.setItem('arena_session_id', sid);
        }
        return sid;
    }

    getInitialPin() {
        // 1. URL 쿼리 파라미터 우선 (?pin=7777)
        const params = new URLSearchParams(window.location.search);
        const urlPin = params.get('pin');
        if (urlPin) {
            sessionStorage.setItem('arena_pin', urlPin);
            return urlPin;
        }
        // 2. 세션스토리지
        return sessionStorage.getItem('arena_pin') || '';
    }

    initDOMElements() {
        this.potDisplay = document.getElementById('pot-display');
        this.streetDisplay = document.getElementById('street-display');
        this.boardCardsContainer = document.getElementById('board-cards');
        
        this.statHands = document.getElementById('stat-hands');
        this.statNet = document.getElementById('stat-net');
        this.statBb100 = document.getElementById('stat-bb100');
        this.statWinrate = document.getElementById('stat-winrate');
        
        this.winnerBanner = document.getElementById('winner-banner');
        this.winnerText = document.getElementById('winner-text');
        this.turnBanner = document.getElementById('turn-banner');
        
        this.pinModal = document.getElementById('pin-modal');
        this.pinForm = document.getElementById('pin-form');
        this.pinInput = document.getElementById('pin-input');
        this.pinError = document.getElementById('pin-error');
        
        this.actionLog = document.getElementById('action-log');
        this.btnNewHand = document.getElementById('btn-new-hand');
        this.btnReset = document.getElementById('btn-reset');
        
        this.buttons = {
            0: document.getElementById('btn-fold'),
            1: document.getElementById('btn-call'),
            2: document.getElementById('btn-raise-50'),
            3: document.getElementById('btn-raise-100'),
            4: document.getElementById('btn-raise-200'),
            5: document.getElementById('btn-allin')
        };
        
        this.lblCall = document.getElementById('lbl-call');
        this.lblRaise50 = document.getElementById('lbl-raise-50');
        this.lblRaise100 = document.getElementById('lbl-raise-100');
        this.lblRaise200 = document.getElementById('lbl-raise-200');
        this.lblAllin = document.getElementById('lbl-allin');
        
        this.feedbackForm = document.getElementById('feedback-form');
        this.fbStatus = document.getElementById('fb-status');
        
        this.btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
        this.sidebarContent = document.getElementById('sidebar-content');
        this.sidebarArrow = document.getElementById('sidebar-arrow');
    }

    showPinModal() {
        if (this.pinModal) {
            this.pinModal.classList.remove('hidden');
            this.pinInput.focus();
        }
    }

    hidePinModal() {
        if (this.pinModal) {
            this.pinModal.classList.add('hidden');
        }
    }

    async verifyAndConnect(pin) {
        try {
            const res = await fetch('/api/verify-pin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pin: pin })
            });
            const data = await res.json();
            if (data.valid) {
                this.pin = pin;
                sessionStorage.setItem('arena_pin', pin);
                this.hidePinModal();
                this.connectWebSocket();
            } else {
                this.showPinModal();
                if (this.pinError) this.pinError.innerText = data.message || '입장 번호가 올바르지 않습니다.';
            }
        } catch (e) {
            console.error('PIN verification error:', e);
            // 오프라인/로컬 fallback
            this.connectWebSocket();
        }
    }

    connectWebSocket() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}?pin=${encodeURIComponent(this.pin)}`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected!');
            this.addLog('아레나 게임 서버와 실시간으로 연결되었습니다.');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.onStateUpdate(data);
        };
        
        this.ws.onclose = (e) => {
            console.log(`WebSocket closed (code: ${e.code})`);
            if (e.code === 1008) {
                // PIN 오류로 거절됨
                sessionStorage.removeItem('arena_pin');
                this.pin = '';
                this.showPinModal();
                if (this.pinError) this.pinError.innerText = '인증에 실패했습니다. 올바른 입장 번호를 입력하세요.';
            } else {
                setTimeout(() => {
                    if (this.pin) this.connectWebSocket();
                }, 2500);
            }
        };
        
        this.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    attachEventListeners() {
        // PIN 폼 제출
        if (this.pinForm) {
            this.pinForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const inputVal = this.pinInput.value.trim();
                await this.verifyAndConnect(inputVal);
            });
        }
        
        this.btnNewHand.addEventListener('click', () => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'start_hand' }));
            }
        });
        
        this.btnReset.addEventListener('click', () => {
            if (confirm('스택과 세션 통계를 초기화하시겠습니까? (초기 100BB로 리셋)')) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'reset_stacks' }));
                }
            }
        });
        
        Object.entries(this.buttons).forEach(([actId, btn]) => {
            btn.addEventListener('click', () => {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'human_action',
                        action_id: parseInt(actId)
                    }));
                }
            });
        });
        
        if (this.btnToggleSidebar) {
            this.btnToggleSidebar.addEventListener('click', () => {
                this.sidebarContent.classList.toggle('collapsed');
                const isCollapsed = this.sidebarContent.classList.contains('collapsed');
                this.sidebarArrow.innerText = isCollapsed ? '▲' : '▼';
            });
        }
        
        this.feedbackForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const comment = document.getElementById('fb-comment').value;
            
            try {
                const res = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: this.sessionId,
                        pin: this.pin,
                        hand_number: this.state ? this.state.hand_number : 0,
                        category: "User_Feedback",
                        comment: comment
                    })
                });
                if (res.ok) {
                    this.fbStatus.innerText = '소중한 의견이 전송되었습니다. 감사합니다!';
                    document.getElementById('fb-comment').value = '';
                    setTimeout(() => { this.fbStatus.innerText = ''; }, 4000);
                } else {
                    this.fbStatus.innerText = '전송 실패 (인증 오류)';
                }
            } catch (err) {
                this.fbStatus.innerText = '전송 실패: ' + err.message;
            }
        });
    }

    renderCard(cardStr) {
        if (!cardStr || cardStr === '??') {
            return `<div class="card hidden"></div>`;
        }
        const rank = cardStr[0];
        const suitChar = cardStr[1];
        const suitSymbol = SUIT_SYMBOLS[suitChar] || suitChar;
        const colorClass = SUIT_COLORS[suitChar] || 'black';
        
        return `
            <div class="card ${colorClass}">
                <div class="card-rank">${rank}</div>
                <div class="card-suit">${suitSymbol}</div>
            </div>
        `;
    }

    onStateUpdate(state) {
        this.state = state;
        
        // 1. 헤더 통계
        const stats = state.hero_stats || {};
        this.statHands.innerText = (stats.hands || 0) + '판';
        const net = stats.total_profit_bb || 0.0;
        this.statNet.innerText = (net >= 0 ? '+' : '') + net.toFixed(1) + ' BB';
        this.statNet.style.color = net >= 0 ? '#10b981' : '#ef4444';
        this.statBb100.innerText = (stats.bb100 >= 0 ? '+' : '') + (stats.bb100 || 0.0).toFixed(1);
        this.statWinrate.innerText = (stats.win_rate_pct || 0.0).toFixed(1) + '%';
        
        // 2. 테이블 중앙 (팟 & 스트리트 & 커뮤니티 카드)
        this.potDisplay.innerText = (state.pot_bb || 0.0).toFixed(1) + ' BB';
        this.streetDisplay.innerText = state.round_name || '프리플랍 (Preflop)';
        
        const boardHtml = (state.board_cards || []).map(c => this.renderCard(c)).join('');
        this.boardCardsContainer.innerHTML = boardHtml;
        
        // 3. 5인 좌석 렌더링
        (state.players || []).forEach(p => {
            const seatEl = document.getElementById(`seat-${p.seat}`);
            if (!seatEl) return;
            
            seatEl.className = 'player-seat seat-' + p.seat;
            if (p.is_active) seatEl.classList.add('active');
            if (p.is_folded) seatEl.classList.add('folded');
            
            const cardsHtml = (p.cards || []).map(c => this.renderCard(c)).join('');
            const dealerBadge = p.is_dealer ? `<span class="dealer-button" title="딜러 버튼">D</span>` : '';
            const betBadge = p.current_bet_bb > 0 ? `<div class="player-bet-badge">베팅: ${p.current_bet_bb.toFixed(1)} BB</div>` : '';
            
            seatEl.innerHTML = `
                <div class="player-header">
                    <span>${p.avatar} ${p.name}</span>
                    ${dealerBadge}
                </div>
                <div class="player-stack">${p.stack_bb.toFixed(1)} BB</div>
                ${betBadge}
                <div class="player-cards">${cardsHtml}</div>
            `;
        });
        
        // 4. 승자 배너 & 턴 배너
        if (state.winner_announcement && state.is_terminal) {
            this.winnerBanner.classList.remove('hidden');
            this.winnerText.innerText = state.winner_announcement;
        } else {
            this.winnerBanner.classList.add('hidden');
        }
        
        const isHumanTurn = (state.current_player === 0 && !state.is_terminal);
        if (isHumanTurn) {
            this.turnBanner.classList.remove('hidden');
        } else {
            this.turnBanner.classList.add('hidden');
        }
        
        // 5. 인간(Hero) 액션 버튼 제어 및 금액 업데이트
        const legal = state.legal_actions || [];
        const raises = state.raise_amounts || {};
        
        Object.entries(this.buttons).forEach(([actId, btn]) => {
            const aNum = parseInt(actId);
            const isLegal = isHumanTurn && legal.includes(aNum);
            btn.disabled = !isLegal;
            
            if (aNum === 1) {
                if (state.call_cost_bb > 0) {
                    this.lblCall.innerText = `콜 (Call ${state.call_cost_bb.toFixed(1)} BB)`;
                } else {
                    this.lblCall.innerText = `체크 (Check)`;
                }
            }
            if (aNum === 2) {
                const amt = raises['2'] ? ` (${raises['2']} BB)` : '';
                this.lblRaise50.innerText = `50% 팟 레이즈${amt}`;
            }
            if (aNum === 3) {
                const amt = raises['3'] ? ` (${raises['3']} BB)` : '';
                this.lblRaise100.innerText = `100% 팟 레이즈${amt}`;
            }
            if (aNum === 4) {
                const amt = raises['4'] ? ` (${raises['4']} BB)` : '';
                this.lblRaise200.innerText = `200% 팟 레이즈${amt}`;
            }
            if (aNum === 5) {
                const amt = raises['5'] ? ` (${raises['5']} BB)` : '';
                this.lblAllin.innerText = `올인 (All-In${amt})`;
            }
        });
        
        // 새 핸드 버튼
        this.btnNewHand.disabled = !state.is_terminal;
        
        // 6. 최근 로그 렌더링
        if (state.recent_logs && state.recent_logs.length > 0) {
            this.actionLog.innerHTML = state.recent_logs.map(line => `<div class="log-entry">${line}</div>`).join('');
            this.actionLog.scrollTop = this.actionLog.scrollHeight;
        }
    }

    addLog(msg) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerText = msg;
        this.actionLog.appendChild(entry);
        this.actionLog.scrollTop = this.actionLog.scrollHeight;
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.pokerClient = new PokerArenaClient();
});
