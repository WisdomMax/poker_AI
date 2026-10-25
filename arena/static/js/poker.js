// POKER AI HUMAN ARENA LITE - WEBSOCKET & UI CONTROLLER

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
        this.ws = null;
        this.state = null;
        
        this.initDOMElements();
        this.connectWebSocket();
        this.attachEventListeners();
    }

    getOrCreateSessionId() {
        let sid = localStorage.getItem('arena_session_id');
        if (!sid) {
            sid = 'session_' + Math.random().toString(36).substring(2, 10);
            localStorage.setItem('arena_session_id', sid);
        }
        return sid;
    }

    initDOMElements() {
        this.potDisplay = document.getElementById('pot-display');
        this.streetDisplay = document.getElementById('street-display');
        this.boardCardsContainer = document.getElementById('board-cards');
        
        this.statHands = document.getElementById('stat-hands');
        this.statNet = document.getElementById('stat-net');
        this.statBb100 = document.getElementById('stat-bb100');
        this.statWinrate = document.getElementById('stat-winrate');
        
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
        
        this.feedbackForm = document.getElementById('feedback-form');
        this.fbStatus = document.getElementById('fb-status');
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected!');
            this.addLog('아레나 서버와 실시간 연결되었습니다.');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.onStateUpdate(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket closed, attempting reconnect in 2s...');
            setTimeout(() => this.connectWebSocket(), 2000);
        };
        
        this.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    attachEventListeners() {
        this.btnNewHand.addEventListener('click', () => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'start_hand' }));
            }
        });
        
        this.btnReset.addEventListener('click', () => {
            if (confirm('스택과 세션 통계를 초기화하시겠습니까?')) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'reset_stacks' }));
                }
            }
        });
        
        // 6개 액션 버튼 클릭
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
        
        // 피드백 폼 제출
        this.feedbackForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const cat = document.getElementById('fb-category').value;
            const comment = document.getElementById('fb-comment').value;
            
            try {
                const res = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: this.sessionId,
                        hand_number: this.state ? this.state.hand_number : 0,
                        category: cat,
                        comment: comment
                    })
                });
                if (res.ok) {
                    this.fbStatus.innerText = '피드백이 성공적으로 기록되었습니다. 감사합니다!';
                    document.getElementById('fb-comment').value = '';
                    setTimeout(() => { this.fbStatus.innerText = ''; }, 4000);
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
        this.statHands.innerText = stats.hands || 0;
        const net = stats.total_profit_bb || 0.0;
        this.statNet.innerText = (net >= 0 ? '+' : '') + net.toFixed(1) + ' BB';
        this.statNet.style.color = net >= 0 ? '#10b981' : '#ef4444';
        this.statBb100.innerText = (stats.bb100 >= 0 ? '+' : '') + (stats.bb100 || 0.0).toFixed(1);
        this.statWinrate.innerText = (stats.win_rate_pct || 0.0).toFixed(1) + '%';
        
        // 2. 테이블 중앙 (팟 & 커뮤니티 카드)
        this.potDisplay.innerText = (state.pot_bb || 0.0).toFixed(1) + ' BB';
        this.streetDisplay.innerText = state.round_name || 'Preflop';
        
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
            const dealerBadge = p.is_dealer ? `<span class="dealer-button">D</span>` : '';
            const betBadge = p.current_bet_bb > 0 ? `<div class="player-bet-badge">Bet: ${p.current_bet_bb.toFixed(1)} BB</div>` : '';
            
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
        
        // 4. 인간(Hero) 액션 버튼 제어
        const isHumanTurn = (state.current_player === 0 && !state.is_terminal);
        const legal = state.legal_actions || [];
        
        Object.entries(this.buttons).forEach(([actId, btn]) => {
            const aNum = parseInt(actId);
            const isLegal = isHumanTurn && legal.includes(aNum);
            btn.disabled = !isLegal;
            
            // 체크 vs 콜 텍스트 업데이트
            if (aNum === 1) {
                if (state.call_cost_bb > 0) {
                    btn.innerText = `Call (${state.call_cost_bb.toFixed(1)} BB)`;
                } else {
                    btn.innerText = `Check (체크)`;
                }
            }
        });
        
        // 새 핸드 버튼 제어
        this.btnNewHand.disabled = !state.is_terminal;
        if (state.is_terminal) {
            this.btnNewHand.style.animation = 'pulse 1.5s infinite';
        } else {
            this.btnNewHand.style.animation = 'none';
        }
        
        // 5. 최근 로그 렌더링
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
