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
        this.statBalance = document.getElementById('stat-balance');
        this.statNet = document.getElementById('stat-net');
        this.statLastHand = document.getElementById('stat-last-hand');
        this.statWinrate = document.getElementById('stat-winrate');
        this.h2hRows = document.getElementById('h2h-rows');
        
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
        
        // 오디오 & UX 컨트롤러 (DIRECTIVE 028)
        this.btnAudioMute = document.getElementById('btn-audio-mute');
        this.audioVolume = document.getElementById('audio-volume');
        this.btnSpeedToggle = document.getElementById('btn-speed-toggle');
        this.btnGuideToggle = document.getElementById('btn-guide-toggle');
        
        this.streetHelpTip = document.getElementById('street-help-tip');
        this.commStageText = document.getElementById('comm-stage-text');
        
        this.hintCall = document.getElementById('hint-call');
        this.hintRaise50 = document.getElementById('hint-raise-50');
        this.hintRaise100 = document.getElementById('hint-raise-100');
        this.hintRaise200 = document.getElementById('hint-raise-200');
        this.hintAllin = document.getElementById('hint-allin');
        
        this.gameSpeed = 1.0;
        this.guideEnabled = true;
        this.renderedBoardCards = [];
        
        if (this.btnAudioMute && window.arenaAudio) {
            this.btnAudioMute.innerText = window.arenaAudio.isMuted() ? '🔇' : '🔊';
        }
        if (this.audioVolume && window.arenaAudio) {
            this.audioVolume.value = window.arenaAudio.volume;
        }
        
        this.prevRoundName = null;
        this.prevHandNumber = 0;
        this.processedActionKeys = new Set();
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
            if (confirm('게임머니($10,000)와 모든 대전 기록을 처음으로 초기화하시겠습니까?')) {
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

        // 사운드 음소거 토글
        if (this.btnAudioMute) {
            this.btnAudioMute.addEventListener('click', () => {
                if (window.arenaAudio) {
                    const isMuted = window.arenaAudio.toggleMute();
                    this.btnAudioMute.innerText = isMuted ? '🔇' : '🔊';
                }
            });
        }

        // 초보자 가이드 토글
        if (this.btnGuideToggle) {
            this.btnGuideToggle.addEventListener('click', () => {
                this.guideEnabled = !this.guideEnabled;
                document.body.classList.toggle('guide-mode-off', !this.guideEnabled);
                this.btnGuideToggle.innerText = this.guideEnabled ? '💡 가이드 ON' : '💡 가이드 OFF';
                this.btnGuideToggle.classList.toggle('active', this.guideEnabled);
            });
        }

        // 게임 속도 토글 (1.0x <-> 2.0x)
        if (this.btnSpeedToggle) {
            this.btnSpeedToggle.addEventListener('click', () => {
                this.gameSpeed = this.gameSpeed === 1.0 ? 2.0 : 1.0;
                this.btnSpeedToggle.innerText = this.gameSpeed === 1.0 ? '⚡ 속도: 보통' : '⚡ 속도: 빠름 (2x)';
                this.btnSpeedToggle.classList.toggle('fast-mode', this.gameSpeed === 2.0);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'set_speed', speed: this.gameSpeed }));
                }
            });
        }
    }

    renderCard(cardStr, extraClass = '') {
        if (!cardStr || cardStr === '??') {
            return `<div class="card card-back ${extraClass}"></div>`;
        }
        const rank = cardStr[0];
        const suitChar = cardStr[1];
        const suitSymbol = SUIT_SYMBOLS[suitChar] || suitChar;
        const colorClass = SUIT_COLORS[suitChar] || 'black';
        
        return `
            <div class="card ${colorClass} ${extraClass}">
                <div class="card-corner card-corner-top">
                    <span class="corner-rank">${rank}</span>
                    <span class="corner-suit">${suitSymbol}</span>
                </div>
                <div class="card-center-suit">${suitSymbol}</div>
                <div class="card-corner card-corner-bottom">
                    <span class="corner-rank">${rank}</span>
                    <span class="corner-suit">${suitSymbol}</span>
                </div>
            </div>
        `;
    }

    renderCommunityBoard(boardCards) {
        if (!this.boardCardsContainer) return;
        
        const slotLabels = ["플랍 1", "플랍 2", "플랍 3", "턴 4", "리버 5"];
        const cards = boardCards || [];
        
        let html = '';
        for (let i = 0; i < 5; i++) {
            const cardStr = cards[i];
            const slotName = slotLabels[i];
            const slotClass = i < 3 ? 'slot-flop' : (i === 3 ? 'slot-turn' : 'slot-river');
            
            if (cardStr) {
                const isNew = !this.renderedBoardCards.includes(cardStr);
                const anim = isNew ? 'animate-card-flip' : '';
                html += `
                    <div class="board-slot ${slotClass} active">
                        <span class="slot-badge">${slotName}</span>
                        ${this.renderCard(cardStr, anim)}
                    </div>
                `;
            } else {
                html += `
                    <div class="board-slot ${slotClass} empty">
                        <span class="slot-badge">${slotName}</span>
                        <div class="card-slot-placeholder">
                            <span class="ph-icon">🂠</span>
                            <span class="ph-text">${slotName}</span>
                        </div>
                    </div>
                `;
            }
        }
        
        this.boardCardsContainer.innerHTML = html;
        this.renderedBoardCards = [...cards];
        
        // 공용 카드 상태 서브텍스트
        if (this.commStageText) {
            const cnt = cards.length;
            if (cnt === 0) {
                this.commStageText.innerText = "프리플랍 진행 중 (공용 카드 3장 오픈 대기)";
            } else if (cnt === 3) {
                this.commStageText.innerText = "플랍 3장 오픈됨 (턴 카드 대기)";
            } else if (cnt === 4) {
                this.commStageText.innerText = "턴 4번째 카드 오픈됨 (마지막 리버 대기)";
            } else if (cnt === 5) {
                this.commStageText.innerText = "리버 5장 전체 오픈 완료 (쇼다운 판정)";
            }
        }
    }

    formatMoney(amount, showSign = false) {
        const val = Math.round(amount || 0);
        const sign = val > 0 && showSign ? '+' : (val < 0 ? '-' : '');
        return `${sign}$${Math.abs(val).toLocaleString()}`;
    }

    onStateUpdate(state) {
        this.state = state;
        
        // 0. 실감형 음성 및 효과음 디스패치 (DIRECTIVE 028)
        this.handleGameAudioAndAnimations(state);
        
        // 1. 헤더 통계 (가상 달러 $ 단위, BB 노출 금지)
        const stats = state.hero_stats || {};
        if (this.statHands) this.statHands.innerText = (stats.hands || 0) + '판';
        
        const balance = stats.current_balance !== undefined ? stats.current_balance : 10000;
        if (this.statBalance) this.statBalance.innerText = this.formatMoney(balance);
        
        const net = stats.total_profit_dollars !== undefined ? stats.total_profit_dollars : 0;
        if (this.statNet) {
            const prefix = net > 0 ? '이익 ' : (net < 0 ? '손실 ' : '');
            this.statNet.innerText = prefix + this.formatMoney(net, true);
            this.statNet.className = 'profit-val ' + (net > 0 ? 'positive' : (net < 0 ? 'negative' : 'neutral'));
        }
        
        const lastProfit = stats.last_hand_profit_dollars !== undefined ? stats.last_hand_profit_dollars : 0;
        if (this.statLastHand) {
            const lastPrefix = lastProfit > 0 ? '획득 ' : (lastProfit < 0 ? '손실 ' : '');
            this.statLastHand.innerText = lastPrefix + this.formatMoney(lastProfit, true);
            this.statLastHand.className = 'last-hand-val ' + (lastProfit > 0 ? 'positive' : (lastProfit < 0 ? 'negative' : 'neutral'));
        }
        
        if (this.statWinrate) {
            this.statWinrate.innerText = (stats.win_rate_pct || 0.0).toFixed(1) + '%';
        }
        
        // 2. 테이블 중앙 (팟 & 스트리트 & 5구 공용 카드 슬롯)
        const pot = state.pot_dollars !== undefined ? state.pot_dollars : Math.round((state.pot_bb || 0) * 100);
        this.potDisplay.innerText = `팟 ${this.formatMoney(pot)}`;
        this.streetDisplay.innerText = state.round_name || '프리플랍 (Preflop)';
        
        // 단계별 쉬운 한글 팁
        if (this.streetHelpTip) {
            const rName = state.round_name || '';
            if (rName.includes('프리플랍')) {
                this.streetHelpTip.innerText = "개인 카드 2장을 받았습니다. 베팅에 참여할지 결정하세요.";
            } else if (rName.includes('플랍')) {
                this.streetHelpTip.innerText = "모두가 공유하는 공용 카드 3장이 공개되었습니다!";
            } else if (rName.includes('턴')) {
                this.streetHelpTip.innerText = "4번째 공용 카드가 추가되었습니다.";
            } else if (rName.includes('리버')) {
                this.streetHelpTip.innerText = "마지막 5번째 공용 카드가 공개되었습니다. 최종 쇼다운을 준비하세요.";
            } else if (rName.includes('쇼다운')) {
                this.streetHelpTip.innerText = "모든 베팅 종료! 카드를 열어 승자를 확인합니다.";
            }
        }
        
        // 5구 공용 카드 슬롯 렌더링
        this.renderCommunityBoard(state.board_cards);
        
        // 3. 5인 좌석 렌더링 ($ 단위 표시)
        (state.players || []).forEach(p => {
            const seatEl = document.getElementById(`seat-${p.seat}`);
            if (!seatEl) return;
            
            seatEl.className = 'player-seat seat-' + p.seat;
            if (p.is_active) seatEl.classList.add('active');
            if (p.is_folded) seatEl.classList.add('folded');
            
            const isHero = (p.seat === 0);
            const handLabel = isHero ? `<div class="hero-hand-badge">내 개인 카드 2장</div>` : '';
            const cardsHtml = (p.cards || []).map(c => this.renderCard(c)).join('');
            const dealerBadge = p.is_dealer ? `<span class="dealer-button" title="딜러 버튼">D</span>` : '';
            
            const stackVal = p.stack_dollars !== undefined ? p.stack_dollars : Math.round(p.stack_bb * 100);
            const betVal = p.current_bet_dollars !== undefined ? p.current_bet_dollars : Math.round(p.current_bet_bb * 100);
            const betBadge = betVal > 0 ? `<div class="player-bet-badge">베팅: ${this.formatMoney(betVal)}</div>` : '';
            
            seatEl.innerHTML = `
                <div class="player-header">
                    <span>${p.avatar} ${p.name}</span>
                    ${dealerBadge}
                </div>
                <div class="player-stack">${this.formatMoney(stackVal)}</div>
                ${betBadge}
                ${handLabel}
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
        
        // 5. 인간(Hero) 액션 버튼 제어 및 금액 업데이트 ($ 단위)
        const legal = state.legal_actions || [];
        const raises = state.raise_amounts_dollars || {};
        const callCost = state.call_cost_dollars !== undefined ? state.call_cost_dollars : Math.round((state.call_cost_bb || 0) * 100);
        
        Object.entries(this.buttons).forEach(([actId, btn]) => {
            const aNum = parseInt(actId);
            const isLegal = isHumanTurn && legal.includes(aNum);
            btn.disabled = !isLegal;
            
            if (aNum === 1) {
                if (callCost > 0) {
                    this.lblCall.innerText = `콜 ${this.formatMoney(callCost)}`;
                    if (this.hintCall) this.hintCall.innerText = `베팅 맞추기 (${this.formatMoney(callCost)})`;
                } else {
                    this.lblCall.innerText = `체크 (Check)`;
                    if (this.hintCall) this.hintCall.innerText = `차례 넘기기 (비용 $0)`;
                }
            }
            if (aNum === 2) {
                const amt = raises['2'] !== undefined ? raises['2'] : null;
                this.lblRaise50.innerText = amt ? `50% 레이즈 (${this.formatMoney(amt)})` : `50% 팟 레이즈`;
                if (this.hintRaise50) this.hintRaise50.innerText = amt ? `판돈 ${this.formatMoney(amt)}로 올리기` : `판돈 올리기`;
            }
            if (aNum === 3) {
                const amt = raises['3'] !== undefined ? raises['3'] : null;
                this.lblRaise100.innerText = amt ? `100% 레이즈 (${this.formatMoney(amt)})` : `100% 팟 레이즈`;
                if (this.hintRaise100) this.hintRaise100.innerText = amt ? `판돈 2배 (${this.formatMoney(amt)})` : `판돈 2배 올리기`;
            }
            if (aNum === 4) {
                const amt = raises['4'] !== undefined ? raises['4'] : null;
                this.lblRaise200.innerText = amt ? `200% 레이즈 (${this.formatMoney(amt)})` : `200% 팟 레이즈`;
                if (this.hintRaise200) this.hintRaise200.innerText = amt ? `판돈 3배 (${this.formatMoney(amt)})` : `판돈 3배 올리기`;
            }
            if (aNum === 5) {
                const amt = raises['5'] !== undefined ? raises['5'] : null;
                this.lblAllin.innerText = amt ? `올인 (${this.formatMoney(amt)})` : `올인 (All-In)`;
                if (this.hintAllin) this.hintAllin.innerText = amt ? `남은 ${this.formatMoney(amt)} 전부 걸기` : `가진 돈 전부 걸기`;
            }
        });
        
        // 새 핸드 버튼
        this.btnNewHand.disabled = !state.is_terminal;
        
        // 6. AI 상대별 전적 렌더링
        if (this.h2hRows && stats.ai_records) {
            this.h2hRows.innerHTML = stats.ai_records.map(r => `
                <div class="h2h-row">
                    <span class="h2h-name">${r.avatar} ${r.name}</span>
                    <span class="h2h-score"><span class="badge-win">${r.wins}승</span> <span class="badge-loss">${r.losses}패</span> <span class="badge-tie">${r.ties}무</span></span>
                </div>
            `).join('');
        }
        
        // 7. 최근 로그 렌더링
        if (state.recent_logs && state.recent_logs.length > 0) {
            this.actionLog.innerHTML = state.recent_logs.map(line => `<div class="log-entry">${line}</div>`).join('');
            this.actionLog.scrollTop = this.actionLog.scrollHeight;
        }
    }

    handleGameAudioAndAnimations(state) {
        if (!window.arenaAudio) return;

        // 핸드 변경 시 오디오 매니저 큐 즉시 동기화
        if (state.hand_number) {
            window.arenaAudio.setCurrentHand(state.hand_number);
        }

        // A. 새 핸드 시작 감지
        if (state.hand_number !== this.prevHandNumber && state.hand_number > 0) {
            this.prevHandNumber = state.hand_number;
            this.prevRoundName = null;
            this.renderedBoardCards = [];
            this.processedActionKeys.clear();
            
            window.arenaAudio.playSfx('card_shuffle');
            window.arenaAudio.playDealerVoice('start', state.hand_number);
            window.arenaAudio.playSfx('card_deal');
        }

        // B. 스트리트 전환 감지 (플랍, 턴, 리버, 쇼다운)
        if (state.round_name && state.round_name !== this.prevRoundName) {
            const isInitial = (this.prevRoundName === null);
            this.prevRoundName = state.round_name;

            if (!isInitial) {
                if (state.round_name.includes('플랍')) {
                    window.arenaAudio.playDealerVoice('flop', state.hand_number);
                    window.arenaAudio.playSfx('card_flip');
                } else if (state.round_name.includes('턴')) {
                    window.arenaAudio.playDealerVoice('turn', state.hand_number);
                    window.arenaAudio.playSfx('card_flip');
                } else if (state.round_name.includes('리버')) {
                    window.arenaAudio.playDealerVoice('river', state.hand_number);
                    window.arenaAudio.playSfx('card_flip');
                } else if (state.round_name.includes('쇼다운')) {
                    window.arenaAudio.playDealerVoice('showdown', state.hand_number);
                    window.arenaAudio.playSfx('card_flip');
                }
            }
        }

        // C. 마지막 액션 이벤트 (AI 및 인간) 처리 & 캐릭터 강조 애니메이션
        const ev = state.last_event;
        if (ev && ev.type === 'action') {
            const evKey = `${state.hand_number}_${ev.seat}_${ev.action_id}_${ev.amount}_${state.recent_logs ? state.recent_logs.length : 0}`;
            if (!this.processedActionKeys.has(evKey)) {
                this.processedActionKeys.add(evKey);

                // 좌석 하이라이트 애니메이션
                const seatEl = document.getElementById(`seat-${ev.seat}`);
                if (seatEl) {
                    const animClass = `action-highlight-${ev.action_key}`;
                    seatEl.classList.add(animClass);
                    setTimeout(() => {
                        seatEl.classList.remove(animClass);
                    }, 1400);
                }

                // 음성 및 물리 효과음 동시 트리거 (현재 핸드 번호 바인딩)
                window.arenaAudio.handleActionSound(ev.seat, ev.char_id, ev.action_key, state.hand_number);
            }
        } else if (ev && ev.type === 'street_deal') {
            const dealKey = `deal_${state.hand_number}_${ev.round_name}`;
            if (!this.processedActionKeys.has(dealKey)) {
                this.processedActionKeys.add(dealKey);
                window.arenaAudio.playSfx('card_flip');
            }
        } else if (ev && ev.type === 'terminal') {
            const termKey = `term_${state.hand_number}`;
            if (!this.processedActionKeys.has(termKey)) {
                this.processedActionKeys.add(termKey);
                
                if (ev.hero_win) {
                    window.arenaAudio.playDealerVoice('hero_win', state.hand_number);
                    window.arenaAudio.playSfx('victory');
                    window.arenaAudio.playSfx('pot_win');
                } else {
                    window.arenaAudio.playDealerVoice('winner', state.hand_number);
                    window.arenaAudio.playSfx('defeat');
                }
            }
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
