// POKER AI HUMAN ARENA: Advanced AudioManager (DIRECTIVE 028 Integrated)
// ======================================================================
// Upgraded audio engine featuring:
// - audio.onended event-driven asynchronous sequential playback (no fixed setTimeout)
// - Drop stale events (>2.5s old or from previous hand) to prevent lag build-up
// - Instant queue clear & active sound cancellation on new hand or mute
// - Zero audio overlaps for voices while allowing layered physical sound effects
// - Web Audio unlock on first user gesture

class AudioManager {
    constructor() {
        this.enabled = true;
        this.muted = localStorage.getItem('arena_audio_muted') === 'true';
        this.volume = parseFloat(localStorage.getItem('arena_audio_volume') || '0.75');
        
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentAudio = null;
        this.currentHand = 0;
        this.unlocked = false;
        
        this.initAudioContext();
    }

    initAudioContext() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                this.ctx = new AudioContext();
            }
        } catch (e) {
            console.warn('AudioContext not supported:', e);
        }

        const unlock = () => {
            if (this.unlocked) return;
            if (this.ctx && this.ctx.state === 'suspended') {
                this.ctx.resume();
            }
            this.unlocked = true;
            console.log('[AudioManager] Audio unlocked by user gesture');
            window.removeEventListener('click', unlock);
            window.removeEventListener('touchstart', unlock);
            window.removeEventListener('keydown', unlock);
        };

        window.addEventListener('click', unlock, { passive: true });
        window.addEventListener('touchstart', unlock, { passive: true });
        window.addEventListener('keydown', unlock, { passive: true });
    }

    setVolume(vol) {
        this.volume = Math.max(0.0, Math.min(1.0, vol));
        localStorage.setItem('arena_audio_volume', this.volume.toString());
        if (this.currentAudio) {
            this.currentAudio.volume = this.volume;
        }
    }

    toggleMute() {
        this.muted = !this.muted;
        localStorage.setItem('arena_audio_muted', this.muted ? 'true' : 'false');
        if (this.muted) {
            this.stopAllAndClear();
        }
        return this.muted;
    }

    isMuted() {
        return this.muted;
    }

    setCurrentHand(handNum) {
        if (handNum !== this.currentHand) {
            this.currentHand = handNum;
            // 새 핸드 시작 시 이전 핸드의 모든 음성 즉시 제거 및 중지
            this.stopAllAndClear();
        }
    }

    stopAllAndClear() {
        this.audioQueue = [];
        if (this.currentAudio) {
            try {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
            } catch (e) {}
            this.currentAudio = null;
        }
        this.isPlaying = false;
    }

    // 오디오 항목 큐에 인큐
    enqueue(url, type = 'sfx', handNum = 0) {
        if (this.muted) return;
        
        const timestamp = Date.now();
        this.audioQueue.push({
            url,
            type,
            handNum: handNum || this.currentHand,
            timestamp
        });

        if (!this.isPlaying) {
            this.playNext();
        }
    }

    async playNext() {
        if (this.muted || this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        this.isPlaying = true;
        const item = this.audioQueue.shift();
        const now = Date.now();

        // 1. 오래된 이벤트(2.5초 이상 지연되었거나 이전 핸드 것) 자동 폐기
        if ((now - item.timestamp > 2500) || (item.handNum > 0 && item.handNum < this.currentHand)) {
            console.debug('[AudioManager] Dropped stale audio:', item.url);
            this.playNext();
            return;
        }

        try {
            const audio = new Audio(item.url);
            audio.volume = this.volume;
            this.currentAudio = audio;

            // 2. 실제 audio.onended 이벤트 기반 순차 재생
            await new Promise((resolve) => {
                let resolved = false;
                const finish = () => {
                    if (!resolved) {
                        resolved = true;
                        resolve();
                    }
                };

                audio.onended = finish;
                audio.onerror = finish;
                
                // 최대 타임아웃 방어 (음성 3초, SFX 1.5초 이상 걸리면 강제 통과)
                const maxSafetyMs = item.type === 'voice' ? 3000 : 1500;
                setTimeout(finish, maxSafetyMs);

                audio.play().catch(err => {
                    console.debug('[AudioManager] Play failed/blocked:', item.url, err);
                    finish();
                });
            });

            this.currentAudio = null;
        } catch (e) {
            console.warn('[AudioManager] Error playing:', item.url, e);
        }

        // 약간의 자연스러운 호흡(100ms) 후 다음 큐 처리
        setTimeout(() => this.playNext(), 100);
    }

    // 캐릭터별 음성 재생 (김프로, 제니, 잭, 미스터 K)
    playCharacterVoice(charId, actionKey, handNum = 0) {
        if (!charId || !actionKey || this.muted) return;
        const validChars = ['kim', 'jenny', 'jack', 'mr_k'];
        if (!validChars.includes(charId)) return;
        
        const url = `/audio/voices/${charId}/${actionKey}.mp3`;
        this.enqueue(url, 'voice', handNum);
    }

    // 딜러 안내 음성 재생
    playDealerVoice(cueKey, handNum = 0) {
        if (!cueKey || this.muted) return;
        const url = `/audio/voices/dealer/${cueKey}.mp3`;
        this.enqueue(url, 'voice', handNum);
    }

    // 효과음(SFX) 재생 (물리 효과음은 큐를 막지 않고 오버랩 재생 허용)
    playSfx(sfxKey) {
        if (this.muted) return;
        try {
            const audio = new Audio(`/audio/sfx/${sfxKey}.mp3`);
            audio.volume = this.volume;
            audio.play().catch(() => {});
        } catch (e) {}
    }

    // 액션 사운드 디스패치 (음성과 물리 효과음의 이상적인 조화)
    handleActionSound(seat, charId, actionKey, handNum = 0) {
        // 1. 물리적 세라믹 칩 및 카드 효과음은 즉각 발동
        switch (actionKey) {
            case 'check':
                this.playSfx('check_knock');
                break;
            case 'call':
                this.playSfx('chip_call');
                break;
            case 'raise':
                this.playSfx('chip_raise');
                break;
            case 'allin':
                this.playSfx('chip_allin');
                break;
            case 'fold':
                this.playSfx('card_flip');
                break;
        }

        // 2. 캐릭터 음성은 순차 큐를 통해 발화
        if (seat !== 0 && charId && charId !== 'hero') {
            this.playCharacterVoice(charId, actionKey, handNum);
        }
    }
}

// 글로벌 인스턴스
window.arenaAudio = new AudioManager();
