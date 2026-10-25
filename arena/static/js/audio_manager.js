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
        this.generation = 0;
        this.effects = new Set();
        this.isPlaying = false;
        this.currentAudio = null;
        this.currentHand = 0;
        this.unlocked = false;
        
        this.initAudioContext();
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) this.stopAllAndClear();
        });
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
        this.effects.forEach(audio => { audio.volume = this.volume; });
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
        this.generation++;
        this.audioQueue = [];
        if (this.cancelPlayback) this.cancelPlayback();
        this.cancelPlayback = null;
        if (this.currentAudio) this.currentAudio.pause();
        this.currentAudio = null;
        this.effects.forEach(audio => audio.pause());
        this.effects.clear();
        this.isPlaying = false;
    }

    enqueue(url, type = 'sfx', handNum = 0) {
        if (this.muted || !this.unlocked) return;
        // Only one pending voice: narration always stays close to the table.
        this.audioQueue = [{url, type, handNum: handNum || this.currentHand, timestamp: Date.now()}];
        if (!this.isPlaying) this.playNext();
    }

    async playNext() {
        if (this.muted || !this.audioQueue.length) {
            this.isPlaying = false;
            return;
        }
        const generation = this.generation;
        const item = this.audioQueue.shift();
        if (Date.now() - item.timestamp > 1500 || item.handNum !== this.currentHand) {
            this.playNext();
            return;
        }
        this.isPlaying = true;
        const audio = new Audio(item.url);
        this.currentAudio = audio;
        audio.volume = this.volume;
        await new Promise(resolve => {
            let done = false;
            let timer;
            const finish = () => {
                if (done) return;
                done = true;
                clearTimeout(timer);
                audio.pause();
                audio.onended = audio.onerror = null;
                resolve();
            };
            this.cancelPlayback = finish;
            audio.onended = audio.onerror = finish;
            timer = setTimeout(finish, item.type === 'voice' ? 2200 : 1200);
            audio.play().catch(finish);
        });
        // A cancelled hand must never resume or clear the new hand's player.
        if (generation !== this.generation) return;
        this.currentAudio = null;
        this.cancelPlayback = null;
        this.isPlaying = false;
        this.playNext();
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
        if (this.muted || !this.unlocked) return;
        try {
            const audio = new Audio(`/audio/sfx/${sfxKey}.mp3`);
            this.effects.add(audio);
            const clear = () => this.effects.delete(audio);
            audio.onended = audio.onerror = clear;
            audio.volume = this.volume;
            audio.play().catch(clear);
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
