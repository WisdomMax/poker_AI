// POKER AI HUMAN ARENA: AudioManager (DIRECTIVE 025)
// =======================================================
// Manages real casino sound effects, AI Korean voices, dealer announcements,
// audio queueing, volume control, and browser autoplay unlocks.

class AudioManager {
    constructor() {
        this.enabled = true;
        this.muted = localStorage.getItem('arena_audio_muted') === 'true';
        this.volume = parseFloat(localStorage.getItem('arena_audio_volume') || '0.75');
        
        this.audioCache = new Map();
        this.audioQueue = [];
        this.isPlayingQueue = false;
        this.lastPlayedMap = new Map();
        this.minIntervalMs = 280; // 동일 오디오 연타 방지 쿨다운
        this.unlocked = false;
        
        this.initAudioContext();
        this.preloadEssentialAudio();
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

        // 브라우저 첫 사용자 제스처 시 오디오 언락
        const unlock = () => {
            if (this.unlocked) return;
            if (this.ctx && this.ctx.state === 'suspended') {
                this.ctx.resume();
            }
            this.unlocked = true;
            console.log('AudioManager unlocked by user interaction');
            window.removeEventListener('click', unlock);
            window.removeEventListener('keydown', unlock);
            window.removeEventListener('touchstart', unlock);
        };

        window.addEventListener('click', unlock);
        window.addEventListener('keydown', unlock);
        window.addEventListener('touchstart', unlock);
    }

    preloadEssentialAudio() {
        // 주요 SFX 및 딜러 보이스 사전 로드
        const essentials = [
            '/audio/sfx/chip_call.mp3',
            '/audio/sfx/chip_raise.mp3',
            '/audio/sfx/check_knock.mp3',
            '/audio/sfx/card_deal.mp3',
            '/audio/sfx/card_flip.mp3',
            '/audio/sfx/pot_win.mp3',
            '/audio/sfx/victory.mp3',
            '/audio/voices/dealer/start.mp3'
        ];
        essentials.forEach(url => this.getOrCreateAudio(url));
    }

    getOrCreateAudio(url) {
        if (!this.audioCache.has(url)) {
            const audio = new Audio(url);
            audio.preload = 'auto';
            this.audioCache.set(url, audio);
        }
        return this.audioCache.get(url);
    }

    setVolume(vol) {
        this.volume = Math.max(0.0, Math.min(1.0, vol));
        localStorage.setItem('arena_audio_volume', this.volume.toString());
        this.audioCache.forEach(a => { a.volume = this.volume; });
    }

    toggleMute() {
        this.muted = !this.muted;
        localStorage.setItem('arena_audio_muted', this.muted ? 'true' : 'false');
        return this.muted;
    }

    isMuted() {
        return this.muted;
    }

    // 오디오 즉시 또는 큐 재생
    enqueue(url, type = 'sfx', delayBefore = 0) {
        if (this.muted) return;
        
        // 쿨다운 검사
        const now = Date.now();
        const lastPlay = this.lastPlayedMap.get(url) || 0;
        if (now - lastPlay < this.minIntervalMs) {
            return;
        }
        this.lastPlayedMap.set(url, now);

        this.audioQueue.push({ url, type, delayBefore });
        if (!this.isPlayingQueue) {
            this.processQueue();
        }
    }

    async processQueue() {
        if (this.audioQueue.length === 0) {
            this.isPlayingQueue = false;
            return;
        }

        this.isPlayingQueue = true;
        const item = this.audioQueue.shift();

        if (item.delayBefore > 0) {
            await new Promise(r => setTimeout(r, item.delayBefore));
        }

        if (!this.muted) {
            try {
                // 클론 오디오로 오버랩 재생 허용
                const audio = new Audio(item.url);
                audio.volume = this.volume;
                await audio.play();
                
                // 음성(voice)은 약간의 시간 간격(약 380ms)을 두고 다음 음성 재생
                const waitTime = item.type === 'voice' ? 420 : 180;
                await new Promise(r => setTimeout(r, waitTime));
            } catch (err) {
                // 브라우저 정책으로 재생 차단된 경우 가볍게 스킵
                console.debug('Audio play skipped (waiting user gesture):', item.url);
            }
        }

        this.processQueue();
    }

    // 캐릭터별 음성 재생 ("김프로", "제니", "잭", "미스터 K")
    playCharacterVoice(charId, actionKey) {
        if (!charId || !actionKey) return;
        const validChars = ['kim', 'jenny', 'jack', 'mr_k'];
        if (!validChars.includes(charId)) return;
        
        const url = `/audio/voices/${charId}/${actionKey}.mp3`;
        this.enqueue(url, 'voice');
    }

    // 딜러 음성 재생
    playDealerVoice(cueKey, delayMs = 150) {
        const url = `/audio/voices/dealer/${cueKey}.mp3`;
        this.enqueue(url, 'voice', delayMs);
    }

    // 효과음(SFX) 재생
    playSfx(sfxKey, delayMs = 0) {
        const url = `/audio/sfx/${sfxKey}.mp3`;
        this.enqueue(url, 'sfx', delayMs);
    }

    // 포커 액션 종합 사운드 디스패치 (음성 + 적절한 효과음 동시 트리거)
    handleActionSound(seat, charId, actionKey, amount = 0) {
        // 1. 캐릭터 음성 (인간 Hero 제외)
        if (seat !== 0 && charId && charId !== 'hero') {
            this.playCharacterVoice(charId, actionKey);
        }

        // 2. 물리 효과음 동기화
        switch (actionKey) {
            case 'check':
                this.playSfx('check_knock', 50);
                break;
            case 'call':
                this.playSfx('chip_call', 60);
                break;
            case 'raise':
                this.playSfx('chip_raise', 60);
                break;
            case 'allin':
                this.playSfx('chip_allin', 50);
                break;
            case 'fold':
                this.playSfx('card_flip', 80);
                break;
        }
    }
}

// 글로벌 인스턴스
window.arenaAudio = new AudioManager();
