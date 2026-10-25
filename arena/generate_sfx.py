"""
POKER AI HUMAN ARENA: DIRECTIVE 025 Realistic Casino SFX Generator
==================================================================
Physically-modeled acoustic casino sounds using numpy & ffmpeg:
1. card_shuffle  : Cascading friction bursts (card riffle & shuffle)
2. card_deal     : Quick swoosh/snap of card dealt across felt
3. card_flip     : Crisp card bend & snap
4. check_knock   : Double-knock on wooden felt table (thump-thump)
5. chip_call     : Ceramic clay chip clink (2.8kHz resonant ring)
6. chip_raise    : Stacking clay chips with weighted impacts
7. chip_allin    : Massive cascading chip slide & avalanche
8. pot_win       : Gathering all chips across table (rich sweep)
9. victory       : Uplifting major arpeggio chime (C5-E5-G5-C6)
10. defeat       : Soft melancholic minor resolution chime
"""

import os
import subprocess
import numpy as np

SAMPLE_RATE = 44100

def save_wav_and_mp3(samples: np.ndarray, mp3_path: str):
    # 정규화 (-0.95 ~ +0.95)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.95
    int_samples = (samples * 32767).astype(np.int16)
    
    wav_path = mp3_path.replace(".mp3", ".wav")
    import wave
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_samples.tobytes())
        
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-codec:a", "libmp3lame", "-b:a", "128k",
        mp3_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(wav_path):
        os.remove(wav_path)

def gen_noise(duration, decay=5.0):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    env = np.exp(-decay * t)
    return noise * env

# 1. Card Shuffle
def gen_card_shuffle():
    dur = 1.0
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 20개의 작은 리플 틱 연속
    for i in range(25):
        start_t = 0.05 + i * 0.03 + np.random.uniform(-0.005, 0.005)
        idx = int(start_t * SAMPLE_RATE)
        tick_dur = 0.04
        t_tick = np.linspace(0, tick_dur, int(SAMPLE_RATE * tick_dur), endpoint=False)
        tick = np.random.uniform(-1, 1, len(t_tick)) * np.exp(-70 * t_tick)
        end_idx = min(len(sound), idx + len(tick))
        sound[idx:end_idx] += tick[:end_idx - idx] * np.random.uniform(0.5, 1.0)
    # 전체 셔플 사각거림
    sound += gen_noise(dur, decay=3.0) * 0.15
    return sound

# 2. Card Deal
def gen_card_deal():
    dur = 0.22
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    # 스와이프 노이즈 + 끝에 살짝 스냅
    noise = np.random.uniform(-1, 1, len(t))
    env = np.sin(np.pi * (t / dur)) ** 1.5 * np.exp(-4 * t)
    snap = np.sin(2 * np.pi * 800 * t) * np.exp(-40 * t)
    return noise * env * 0.8 + snap * 0.3

# 3. Card Flip
def gen_card_flip():
    dur = 0.18
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    snap = np.sin(2 * np.pi * 1200 * t) * np.exp(-60 * t)
    noise = np.random.uniform(-1, 1, len(t)) * np.exp(-35 * t)
    return snap * 0.6 + noise * 0.5

# 4. Check Knock (Double knock on felt table)
def gen_check_knock():
    dur = 0.45
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 두 번 노크 (0.05s, 0.20s)
    for knock_t in [0.04, 0.18]:
        idx = int(knock_t * SAMPLE_RATE)
        k_dur = 0.12
        tk = np.linspace(0, k_dur, int(SAMPLE_RATE * k_dur), endpoint=False)
        # 묵직한 우드 타격음 (140Hz 기본음 + 280Hz 하모닉스 + 노이즈 펠트 탭)
        body = (np.sin(2 * np.pi * 140 * tk) * 0.6 + np.sin(2 * np.pi * 260 * tk) * 0.3) * np.exp(-35 * tk)
        thump = np.random.uniform(-1, 1, len(tk)) * np.exp(-80 * tk) * 0.4
        k_sound = body + thump
        end_idx = min(len(sound), idx + len(k_sound))
        sound[idx:end_idx] += k_sound[:end_idx - idx]
    return sound

# 5. Chip Call (Ceramic Clay Clink)
def gen_chip_call():
    dur = 0.35
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 2~3개 칩 충돌
    for offset in [0.02, 0.06, 0.11]:
        idx = int(offset * SAMPLE_RATE)
        c_dur = 0.18
        tc = np.linspace(0, c_dur, int(SAMPLE_RATE * c_dur), endpoint=False)
        # 세라믹 칩 고주파 공명 (2600Hz, 3400Hz, 4800Hz)
        res = (
            np.sin(2 * np.pi * 2700 * tc) * 0.5 +
            np.sin(2 * np.pi * 3500 * tc) * 0.3 +
            np.sin(2 * np.pi * 4900 * tc) * 0.2
        ) * np.exp(-35 * tc)
        clack = np.random.uniform(-1, 1, len(tc)) * np.exp(-120 * tc) * 0.5
        c_sound = res + clack
        end_idx = min(len(sound), idx + len(c_sound))
        sound[idx:end_idx] += c_sound[:end_idx - idx]
    return sound

# 6. Chip Raise (Stacking chips with impact)
def gen_chip_raise():
    dur = 0.5
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 5개의 연속된 칩 스택 임팩트
    offsets = [0.02, 0.07, 0.13, 0.19, 0.26]
    for i, offset in enumerate(offsets):
        idx = int(offset * SAMPLE_RATE)
        c_dur = 0.18
        tc = np.linspace(0, c_dur, int(SAMPLE_RATE * c_dur), endpoint=False)
        pitch_mod = 1.0 + i * 0.04
        res = (
            np.sin(2 * np.pi * (2500 * pitch_mod) * tc) * 0.4 +
            np.sin(2 * np.pi * (3200 * pitch_mod) * tc) * 0.3 +
            np.sin(2 * np.pi * 180 * tc) * 0.4 # 묵직한 스택 베이스
        ) * np.exp(-40 * tc)
        clack = np.random.uniform(-1, 1, len(tc)) * np.exp(-90 * tc) * 0.4
        c_sound = res + clack
        end_idx = min(len(sound), idx + len(c_sound))
        sound[idx:end_idx] += c_sound[:end_idx - idx]
    return sound

# 7. Chip All-in (Heavy avalanche of all chips)
def gen_chip_allin():
    dur = 0.85
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 15개의 무작위 칩 충돌
    for _ in range(16):
        offset = np.random.uniform(0.02, 0.5)
        idx = int(offset * SAMPLE_RATE)
        c_dur = 0.2
        tc = np.linspace(0, c_dur, int(SAMPLE_RATE * c_dur), endpoint=False)
        freq = np.random.uniform(2200, 3800)
        res = np.sin(2 * np.pi * freq * tc) * np.exp(-45 * tc)
        clack = np.random.uniform(-1, 1, len(tc)) * np.exp(-80 * tc) * 0.3
        c_sound = (res + clack) * np.random.uniform(0.4, 0.9)
        end_idx = min(len(sound), idx + len(c_sound))
        sound[idx:end_idx] += c_sound[:end_idx - idx]
    # 전체 칩 슬라이드 서브베이스
    slide = np.sin(2 * np.pi * 90 * t) * np.exp(-4 * t) * 0.3
    sound += slide
    return sound

# 8. Pot Win (Collecting the pot sweep)
def gen_pot_win():
    dur = 0.9
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    # 스윕 마찰음 + 짤랑거림
    for _ in range(12):
        offset = np.random.uniform(0.05, 0.6)
        idx = int(offset * SAMPLE_RATE)
        c_dur = 0.15
        tc = np.linspace(0, c_dur, int(SAMPLE_RATE * c_dur), endpoint=False)
        freq = np.random.uniform(2600, 4200)
        c_sound = np.sin(2 * np.pi * freq * tc) * np.exp(-50 * tc) * 0.4
        end_idx = min(len(sound), idx + len(c_sound))
        sound[idx:end_idx] += c_sound[:end_idx - idx]
    # 기분 좋은 슬라이드
    sweep = np.random.uniform(-1, 1, len(t)) * (np.sin(np.pi * (t / dur)) ** 2) * 0.2
    sound += sweep
    return sound

# 9. Victory (Gentle uplifting major chime: C5, E5, G5, C6)
def gen_victory():
    dur = 1.3
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    notes = [
        (0.04, 523.25),  # C5
        (0.18, 659.25),  # E5
        (0.32, 783.99),  # G5
        (0.48, 1046.50), # C6
    ]
    for start_t, freq in notes:
        idx = int(start_t * SAMPLE_RATE)
        n_dur = dur - start_t
        tn = np.linspace(0, n_dur, int(SAMPLE_RATE * n_dur), endpoint=False)
        # 맑은 벨 하모닉스
        bell = (
            np.sin(2 * np.pi * freq * tn) * 0.6 +
            np.sin(2 * np.pi * freq * 2 * tn) * 0.25 +
            np.sin(2 * np.pi * freq * 3 * tn) * 0.1
        ) * np.exp(-3.5 * tn)
        end_idx = min(len(sound), idx + len(bell))
        sound[idx:end_idx] += bell[:end_idx - idx]
    return sound

# 10. Defeat (Soft minor resolution)
def gen_defeat():
    dur = 1.2
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sound = np.zeros_like(t)
    notes = [
        (0.05, 440.00),  # A4
        (0.25, 392.00),  # G4
        (0.45, 349.23),  # F4
        (0.70, 329.63),  # E4
    ]
    for start_t, freq in notes:
        idx = int(start_t * SAMPLE_RATE)
        n_dur = dur - start_t
        tn = np.linspace(0, n_dur, int(SAMPLE_RATE * n_dur), endpoint=False)
        bell = (
            np.sin(2 * np.pi * freq * tn) * 0.5 +
            np.sin(2 * np.pi * freq * 2 * tn) * 0.2
        ) * np.exp(-3.0 * tn)
        end_idx = min(len(sound), idx + len(bell))
        sound[idx:end_idx] += bell[:end_idx - idx]
    return sound

def main():
    sfx_dir = "arena/static/audio/sfx"
    os.makedirs(sfx_dir, exist_ok=True)
    
    generators = {
        "card_shuffle": gen_card_shuffle,
        "card_deal": gen_card_deal,
        "card_flip": gen_card_flip,
        "check_knock": gen_check_knock,
        "chip_call": gen_chip_call,
        "chip_raise": gen_chip_raise,
        "chip_allin": gen_chip_allin,
        "pot_win": gen_pot_win,
        "victory": gen_victory,
        "defeat": gen_defeat,
    }
    
    print(f"Generating {len(generators)} realistic casino SFX files...")
    for name, gen_fn in generators.items():
        mp3_path = os.path.join(sfx_dir, f"{name}.mp3")
        samples = gen_fn()
        save_wav_and_mp3(samples, mp3_path)
        print(f"  [+] Generated sfx/{name}.mp3 ({len(samples)/SAMPLE_RATE:.2f}s)")
        
    print("All 10 casino SFX files generated successfully!")

if __name__ == "__main__":
    main()
