"""
POKER AI HUMAN ARENA: DIRECTIVE 025 Voice Generator (macOS Local Speech Engine)
=============================================================================
Uses local macOS Korean neural voices:
- Kim Pro (김프로 - Cycle 4): Grandpa (중후한 남성)
- Jenny (제니 - Cycle 13): Yuna (차분한 여성)
- Jack (잭 - Cycle 10+MC): Eddy (자신감 있는 남성)
- Mr. K (미스터 K - Baseline): Reed (낮고 냉철한 남성)
- Dealer (딜러): Yuna -r 190 (정갈하고 명확한 아나운서 톤)
"""

import os
import subprocess
import shutil

VOICE_CONFIGS = {
    "kim": {
        "voice": "Grandpa",
        "rate": 160,
        "lines": {
            "call": "콜!",
            "check": "체크!",
            "fold": "다이!",
            "raise": "레이즈!",
            "allin": "올인!"
        }
    },
    "jenny": {
        "voice": "Yuna",
        "rate": 165,
        "lines": {
            "call": "콜.",
            "check": "체크.",
            "fold": "다이.",
            "raise": "레이즈.",
            "allin": "올인."
        }
    },
    "jack": {
        "voice": "Eddy",
        "rate": 185,
        "lines": {
            "call": "콜!",
            "check": "체크!",
            "fold": "다이!",
            "raise": "레이즈!",
            "allin": "올인!"
        }
    },
    "mr_k": {
        "voice": "Reed",
        "rate": 155,
        "lines": {
            "call": "콜.",
            "check": "체크.",
            "fold": "다이.",
            "raise": "레이즈.",
            "allin": "올인."
        }
    },
    "dealer": {
        "voice": "Yuna",
        "rate": 185,
        "lines": {
            "start": "게임을 시작합니다.",
            "flop": "플랍입니다.",
            "turn": "턴입니다.",
            "river": "리버입니다.",
            "showdown": "쇼다운입니다. 카드를 오픈하세요.",
            "hero_win": "축하합니다! 회원님이 승리하셨습니다.",
            "winner": "핸드가 종료되었습니다.",
            "next_hand": "다음 핸드를 진행합니다."
        }
    }
}

def generate_all():
    base_dir = "arena/static/audio/voices"
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = "/tmp/arena_tts"
    os.makedirs(temp_dir, exist_ok=True)
    
    total = sum(len(cfg["lines"]) for cfg in VOICE_CONFIGS.values())
    count = 0
    print(f"Generating {total} Korean voice files via macOS local speech engine...")

    for char_id, cfg in VOICE_CONFIGS.items():
        char_dir = os.path.join(base_dir, char_id)
        os.makedirs(char_dir, exist_ok=True)
        
        for key, text in cfg["lines"].items():
            count += 1
            aiff_path = os.path.join(temp_dir, f"{char_id}_{key}.aiff")
            mp3_path = os.path.join(char_dir, f"{key}.mp3")
            
            # 1. macOS say 명령으로 AIFF 생성
            cmd_say = [
                "say",
                "-v", cfg["voice"],
                "-r", str(cfg["rate"]),
                "-o", aiff_path,
                text
            ]
            subprocess.run(cmd_say, check=True)
            
            # 2. ffmpeg로 최적화된 웹용 MP3 변환
            cmd_ffmpeg = [
                "ffmpeg", "-y", "-i", aiff_path,
                "-codec:a", "libmp3lame", "-b:a", "64k",
                mp3_path
            ]
            subprocess.run(cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  [{count}/{total}] {char_id}/{key}.mp3 ({cfg['voice']}): \"{text}\"")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("All 28 voice files generated successfully!")

if __name__ == "__main__":
    generate_all()
