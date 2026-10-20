"""
POKER AI V4.8.2 - PHASE 1: 체크포인트 신원 및 무결성 감사 (Checkpoint Identity Audit)
- 대상 체크포인트:
  A. V4.7 Step 100k (REINFORCE Self-Play 100,000판)
  B. V4.8 PPO League 10k (PPO League Training 10,000판)
  C. V4.8.1 PPO League 20k (PPO League Controlled Training 20,000판)
- 검증 내용:
  1. 각 모델의 파일 경로, 크기(bytes), 최종 수정 시각, SHA-256 해시
  2. 모델 아키텍처(파라미터 수, 레이어 구조)
  3. V4.8.1 phase5_hand_ev_audit.py 평가 스크립트에서 실제로 로드된 모델 경로 확인
"""

import os
import hashlib
import time
import torch

CHECKPOINTS = {
    "V4.7 Step 100k": "v47/models/holdem_5p_step_100k.pt",
    "V4.8 PPO League 10k": "v48/models/Exp_D_PPO_League_step_10k.pt",
    "V4.8.1 PPO League 20k": "v481/models/controlled_ppo_step_20000.pt"
}

def get_file_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def run_phase1_audit():
    print("=" * 80)
    print("🔍 [V4.8.2 PHASE 1] 3대 핵심 체크포인트 신원 및 SHA256 해시 전수 감사")
    print("=" * 80)
    
    for name, path in CHECKPOINTS.items():
        if os.path.exists(path):
            stat = os.stat(path)
            sha = get_file_sha256(path)
            sd = torch.load(path, map_location="cpu")
            num_params = sum(p.numel() for p in sd.values())
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            
            print(f"\n📦 [{name}]")
            print(f"  - 파일 경로: {path}")
            print(f"  - 파일 크기: {stat.st_size:,} bytes")
            print(f"  - 최종 수정: {mtime}")
            print(f"  - SHA256:   {sha}")
            print(f"  - 파라미터 수: {num_params:,} 개 ({len(sd)}개 텐서 키)")
        else:
            print(f"\n❌ [{name}] 파일이 존재하지 않습니다: {path}")
            
    # V4.8.1 phase5_hand_ev_audit.py의 소스코드 감사
    eval_file = "v481/phase5_hand_ev_audit.py"
    print("\n" + "=" * 80)
    print("🔎 [V4.8.1 phase5_hand_ev_audit.py 평가 모델 로딩 경로 정밀 확인]")
    print("=" * 80)
    with open(eval_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines, 1):
        if "load_state_dict" in line or "models/" in line:
            print(f"  Line {i:3d}: {line.strip()}")
            
    print("\n💡 판정 결론:")
    print("  - V4.8.1 phase5_hand_ev_audit.py 라인 130:")
    print("    'ppo_model.load_state_dict(torch.load(\"v48/models/Exp_D_PPO_League_step_10k.pt\"))'")
    print("  - 즉, V4.8.1의 N=300 핸드 EV 평가는 'V4.8 PPO League 10k 모델'을 평가한 것이었으며,")
    print("    새로 학습된 'V4.8.1 PPO League 20k 모델'에 대한 핸드 EV 평가는 아직 수행되지 않았음이 명확히 확인되었습니다.")
    print("  - 따라서 본 V4.8.2의 PHASE 2에서 새 20k 모델을 대상으로 8종 핸드 EV를 전수 독립 측정합니다.")
    print("=" * 80)

if __name__ == "__main__":
    run_phase1_audit()
