#!/bin/bash
# V10.24 5시간 강화학습 & 자율 핑퐁 실행 스크립트 (CPU 70% 제한)
cd /Volumes/NVME/7.AI_vibe_coding/hangame

echo "[V10.24] 5-Hour Autonomous RL & Ping-Pong Pipeline Starting..."
python3 v1024/train_v1024_5hour_autonomous_loop.py > v1024/train_5hour.log 2>&1 &
TRAIN_PID=$!
echo "  학습 메인 PID: $TRAIN_PID"

# CPU 70% Throttler 병렬 체결 (비율 0.70)
python3 cpu_throttler.py $TRAIN_PID 0.70 >> v1024/throttler_70.log 2>&1 &
THROTTLER_PID=$!
echo "  스로틀러 PID: $THROTTLER_PID (CPU 70% 제한)"

wait $TRAIN_PID
EXIT_CODE=$?
echo "[V10.24] 5시간 강화학습 파이프라인 완료 (exit code: $EXIT_CODE)"
