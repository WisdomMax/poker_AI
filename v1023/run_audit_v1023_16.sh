#!/bin/bash
# V10.23.16 실행 스크립트 (CPU 80% 제한 포함)
cd /Volumes/NVME/7.AI_vibe_coding/hangame

echo "[V10.23.16] Rigorous MC Stability & Multiway Tie Audit 시작..."
python3 v1023/v1023_16_rigorous_stability_and_multiway_audit.py > v1023/v1023_16_run.log 2>&1 &
EVAL_PID=$!
echo "  실험 PID: $EVAL_PID"

# CPU 80% 스로틀러 실행
python3 cpu_throttler.py $EVAL_PID 0.80 >> v1023/throttler_v1023_16.log 2>&1 &
THROTTLER_PID=$!
echo "  스로틀러 PID: $THROTTLER_PID"

wait $EVAL_PID
EXIT_CODE=$?
echo "[V10.23.16] 실험 완료 (exit code: $EXIT_CODE)"
