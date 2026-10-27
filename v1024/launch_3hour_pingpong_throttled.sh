#!/bin/bash
# POKER AI V10.24 3-HOUR AUTONOMOUS RL PINGPONG RUNNER (STRICT CPU 70% THROTTLED)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting 3-Hour Autonomous RL Pingpong Pipeline"
echo "Target: ChatGPT Session 6adca81a-6da4-83e8-baef-2b48268858b1 ('홀덤 기계적 전략')"
echo "Strict CPU Throttle: 70%"
echo "Start: $(date)"
echo "=========================================================="

python3 v1024/run_3hour_autonomous_pingpong.py > v1024/pingpong_3hour.log 2>&1 &
RUN_PID=$!
echo "Pingpong Process PID: $RUN_PID"

python3 cpu_throttler.py $RUN_PID 0.70 > v1024/throttler_3hour_pingpong.log 2>&1 &
THROTTLE_PID=$!
echo "CPU Throttler PID: $THROTTLE_PID"

echo "3-Hour Pingpong pipeline launched successfully in background."
echo "Logs: tail -f v1024/pingpong_3hour.log"
