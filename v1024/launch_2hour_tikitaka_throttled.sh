#!/bin/bash
# POKER AI V10.24 2-HOUR AUTONOMOUS RL TIKITAKA RUNNER (STRICT CPU 70% THROTTLED)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting 2-Hour Autonomous RL Tikitaka Pipeline"
echo "Target: ChatGPT Session 6adca81a-6da4-83e8-baef-2b48268858b1"
echo "Strict CPU Throttle: 70%"
echo "Start: $(date)"
echo "=========================================================="

python3 v1024/run_2hour_autonomous_tikitaka.py > v1024/tikitaka_2hour.log 2>&1 &
RUN_PID=$!
echo "Tikitaka Process PID: $RUN_PID"

python3 cpu_throttler.py $RUN_PID 0.70 > v1024/throttler_2hour_tikitaka.log 2>&1 &
THROTTLE_PID=$!
echo "CPU Throttler PID: $THROTTLE_PID"

echo "Waiting for 2-hour tikitaka run to complete..."
wait $RUN_PID
RUN_EXIT=$?

echo "2-Hour Tikitaka process exited with code $RUN_EXIT"
kill $THROTTLE_PID 2>/dev/null || true

exit $RUN_EXIT
