#!/bin/bash
# POKER AI V10.24 1-HOUR AUTONOMOUS PING-PONG RUNNER (DIRECTIVE 031 & 030)
# Strict CPU 70% Throttled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting 1-Hour Autonomous Ping-Pong Pipeline (DIRECTIVE 031)"
echo "Target CPU Throttle: 70%"
echo "Start: $(date)"
echo "=========================================================="

python3 v1024/run_1hour_pingpong_directive031.py > v1024/pingpong_1hour_d31.log 2>&1 &
RUN_PID=$!
echo "Ping-Pong Process PID: $RUN_PID"

python3 cpu_throttler.py $RUN_PID 0.70 > v1024/throttler_1hour_d31.log 2>&1 &
THROTTLE_PID=$!
echo "CPU Throttler PID: $THROTTLE_PID"

echo "Waiting for 1-hour pingpong run to complete..."
wait $RUN_PID
RUN_EXIT=$?

echo "1-Hour Ping-Pong process exited with code $RUN_EXIT"
kill $THROTTLE_PID 2>/dev/null || true

exit $RUN_EXIT
