#!/bin/bash
# POKER AI V10.24 DIRECTIVE 015 Runner with strict 70% CPU Throttler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting DIRECTIVE 015: 20,000-Hand Qualification League"
echo "Target CPU Throttle: 70%"
echo "=========================================================="

python3 v1024/run_directive015_qualification.py > v1024/qualification_20000.log 2>&1 &
RUN_PID=$!
echo "DIRECTIVE 015 Process PID: $RUN_PID"

python3 cpu_throttler.py $RUN_PID 0.70 > v1024/throttler_directive015.log 2>&1 &
THROTTLE_PID=$!
echo "CPU Throttler PID: $THROTTLE_PID"

echo "Waiting for qualification run to complete..."
wait $RUN_PID
RUN_EXIT=$?

echo "Qualification process exited with code $RUN_EXIT"
kill $THROTTLE_PID 2>/dev/null || true

exit $RUN_EXIT
