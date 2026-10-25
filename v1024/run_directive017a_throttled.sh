#!/bin/bash
# POKER AI V10.24 DIRECTIVE 017-A Runner (Strict CPU 70% Throttled)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Starting DIRECTIVE 017-A: Final Champion Re-Qualification"
echo "Target CPU Throttle: 70%"
echo "Start: $(date)"
echo "=========================================================="

python3 v1024/run_directive017a_qualification.py > v1024/directive017a_run.log 2>&1 &
RUN_PID=$!
echo "DIRECTIVE 017-A PID: $RUN_PID"

python3 cpu_throttler.py $RUN_PID 0.70 > v1024/throttler_directive017a.log 2>&1 &
THROTTLE_PID=$!
echo "CPU Throttler PID: $THROTTLE_PID"

echo "Waiting for DIRECTIVE 017-A run to complete..."
wait $RUN_PID
RUN_EXIT=$?

echo "DIRECTIVE 017-A process exited with code $RUN_EXIT"
kill $THROTTLE_PID 2>/dev/null || true

exit $RUN_EXIT
