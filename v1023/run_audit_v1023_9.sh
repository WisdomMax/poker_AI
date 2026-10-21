#!/bin/bash
python3 v1023/v1023_9_hybrid_forensics_and_2x2.py > v1023/v1023_9_run.log 2>&1 &
EVAL_PID=$!
echo "Evaluation started with PID: $EVAL_PID"

python3 cpu_throttler.py $EVAL_PID 0.80 > v1023/throttler_v1023_9.log 2>&1 &
THROTTLE_PID=$!
echo "Throttler started with PID: $THROTTLE_PID"

wait $EVAL_PID
echo "Evaluation completed!"
