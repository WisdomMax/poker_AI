#!/bin/bash
python3 v1023/v1023_3_rng_isolated_audit.py > v1023/v1023_3_run.log 2>&1 &
EVAL_PID=$!
echo "Evaluation started with PID: $EVAL_PID"

python3 cpu_throttler.py $EVAL_PID 0.80 > v1023/throttler_v1023_3.log 2>&1 &
THROTTLE_PID=$!
echo "Throttler started with PID: $THROTTLE_PID"

wait $EVAL_PID
echo "Evaluation completed!"
