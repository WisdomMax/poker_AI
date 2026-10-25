#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/.."

echo "=========================================================="
echo "LAUNCHING POKER AI V10.24 2-HOUR AUTONOMOUS RL PING-PONG"
echo "=========================================================="

# 1. 메인 학습 프로세스 백그라운드 기동
python3 v1024/train_v1024_2hour_autonomous_loop.py > v1024/train_2hour.log 2>&1 &
MAIN_PID=$!
echo "[+] Main Training Loop Started (PID: $MAIN_PID)"

# 2. 1초 대기 후 프로세스 생존 확인
sleep 1
if ! ps -p $MAIN_PID > /dev/null; then
    echo "[-] Main process failed to start! Log:"
    cat v1024/train_2hour.log
    exit 1
fi

# 3. CPU 70% Throttler 백그라운드 기동
python3 cpu_throttler.py $MAIN_PID 0.70 > v1024/throttler_70_2hour.log 2>&1 &
THROTTLE_PID=$!
echo "[+] CPU 70% Throttler Attached (PID: $THROTTLE_PID) to Target $MAIN_PID"

echo "=========================================================="
echo "2-HOUR RL PIPELINE IS NOW RUNNING IN BACKGROUND!"
echo "Target Duration: 2.0 Hours (~19:18 KST)"
echo "Logs: v1024/train_2hour.log | v1024/throttler_70_2hour.log"
echo "=========================================================="
