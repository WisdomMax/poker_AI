import os
import sys
import time
import signal

def throttle(pid, ratio=0.80, period=0.05):
    run_time = period * ratio
    pause_time = period * (1.0 - ratio)
    print(f"Throttling PID {pid} to {int(ratio*100)}% CPU (Run {run_time*1000:.1f}ms, Pause {pause_time*1000:.1f}ms)")
    
    while True:
        try:
            # Check if process is still alive
            os.kill(pid, 0)
        except OSError:
            print("Target process finished. Exiting throttler.")
            break
            
        time.sleep(run_time)
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            break
            
        time.sleep(pause_time)
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    target_pid = int(sys.argv[1])
    target_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
    throttle(target_pid, target_ratio)
