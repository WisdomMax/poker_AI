"""
POKER AI V4.1 - 7. 하드웨어 학습 속도 최적화 벤치마크 (CPU vs MPS)
- Mac Apple Silicon 환경에서 CPU와 MPS(GPU)의 실제 초당 업데이트 / 게임 수 측정
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v4.v41_nfsp import QNetwork, encode_kuhn_state


def benchmark_device(device_name: str, num_batches: int = 1000, batch_size: int = 128):
    if device_name == "mps" and not torch.backends.mps.is_available():
        return None, "MPS not available"

    device = torch.device(device_name)
    model = QNetwork(7, 2, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 가상 데이터 생성
    x = torch.randn(batch_size, 7, device=device)
    y = torch.randn(batch_size, 2, device=device)

    # 워밍업
    for _ in range(50):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    if device_name == "mps":
        torch.mps.synchronize()

    start_time = time.time()
    for _ in range(num_batches):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    if device_name == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - start_time
    updates_per_sec = num_batches / elapsed
    return updates_per_sec, f"{elapsed:.4f}s"


def run_hardware_benchmark():
    print("=" * 65)
    print("⚡ [V4.1 STEP 7] Mac Apple Silicon 하드웨어 연산 벤치마크")
    print("=" * 65)

    cpu_ups, cpu_time = benchmark_device("cpu", num_batches=1500, batch_size=128)
    print(f"  - CPU 훈련 속도: {cpu_ups:.1f} updates/sec (소요시간: {cpu_time})")

    mps_ups, mps_time = benchmark_device("mps", num_batches=1500, batch_size=128)
    if mps_ups:
        print(f"  - MPS (Metal GPU) 속도: {mps_ups:.1f} updates/sec (소요시간: {mps_time})")
        if cpu_ups > mps_ups:
            print(f"  👉 분석: 소형 MLP(7->64->2)에서는 CPU-GPU 메모리 전송 오버헤드로 인해 CPU가 {cpu_ups/mps_ups:.1f}배 더 빠름.")
            print("  👉 채택: CPU 기반 초고속 단일 파이프라인 적용.")
        else:
            print(f"  👉 채택: MPS 기반 가속 적용.")
    else:
        print(f"  - MPS: {mps_time}")

    print("=" * 65)


if __name__ == "__main__":
    run_hardware_benchmark()
