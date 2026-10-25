"""
POKER AI HUMAN ARENA: Model Manager (Singleton AI Loader)
=========================================================
Loads and caches the 4 champion checkpoints for real-time inference:
- Seat 1: Cycle 4 Pure PPO (1.2M hands, Standard Champion)
- Seat 2: Cycle 13 Pure PPO (2.9M hands, Novel Pool Champion)
- Seat 3: Cycle 10 + Adaptive MC (2.3M hands, High-Alpha Hybrid)
- Seat 4: Baseline Pure PPO (V10.22 100K hands, Pre-trained Baseline)
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1023.v1023_13_differential_forensics import PPOAgentWrapper
from v1013.run_v1013_replay_and_controller_validation import DeterministicTrueNFSP_Agent, ModelA

class ArenaModelManager:
    _instance = None

    def __init__(self):
        self.models = {}
        self.guards = {}
        self.wrappers = {}
        self._load_all()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ArenaModelManager()
        return cls._instance

    def _load_all(self):
        ckpts = {
            "cycle4": ("v1024/models/V1024_PPO_Cycle_4_1211670hands.pt", "Cycle 4 Pure PPO (1.2M)", "Standard Champion"),
            "cycle13": ("v1024/models/V1024_PPO_Cycle_13_2890003hands.pt", "Cycle 13 Pure PPO (2.9M)", "Novel Pool Contender"),
            "cycle10_mc": ("v1024/models/V1024_PPO_Cycle_10_2289023hands.pt", "Cycle 10 + Adaptive MC (2.3M)", "Hybrid Specialist"),
            "baseline": ("v1022/models/StrictAntiLimper_R2_100K.pt", "Baseline Pure PPO (100K)", "Pre-trained Baseline")
        }

        print("[ArenaModelManager] Loading champion checkpoints into Mac mini CPU memory...", flush=True)
        for key, (path, label, role) in ckpts.items():
            if not os.path.exists(path):
                print(f"[!] Warning: Checkpoint {path} not found!", flush=True)
                continue
            m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            m.eval()
            self.models[key] = m
            self.wrappers[key] = PPOAgentWrapper(m)
            print(f"  [+] Loaded {label} ({role}) from {path}", flush=True)

        bp_agent = DeterministicTrueNFSP_Agent()
        self.adaptive_guard = ModelA(bp_agent, sims=150, adaptive=True, adaptive_sims=1500, margin_threshold=0.025)
        print("[ArenaModelManager] All models ready for low-latency Mac mini serving!", flush=True)

    def select_action(self, key, st, cp, legal_6, folded_set, sim_seed, dec_idx, round_num):
        """
        AI 플레이어의 의사결정 수행:
        - key: 'cycle4', 'cycle13', 'cycle10_mc', 'baseline'
        - 반환: (선택 액션 ID 0~5, 추론 소요시간 ms, MC 개입 여부 bool)
        """
        t0 = time.perf_counter()
        wrapper = self.wrappers.get(key)
        if not wrapper:
            return 1, 0.1, False  # Fallback Check/Call

        mc_used = False
        if key == "cycle10_mc" and round_num > 0:
            act = self.adaptive_guard.act(st, cp, legal_6, folded_set, sim_seed, dec_idx)
            mc_used = True
        else:
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            act = wrapper.act(inf, legal_6)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return act, latency_ms, mc_used
