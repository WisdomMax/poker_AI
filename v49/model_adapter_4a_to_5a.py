"""
POKER AI V4.9 - 4-Action 모델 호환 어댑터 (Adapter from FCPA to FCHPA)
- 목적: 253차원 입력 및 4개 행동(FCPA)을 출력하는 기존 V4.8.3 모델을
  265차원 입력 및 5개 행동(FCHPA)을 요구하는 V4.9 환경에서 무결하게 실행.
- 행동 변환:
  - Fold (0) -> Fold (0)
  - Call (1) -> Call (1)
  - Pot (2) -> Pot (2)
  - All-in (3) -> All-in (3)
  - Half-Pot (4) -> 선택 불가 (-inf 마스킹)
- 상태 텐서 변환:
  - 265차원 FCHPA 상태 -> 253차원 FCPA 상태 매핑
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import pyspiel

def map_fchpa_to_fcpa_tensor(v_fchpa: np.ndarray) -> np.ndarray:
    """
    265차원 fchpa 상태 텐서를 253차원 fcpa 상태 텐서로 변환:
    - 0..215: 카드, 포지션, 초기 시퀀스 (동일)
    - 223..261: fchpa 끝단의 스트리트별 팟/투자금 -> fcpa의 215..253에 정확히 매핑
    """
    if isinstance(v_fchpa, torch.Tensor):
        is_torch = True
        device = v_fchpa.device
        v_np = v_fchpa.cpu().numpy()
    else:
        is_torch = False
        v_np = v_fchpa
        
    if v_np.ndim == 1:
        v_fcpa = np.zeros(253, dtype=np.float32)
        v_fcpa[:215] = v_np[:215]
        v_fcpa[215:253] = v_np[223:261]
    else:
        # 배치 입력
        b = v_np.shape[0]
        v_fcpa = np.zeros((b, 253), dtype=np.float32)
        v_fcpa[:, :215] = v_np[:, :215]
        v_fcpa[:, 215:253] = v_np[:, 223:261]
        
    if is_torch:
        return torch.tensor(v_fcpa, dtype=torch.float32, device=device)
    return v_fcpa

class FourActionToFiveActionAdapter(nn.Module):
    """
    기존 4-Action HoldemActorCritic 모델을 감싸는 5-Action 호환 어댑터:
    - 입력: x (shape [B, 265]), mask (shape [B, 5])
    - 출력: logits_5a (shape [B, 5]), value (shape [B])
    - Half-Pot(행동 4)의 로짓은 -1e9로 마스킹되어 절대 선택되지 않음.
    """
    def __init__(self, model_4a: nn.Module):
        super().__init__()
        self.model_4a = model_4a
        
    def forward(self, x_5a: torch.Tensor, mask_5a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_5a.device
        b = x_5a.shape[0]
        
        # 1. 265차원 -> 253차원 텐서 변환
        x_4a = map_fchpa_to_fcpa_tensor(x_5a)
        
        # 2. 5차원 마스크에서 FCPA 4개 추출
        mask_4a = mask_5a[:, :4].clone()
        
        # 3. 기존 4-Action 모델 순전파
        logits_4a, value = self.model_4a(x_4a, mask_4a)
        
        # 4. 5-Action 로짓으로 확장 (Half-Pot = -1e9 강제 마스킹)
        logits_5a = torch.full((b, 5), -1e9, dtype=torch.float32, device=device)
        logits_5a[:, :4] = logits_4a
        
        # 유효하지 않은 행동 추가 마스킹
        logits_5a = torch.where(mask_5a, logits_5a, torch.tensor(-1e9, device=device))
        
        return logits_5a, value

class FourActionAdapterBot:
    """5-Action 리그에서 플레이할 수 있는 BaseBot 인터페이스 래퍼"""
    def __init__(self, adapter: FourActionToFiveActionAdapter, device: torch.device = torch.device("cpu"), name: str = "4A_Adapter_Bot"):
        self.adapter = adapter
        self.device = device
        self.name = name
        
    def act(self, state: pyspiel.State, cur_player: int, legal_acts: List[int]) -> int:
        vec = state.information_state_tensor(cur_player)
        x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.zeros((1, 5), dtype=torch.bool, device=self.device)
        for a in legal_acts: m[0, a] = True
        
        self.adapter.eval()
        with torch.no_grad():
            logits_5a, _ = self.adapter(x, m)
            probs = torch.softmax(logits_5a, dim=-1).squeeze(0).cpu().numpy()
            
        prs = [probs[a] for a in legal_acts]
        p_sum = sum(prs)
        prs = np.array(prs) / p_sum if p_sum > 0 else np.ones(len(legal_acts)) / len(legal_acts)
        
        chosen = int(np.random.choice(legal_acts, p=prs))
        assert chosen != 4, f"Adapter selected HalfPot (4)! Probs: {probs}"
        return chosen
