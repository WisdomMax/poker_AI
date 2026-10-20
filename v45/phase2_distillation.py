"""
POKER AI V4.5 - PHASE 2: CFR+ Policy Distillation (신경망 정책 증류)
- Teacher: CFR+ 2,000 Iterations 최적 내쉬 전략 (936개 정보집합)
- Student Network: 30차원 입력 -> 128 -> 128 -> 3차원 행동 확률
- 손실 함수 비교: Cross-Entropy (KL Divergence) vs MSE
- Legal Action Masking 적용
- 증류된 Student의 OpenSpiel NashConv 및 Exploitability (bb/100) 정밀 측정
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

# 단위 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from units import format_metrics, nashconv_to_exploitability

# 1. Student 신경망 모델 정의
class LeducStudentNet(nn.Module):
    def __init__(self, input_dim: int = 30, hidden_dim: int = 128, num_actions: int = 3):
        super(LeducStudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x: torch.Tensor, legal_masks: torch.Tensor = None) -> torch.Tensor:
        logits = self.net(x)
        if legal_masks is not None:
            # 불법 행동 마스킹
            logits = logits.masked_fill(~legal_masks, -1e9)
        return logits
    
    def get_action_probs(self, x: torch.Tensor, legal_masks: torch.Tensor = None) -> torch.Tensor:
        logits = self.forward(x, legal_masks)
        return F.softmax(logits, dim=-1)

# 2. OpenSpiel Policy 래퍼 클래스
class StudentOpenSpielPolicy(policy.Policy):
    def __init__(self, game, model: LeducStudentNet, device: torch.device):
        super(StudentOpenSpielPolicy, self).__init__(game, list(range(game.num_players())))
        self.model = model
        self.device = device
        self.model.eval()
        
    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            cur_player = state.current_player()
        else:
            cur_player = player_id
            
        legal_actions = state.legal_actions(cur_player)
        if not legal_actions:
            return {}
            
        info_tensor = state.information_state_tensor(cur_player)
        x = torch.tensor(info_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        mask = torch.zeros((1, 3), dtype=torch.bool, device=self.device)
        for a in legal_actions:
            mask[0, a] = True
            
        with torch.no_grad():
            probs = self.model.get_action_probs(x, mask).squeeze(0).cpu().numpy()
            
        # 딕셔너리로 변환 및 정규화 보장
        prob_dict = {a: float(probs[a]) for a in legal_actions}
        total = sum(prob_dict.values())
        if total > 0:
            for a in prob_dict:
                prob_dict[a] /= total
        else:
            for a in prob_dict:
                prob_dict[a] = 1.0 / len(legal_actions)
        return prob_dict

# 3. 데이터셋 구축 함수
def build_distillation_dataset(game, teacher_policy_path: str):
    with open(teacher_policy_path, "rb") as f:
        teacher_dict = pickle.load(f)
        
    print(f"📖 Teacher 정책 로드 완료 ({len(teacher_dict)}개 정보집합)")
    
    states_dict = {}
    
    def traverse(st):
        if st.is_terminal():
            return
        if st.is_chance_node():
            for a, _ in st.chance_outcomes():
                nxt = st.clone()
                nxt.apply_action(a)
                traverse(nxt)
            return
            
        cp = st.current_player()
        info_str = st.information_state_string(cp)
        if info_str in teacher_dict and info_str not in states_dict:
            info_tensor = np.array(st.information_state_tensor(cp), dtype=np.float32)
            legal_actions = st.legal_actions(cp)
            
            # Target probability vector (크기 3)
            target_p = np.zeros(3, dtype=np.float32)
            for a, p in teacher_dict[info_str].items():
                target_p[a] = float(p)
                
            # Legal mask
            legal_mask = np.zeros(3, dtype=bool)
            for a in legal_actions:
                legal_mask[a] = True
                
            states_dict[info_str] = (info_tensor, legal_mask, target_p)
            
        for a in st.legal_actions(cp):
            nxt = st.clone()
            nxt.apply_action(a)
            traverse(nxt)
            
    traverse(game.new_initial_state())
    print(f"✅ 일치하는 고유 상태 추출 완료: {len(states_dict)}개")
    
    X = np.stack([v[0] for v in states_dict.values()])
    M = np.stack([v[1] for v in states_dict.values()])
    Y = np.stack([v[2] for v in states_dict.values()])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(M, dtype=torch.bool), torch.tensor(Y, dtype=torch.float32)

# 4. 증류 학습 실행 루틴
def train_distillation(game, X, M, Y, loss_type: str = "ce", epochs: int = 1500, lr: float = 1e-3, device=torch.device("cpu")):
    model = LeducStudentNet(input_dim=30, hidden_dim=128, num_actions=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X = X.to(device)
    M = M.to(device)
    Y = Y.to(device)
    
    t0 = time.time()
    best_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X, M)
        
        if loss_type == "ce":
            # Cross-Entropy / KL Divergence: - \sum Y * log(Softmax(logits))
            log_probs = F.log_softmax(logits, dim=-1)
            # 불법 액션이 0인 곳은 log_probs가 -inf일 수 있으므로 Y=0이면 0으로 처리
            loss = -(Y * log_probs).sum(dim=-1).mean()
        elif loss_type == "mse":
            probs = F.softmax(logits, dim=-1)
            loss = F.mse_loss(probs, Y)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            
        if epoch % 300 == 0 or epoch == epochs:
            # 중간 평가 (NashConv)
            osp_policy = StudentOpenSpielPolicy(game, model, device)
            nc = float(exploitability.nash_conv(game, osp_policy))
            print(f"  [{loss_type.upper():>3}] Epoch {epoch:>4}/{epochs} | Loss: {loss.item():.6f} | NashConv: {nc:.6f} (Exp: {nc*50:.2f} bb/100)")
            
    elapsed = time.time() - t0
    final_policy = StudentOpenSpielPolicy(game, model, device)
    final_nc = float(exploitability.nash_conv(game, final_policy))
    
    return model, final_nc, elapsed

def run_phase2():
    print("=" * 80)
    print("🚀 [V4.5 PHASE 2] CFR+ Policy Distillation (신경망 정책 증류 연구)")
    print("=" * 80)
    
    game = pyspiel.load_game("leduc_poker")
    teacher_path = "v44/models/leduc_cfr_plus_2000.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 실행 디바이스: {device}")
    
    # 데이터셋 구축
    X, M, Y = build_distillation_dataset(game, teacher_path)
    print(f"📊 학습 텐서 크기: X={X.shape}, M={M.shape}, Y={Y.shape}")
    
    # 1. Cross-Entropy Loss 증류 학습
    print("\n[실험 1: Cross-Entropy (KL Divergence) 손실 함수 증류]")
    model_ce, nc_ce, time_ce = train_distillation(game, X, M, Y, loss_type="ce", epochs=2000, lr=2e-3, device=device)
    
    # 2. MSE Loss 증류 학습
    print("\n[실험 2: MSE 손실 함수 증류]")
    model_mse, nc_mse, time_mse = train_distillation(game, X, M, Y, loss_type="mse", epochs=2000, lr=2e-3, device=device)
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("📊 [CFR+ 증류 기법별 성능 비교표]")
    print("=" * 80)
    print(f"{'증류 손실 기법':<20} | {'최종 NashConv':<15} | {'Exploitability (bb/100)':<25} | {'학습 시간':<10}")
    print("-" * 80)
    exp_ce = nashconv_to_exploitability(nc_ce)
    exp_mse = nashconv_to_exploitability(nc_mse)
    print(f"{'Cross-Entropy':<20} | {nc_ce:<15.6f} | {exp_ce['exploitability_bb_per_100']:<25.2f} | {time_ce:<10.2f}s")
    print(f"{'MSE':<20} | {nc_mse:<15.6f} | {exp_mse['exploitability_bb_per_100']:<25.2f} | {time_mse:<10.2f}s")
    print("=" * 80)
    
    # 최적 모델 선정 및 저장
    best_model = model_ce if nc_ce <= nc_mse else model_mse
    best_nc = min(nc_ce, nc_mse)
    best_name = "Cross-Entropy" if nc_ce <= nc_mse else "MSE"
    
    save_path = "v45/models/leduc_student_best.pt"
    torch.save(best_model.state_dict(), save_path)
    print(f"\n🎯 최적 Student 선정: {best_name} (NashConv: {best_nc:.6f})")
    print(f"💾 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    run_phase2()
