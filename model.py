# model.py
# Shift-GCN + Mamba
# Layer Norm -> RMS Norm

import torch
import torch.nn as nn
from mamba_ssm import Mamba
import numpy as np
# import config

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        # 감마(gamma)에 해당하는 학습 가능한 가중치
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # 입력 x의 마지막 차원에 대해 RMS(Root Mean Square)를 계산하고 정규화
        # x * (1 / sqrt(mean(x^2) + eps)) * weight
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight

def get_ntu_shift_decompositions(num_joints):
    """
    NTU RGB+D 뼈대 구조를 Shift-GCN을 위한 3개의 인접 행렬로 분해합니다.
    1. Root: 자기 자신과의 연결 (self-loop)
    2. Close: 몸의 중심에서 말단으로 향하는 연결
    3. Far: 말단에서 몸의 중심으로 향하는 연결
    """
        
    # 관절의 부모-자식 관계 정의 (부모 -> 자식)
    # 척추(20)를 중심으로 정의
    parent_child_pairs = [
        (20, 1), (1, 0), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]

    adj_root = np.eye(num_joints)
    adj_close = np.zeros((num_joints, num_joints))
    adj_far = np.zeros((num_joints, num_joints))

    for parent, child in parent_child_pairs:
        adj_close[parent, child] = 1 # 중심 -> 말단
        adj_far[child, parent] = 1   # 말단 -> 중심

    # PyTorch 텐서로 변환하여 스택
    decompositions = [
        torch.from_numpy(adj_root).float(),
        torch.from_numpy(adj_close).float(),
        torch.from_numpy(adj_far).float()
    ]
    return torch.stack(decompositions)


class ShiftGraphConvolution(nn.Module):
    # Shift-GCN 레이어
    def __init__(self, in_features, out_features, num_joints):
        super().__init__()
        # Shift-GCN용 인접 행렬 3개 (root, close, far)를 buffer로 등록
        self.register_buffer('adj_matrices', get_ntu_shift_decompositions(num_joints))
        
        # 가중치: 3(shift 타입) x C_in x C_out
        self.weight = nn.Parameter(torch.FloatTensor(3, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x shape: (N, T, J, C_in)
        
        # x를 (3, N, T, J, C_in) 형태로 확장
        x_expanded = x.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        
        # 각 shift 타입에 맞는 인접행렬과 피처를 곱함 (그래프 shift 연산)
        # (3, J, J) x (3, N, T, J, C_in) -> (3, N, T, J, C_in)
        support = torch.einsum('sjj,sntjc->sntjc', self.adj_matrices, x_expanded)
        
        # 각 shift 타입에 맞는 가중치를 곱함
        # (3, N, T, J, C_in) x (3, C_in, C_out) -> (3, N, T, J, C_out)
        output = torch.einsum('sntjc,scd->sntjd', support, self.weight)
        
        # 3개의 결과를 합침
        output = output.sum(dim=0) # (N, T, J, C_out)
        return output


class ST_Mamba_Block(nn.Module):
    # GCN과 Mamba를 하나로 묶고 잔차 연결을 포함한 블록
    def __init__(self, in_features, out_features, num_joints):
        super().__init__()
        self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)
        
        self.mamba = Mamba(d_model=out_features, d_state=16, d_conv=4, expand=2)
        self.ln = RMSNorm(out_features)
        
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                RMSNorm(out_features)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x): # adj 인자 제거
        # x shape: (N, T, J, C)
        N, T, J, C = x.shape
        
        res = self.residual(x)
        
        # 1. Shift-GCN
        x_gcn = self.gcn(x) # adj 인자 없이 호출
        
        # 2. Mamba
        x_reshaped = x_gcn.permute(0, 2, 1, 3).contiguous().view(N * J, T, -1)
        x_mamba = self.mamba(x_reshaped)
        x_mamba = self.ln(x_mamba)
        
        # 3. 원래 형태로 복원
        x_out = x_mamba.view(N, J, T, -1).permute(0, 2, 1, 3).contiguous()
        
        # 4. 잔차 연결
        x_out = x_out + res
        
        return x_out


class GCNMambaModel(nn.Module):
    def __init__(self, num_joints=25, num_coords=6, num_classes=60):
        super().__init__()
        
        
        self.blocks = nn.ModuleList([
            ST_Mamba_Block(in_features=num_coords, out_features=64, num_joints=num_joints),
            ST_Mamba_Block(in_features=64, out_features=128, num_joints = num_joints),
            ST_Mamba_Block(in_features=128, out_features=256, num_joints = num_joints),
            ST_Mamba_Block(in_features=256, out_features=256, num_joints = num_joints)
        ])

        self.dropout = nn.Dropout(p=0.5)
       
        self.fc = nn.Linear(256, num_classes)

        
        
    def forward(self, x):
        # 입력 shape: (N, C, T, J)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, T, J, C)
        
        # ST-Mamba 블록 통과
        for block in self.blocks:
            x = block(x) # adj 인자 없이 호출
            
        x = x.mean(dim=[1, 2])
        x = self.dropout(x)
        
        output = self.fc(x)
        return output
