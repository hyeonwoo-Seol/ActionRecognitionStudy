# model.py
# Shift-GCN + Transformer 3layer

import torch
import torch.nn as nn
from mamba_ssm import Mamba
import numpy as np
import config
import math

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
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
    if num_joints == 50:
        base_num_joints = 25
    else: # 기존 25개 관절 모델과의 호환성 유지
        base_num_joints = num_joints
        
    # 관절의 부모-자식 관계 정의 (부모 -> 자식)
    parent_child_pairs = [
        (20, 1), (1, 0), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]

    base_adj_root = np.eye(base_num_joints)
    base_adj_close = np.zeros((base_num_joints, base_num_joints))
    base_adj_far = np.zeros((base_num_joints, base_num_joints))

    for parent, child in parent_child_pairs:
        base_adj_close[parent, child] = 1
        base_adj_far[child, parent] = 1

    if num_joints == 50:
        # 50x50 블록 대각 행렬 생성
        adj_root = np.kron(np.eye(2), base_adj_root) # np.kron은 블록 행렬을 쉽게 만듬
        adj_close = np.kron(np.eye(2), base_adj_close)
        adj_far = np.kron(np.eye(2), base_adj_far)
    else:
        adj_root, adj_close, adj_far = base_adj_root, base_adj_close, base_adj_far
        
    decompositions = [
        torch.from_numpy(adj_root).float(),
        torch.from_numpy(adj_close).float(),
        torch.from_numpy(adj_far).float()
    ]
    return torch.stack(decompositions)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 입력 특징(d_model)을 받아 1개의 어텐션 점수로 변환하는 선형 레이어
        self.attention_scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (N, T, J, C)
        N, T, J, C = x.shape

        # 1. 어텐션 점수 계산을 위해 (N, T*J, C) 형태로 변경
        x_reshaped = x.view(N, T * J, C)

        # 2. 각 위치(T*J개)의 중요도를 계산
        # (N, T*J, C) -> (N, T*J, 1)
        attention_logits = self.attention_scorer(x_reshaped)

        # 3. Softmax를 적용하여 합이 1인 어텐션 가중치 생성
        # (N, T*J, 1)
        attention_weights = torch.softmax(attention_logits, dim=1)

        # 4. 가중치와 원래 특징을 곱하여 가중 합(weighted sum) 계산
        # (N, T*J, C) * (N, T*J, 1) -> (N, T*J, C)
        # .sum(dim=1)을 통해 (N, C) 형태의 최종 요약 벡터 생성
        context_vector = (x_reshaped * attention_weights).sum(dim=1)

        return context_vector

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.MAX_FRAMES):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (N, T, C) 형태의 텐서에 위치 정보를 더해줍니다.
        """
        x = x + self.pe[:, :x.size(1), :].unsqueeze(2)
        return self.dropout(x)

class StandardTransformerBlock(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, dim_feedforward=256):
        super().__init__()
        # 1. 공간적 특징을 위한 Shift-GCN
        self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)
        self.norm_gcn = RMSNorm(out_features)

        # 2. 표준 Transformer 인코더
        # 공간과 시간 차원을 합친 전체 시퀀스에 대해 어텐션을 수행
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm_transformer = RMSNorm(out_features)
        
        # 입력과 출력의 채널 수가 다를 경우를 위한 잔차 연결
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                RMSNorm(out_features)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x shape: (N, T, J, C)
        N, T, J, C_in = x.shape
        C_out = self.gcn.weight.shape[-1]
        
        res = self.residual(x)
        
        # --- 1. 공간 그래프 컨볼루션 ---
        x_gcn = self.gcn(x)
        x_gcn = self.norm_gcn(x_gcn)

        # --- 2. 표준 Transformer 어텐션 ---
        # (N, T, J, C) -> (N, T*J, C) 형태로 펼쳐서 하나의 시퀀스로 만듦
        x_flattened = x_gcn.contiguous().view(N, T * J, C_out)
        
        # Transformer 인코더는 (N, SeqLen, C) 형태의 입력을 받음
        x_attn = self.transformer_encoder(x_flattened)
        x_attn = self.norm_transformer(x_attn)
        
        # 원래 형태로 복원: (N, T*J, C) -> (N, T, J, C)
        x_unflattened = x_attn.view(N, T, J, C_out)
        
        # --- 3. 최종 잔차 연결 ---
        x_final = x_unflattened + res
        
        return x_final

class ST_Transformer_Block(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, dim_feedforward=256):
        super().__init__()
        # 1. 공간적 특징을 위한 Shift-GCN (기존과 동일)
        self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)
        self.norm_gcn = RMSNorm(out_features)

        # 2. 공간적 어텐션을 위한 Transformer 인코더
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.spatial_transformer_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=1)
        self.norm_spatial = RMSNorm(out_features)

        # 3. 시간적 어텐션을 위한 Transformer 인코더
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        self.norm_temporal = RMSNorm(out_features)
        
        # 입력과 출력의 채널 수가 다를 경우를 위한 잔차 연결
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                RMSNorm(out_features)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x shape: (N, T, J, C)
        N, T, J, C_out = x.shape[0], x.shape[1], x.shape[2], self.gcn.weight.shape[-1]
        
        res_main = self.residual(x)
        
        # --- 1. 공간 그래프 컨볼루션 ---
        x_gcn = self.gcn(x)
        x_gcn = self.norm_gcn(x_gcn)

        # --- 2. 공간 Transformer 어텐션 ---
        # 각 프레임(T) 내에서 관절(J)들 간의 관계를 학습
        # (N, T, J, C) -> (N*T, J, C) 형태로 변경하여 J를 시퀀스 길이로 취급
        res_spatial = x_gcn
        x_reshaped_spatial = x_gcn.contiguous().view(N * T, J, C_out)
        x_spatial_attn = self.spatial_transformer_encoder(x_reshaped_spatial)
        x_spatial_attn = self.norm_spatial(x_spatial_attn)
        # 원래 형태로 복원: (N*T, J, C) -> (N, T, J, C)
        x_spatial_out = x_spatial_attn.view(N, T, J, C_out)
        x_spatial_out = x_spatial_out + res_spatial # 공간 내 잔차 연결

        # --- 3. 시간 Transformer 어텐션 ---
        # 각 관절(J)의 시간(T)적 흐름에 따른 관계를 학습
        # (N, T, J, C) -> (N*J, T, C) 형태로 변경하여 T를 시퀀스 길이로 취급
        res_temporal = x_spatial_out
        x_reshaped_temporal = x_spatial_out.permute(0, 2, 1, 3).contiguous().view(N * J, T, C_out)
        x_temporal_attn = self.temporal_transformer_encoder(x_reshaped_temporal)
        x_temporal_attn = self.norm_temporal(x_temporal_attn)
        # 원래 형태로 복원: (N*J, T, C) -> (N, T, J, C)
        x_temporal_out = x_temporal_attn.view(N, J, T, C_out).permute(0, 2, 1, 3).contiguous()
        x_temporal_out = x_temporal_out + res_temporal # 시간 내 잔차 연결
        
        # --- 4. 최종 잔차 연결 ---
        x_final = x_temporal_out + res_main
        
        return x_final

class GCNTransformerModel(nn.Module):
    def __init__(self, num_joints=50, num_coords=config.NUM_COORDS, num_classes=60):
        super().__init__()

        self.pose_encoder = nn.Sequential(
            nn.Linear(num_joints * 3, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        self.input_projection = nn.Linear(num_coords, 64)
        
        self.pos_encoder = PositionalEncoding(
            d_model=64, 
            max_len=config.MAX_FRAMES * config.NUM_JOINTS
        )
        
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(in_features=64, out_features=64, num_joints=num_joints),
            StandardTransformerBlock(in_features=64, out_features=128, num_joints=num_joints),
            # StandardTransformerBlock(in_features=128, out_features=256, num_joints=num_joints)
        ])

        self.attention_pool = AttentionPooling(d_model=128)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            RMSNorm(128)
        )
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, motion_features, first_frame_coords):
        N, _, T, J = motion_features.shape
        C_proj = 64 # self.input_projection의 출력 차원
        
        flat_pose = first_frame_coords.view(N, -1)
        pose_vector = self.pose_encoder(flat_pose)
        
        x = motion_features.permute(0, 2, 3, 1).contiguous()
        x = self.input_projection(x) # (N, T, J, 64)

        x_flat = x.view(N, T * J, C_proj)
        x_flat_pe = self.pos_encoder(x_flat)
        x = x_flat_pe.view(N, T, J, C_proj) # 다시 원래 형태로 복원
        
        x = self.blocks[0](x) # (64 -> 64)
        x = self.blocks[1](x) # (64 -> 128)
        # x = self.blocks[2](x)
        
        motion_summary = self.attention_pool(x)
        combined_summary = torch.cat([motion_summary, pose_vector], dim=-1)
        
        final_vector = self.fusion_layer(combined_summary)
        final_vector = self.dropout(final_vector)
        output = self.fc(final_vector)
        return output
class MambaBlock(nn.Module):
    # GCN 없이 Mamba와 잔차 연결만 포함한 블록
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.ln = RMSNorm(d_model)
        self.residual = nn.Identity() # 입력과 출력의 차원이 같으므로 Identity 사용

    def forward(self, x):
        # x shape: (N, T, J, C)
        N, T, J, C = x.shape
        
        res = self.residual(x)
        
        # Mamba 처리를 위해 shape 변경: (N, T, J, C) -> (N*J, T, C)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(N * J, T, -1)
        x_mamba = self.mamba(x_reshaped)
        x_mamba = self.ln(x_mamba)
        
        # 원래 형태로 복원
        x_out = x_mamba.view(N, J, T, -1).permute(0, 2, 1, 3).contiguous()
        
        # 잔차 연결
        x_out = x_out + res
        
        return x_out

class ST_Mamba_Block(nn.Module):
    # GCN과 Mamba를 하나로 묶고 잔차 연결을 포함한 블록
    def __init__(self, in_features, out_features, num_joints):
        super().__init__()
        self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)

        self.norm_gcn = RMSNorm(out_features)

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
        x_gcn = self.gcn(x)

        x_gcn_norm = self.norm_gcn(x_gcn)

        # 2. Mamba
        x_reshaped = x_gcn_norm.permute(0, 2, 1, 3).contiguous().view(N * J, T, -1)
        x_mamba = self.mamba(x_reshaped)
        x_mamba = self.ln(x_mamba)

        # 3. 원래 형태로 복원
        x_out = x_mamba.view(N, J, T, -1).permute(0, 2, 1, 3).contiguous()

        # 4. 잔차 연결
        x_out = x_out + res

        return x_out

class GCNMambaModel(nn.Module):
    def __init__(self, num_joints=25, num_coords=config.NUM_COORDS, num_classes=60):
        super().__init__()

        # 첫 번째 프레임의 좌표를 처리할 인코더
        self.pose_encoder = nn.Sequential(
            nn.Linear(num_joints * 3, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        self.blocks = nn.ModuleList([
            ST_Mamba_Block(in_features=num_coords, out_features=64, num_joints=num_joints),
            ST_Mamba_Block(in_features=64, out_features=128, num_joints=num_joints),
        ])

        self.attention_pool = AttentionPooling(d_model=128)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            RMSNorm(128)
        )
        
        self.dropout = nn.Dropout(p=0.7)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, motion_features, first_frame_coords):
        N = motion_features.shape[0]
        flat_pose = first_frame_coords.view(N, -1)
        pose_vector = self.pose_encoder(flat_pose)
        
        
        # permute 후 (N, T, 50, C)가 되어 모델에 정상적으로 입력됨
        x = motion_features.permute(0, 2, 3, 1).contiguous()

        for i, block in enumerate(self.blocks):
            x = block(x)
            

        motion_summary = self.attention_pool(x) # shape: (N, 128)
        
        combined_summary = torch.cat([motion_summary, pose_vector], dim=-1) # shape: (N, 512)
        


        final_vector = self.fusion_layer(combined_summary)
        final_vector = self.dropout(final_vector)
        output = self.fc(final_vector)
        return output
