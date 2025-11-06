# model.py
# ## ------------------------------------------------------------
#
# ## ------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import config
import math
from torch.autograd import Function


# ## ---------------------------------------------------------------
# NTU RGB+D 뼈대 구조를 Shift-GCN을 위한 3개의 인접 행렬로 분해한다.
# Root: 자기 자신과 연결
# Close: 몸의 중심에서 바깥으로 연결
# Far: 밖에서 몸의 중심으로 연결
# ## ----------------------------------------------------------------
def get_ntu_shift_decompositions(num_joints):
    if num_joints == 50:
        base_num_joints = 25
    else: # 기존 25개 관절 모델과의 호환성 유지
        base_num_joints = num_joints


    # >> 관절의 부모-자식 관계 정의 (부모 -> 자식)
    parent_child_pairs = [
        (20, 1), (1, 0), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]


    # >> 3가지 타입의 인접 행렬을 0으로 초기화한다.
    base_adj_root = np.eye(base_num_joints)
    base_adj_close = np.zeros((base_num_joints, base_num_joints))
    base_adj_far = np.zeros((base_num_joints, base_num_joints))


    # >> 정의된 관계에 따라 close와 far 행렬의 값을 1로 설정한다.
    for parent, child in parent_child_pairs:
        base_adj_close[parent, child] = 1
        base_adj_far[child, parent] = 1


    # >> 50개 관절의 경우, 25 x 25 행렬을 확장해서 50 x 50 블록 대각 행렬을 생성한다.
    if num_joints == 50:
        # >> np.kron은 두 행렬의 외적을 계산한다.
        adj_root = np.kron(np.eye(2), base_adj_root)
        adj_close = np.kron(np.eye(2), base_adj_close)
        adj_far = np.kron(np.eye(2), base_adj_far)
    else:
        adj_root, adj_close, adj_far = base_adj_root, base_adj_close, base_adj_far


    # >> numpy 배열을 torch 텐서로 변환하여 리스트에 저장한다.    
    decompositions = [
        torch.from_numpy(adj_root).float(),
        torch.from_numpy(adj_close).float(),
        torch.from_numpy(adj_far).float()
    ]

    # >> 3개의 인접 행렬을 하나의 텐서로 쌓아서 반환한다.
    return torch.stack(decompositions)




# ## -----------------------------------------------------
# Shift-GCN
# Skeleton 데이터의 공간적 관계를 학습하기 위해
# 3가지로 분해된 인접 행렬을 사용하여 Graph Convolution을 수행한다.
# ## ------------------------------------------------------
class ShiftGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_joints):
        super().__init__()
        # >> Shift-GCN용 인접 행렬 3개를 buffer로 등록한다.
        self.register_buffer('adj_matrices', get_ntu_shift_decompositions(num_joints))
        

        # >> 학습 가능한 가중치를 선언한다.
        self.weight = nn.Parameter(torch.FloatTensor(3, in_features, out_features))
        
        # >> Xavier 균등 분포로 가중치를 초기화한다.
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x):
        # x shape: (N, T, J, C_in)
        
        # >> x를 3개의 shift 타입에 적용하기 위해 (3, N, T, J, C_in) 형태로 확장한다.
        x_expanded = x.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        
        # >> 각 shift 타입에 맞는 인접행렬과 Feature를 곱한다. (그래프 shift 연산)
        # (3, J, J) x (3, N, T, J, C_in) -> (3, N, T, J, C_in)
        support = torch.einsum('sjj,sntjc->sntjc', self.adj_matrices, x_expanded)
        
        # >> 각 shift 타입에 맞는 가중치를 곱한다.
        # (3, N, T, J, C_in) x (3, C_in, C_out) -> (3, N, T, J, C_out)
        output = torch.einsum('sntjc,scd->sntjd', support, self.weight)
        
        # >> 3개의 결과를 합친다.
        output = output.sum(dim=0) # (N, T, J, C_out)
        return output




# ## --------------------------------------------------------
# RMSNorm
# 기존의 LayerNorm의 계산 복잡도를 줄이고 성능은 유지한다.
# Simba 논문에서 참고했다.
# ## --------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        # >> 감마(gamma)에 해당하는 학습 가능한 가중치이다.
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # >> 입력 x의 마지막 차원에 대해 RMS(Root Mean Square)를 계산하고 정규화한다.
        # x * (1 / sqrt(mean(x^2) + eps)) * weight
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight
    




# ## ------------------------------------------------------------
# Attention Pooling
# 시퀀스 데이터의 각 요소에 어텐션 가중치를 부여한다.
# 중요한 정보만 압축하여 하나의 context vector를 생성한다.
# ## ------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # >> 입력 특징(d_model)을 받아 1개의 어텐션 점수로 변환하는 선형 레이어
        self.attention_scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (N, T, J, C)
        N, T, J, C = x.shape

        # >> 1. 어텐션 점수 계산을 위해 (N, T*J, C) 형태로 변경한다.
        x_reshaped = x.view(N, T * J, C)

        # >> 2. 각 위치(T*J개)의 중요도를 계산한다.
        # (N, T*J, C) -> (N, T*J, 1)
        attention_logits = self.attention_scorer(x_reshaped)

        # >> 3. Softmax를 적용하여 합이 1인 어텐션 가중치 생성한다.
        # (N, T*J, 1)
        attention_weights = torch.softmax(attention_logits, dim=1)

        # >> 4. 가중치와 원래 특징을 곱하여 가중 합(weighted sum) 계산한다.
        # (N, T*J, C) * (N, T*J, 1) -> (N, T*J, C)
        # .sum(dim=1)을 통해 (N, C) 형태의 최종 요약 벡터 생성
        context_vector = (x_reshaped * attention_weights).sum(dim=1)

        return context_vector
    

# ## ------------------------------------------------------------
# Positional Encoding
# Transformer는 순서 정보를 다룰 수 없으므로,
# sin, cos 함수를 이용해 각 토큰의 위치 정보를 더해준다.
# ## ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = config.MAX_FRAMES):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # >> 위치 인덱스이다.
        position = torch.arange(max_len).unsqueeze(1)
        
        # >> 위치 인코딩 계산을 위한 분모 항이다.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # >> (1, max_len, d_model) 크기의 위치 인코딩 행렬이다.
        pe = torch.zeros(1, max_len, d_model)

        # >> 짝수 인덱스에 sin 함수를 적용한다.
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # >> 홀수 인덱스에 cos 함수를 적용한다.
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # >> pe를 학습되지 않는 버퍼로 등록한다.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # >> 입력 x에 위치 인코딩을 더한다.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)







# ## -------------------------------------------------------------------------
# Spatial-Temporal Transformer Block
# Shift-GCN 이후, 공간(Spatial) 축과 시간(Temporal) 축에 대해
# 순차적으로 Transformer 어텐션을 적용하여 시공간 특징을 분리하여 학습한다.
# ## -------------------------------------------------------------------------
class ST_Transformer_Block(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, dim_feedforward=256):
        super().__init__()
        
        self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)

        self.norm_gcn = RMSNorm(out_features)

        # >> Evo-ViT: 공간적 토큰 선택을 위한 [SPATIAL_CLS] 토큰 추가
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, out_features))
        self.spatial_keep_rate = config.SPATIAL_KEEP_RATE
        self.num_joints = num_joints

        # >> Slow Fast
        self.spatial_norm1 = RMSNorm(out_features)
        self.spatial_msa = nn.MultiheadAttention(
            embed_dim=out_features, num_heads=nhead, dropout=0.2, batch_first=True
        )
        self.spatial_norm2 = RMSNorm(out_features)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(out_features, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim_feedforward, out_features),
            nn.Dropout(0.2)
        )
        
        # >> Fast Path - 별도 모듈 x


        # >> 2. 시간적 어텐션을 위한 Transformer 인코더이다.
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.2
        )
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        self.norm_temporal = RMSNorm(out_features)
        

        # >> 3.  잔차 연결이다.
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                RMSNorm(out_features)
            )
        else:
            self.residual = nn.Identity()


    def forward(self, x, prev_spatial_attn_scores=None):
        # x shape: (N, T, J, C)
        N, T, J, C_out = x.shape[0], x.shape[1], x.shape[2], self.gcn.weight.shape[-1]
        
        # >> 잔차 연결을 위해 초기 입력을 저장한다.
        res = self.residual(x)

        x_gcn = self.norm_gcn(self.gcn(x)) # (N, T, J, C_out)

        res_spatial = x_gcn
        
        # >> (N, T, J, C) -> (N*T, J, C)
        x_spatial_in = x_gcn.contiguous().view(N * T, J, C_out)
        NT, J, C = x_spatial_in.shape        


        # --- 1. Evo-ViT 공간적 토큰 선택 ---
        k = int(J * self.spatial_keep_rate) # 유지할 중요 관절(informative) 수

        if prev_spatial_attn_scores is None:
            # >> 이전 어텐션 맵이 없음 (첫 번째 블록) -> 모든 토큰을 Slow 경로로
            indices_to_keep = torch.arange(J, device=x.device).unsqueeze(0).repeat(NT, 1)
            indices_to_drop = torch.empty(NT, 0, dtype=torch.long, device=x.device)
            # >> 점수도 없으므로 가중치 없는 평균 집계 사용
            placeholder_scores_norm = None 
        else:
            # >> 이전 어텐션 맵을 기반으로 토큰 선택
            scores = prev_spatial_attn_scores # (NT, J)
            all_indices = torch.argsort(scores, dim=1, descending=True)
            indices_to_keep = all_indices[:, :k] # (NT, k)
            indices_to_drop = all_indices[:, k:] # (NT, J-k)
            
            # >> 집계(Aggregation)를 위한 가중치 계산
            placeholder_scores = torch.gather(scores, 1, indices_to_drop)
            placeholder_scores_norm = torch.softmax(placeholder_scores, dim=1).unsqueeze(-1) # (NT, J-k, 1)


        # --- 2. Slow-Fast 토큰 분리 ---
        
        # >> 2-1. 중요 토큰 (Informative tokens)
        # (NT, J, C) -> (NT, k, C)
        x_inf = torch.gather(x_spatial_in, 1, indices_to_keep.unsqueeze(-1).expand(-1, -1, C))
        
        # >> 2-2. 자리 표시자 토큰 (Placeholder tokens)
        # (NT, J, C) -> (NT, J-k, C)
        x_ph = torch.gather(x_spatial_in, 1, indices_to_drop.unsqueeze(-1).expand(-1, -1, C))

        # >> 2-3. 대표 토큰 (Representative token) - 집계
        if prev_spatial_attn_scores is None or k == J:
            x_rep = torch.empty(NT, 0, C, device=x.device) # (첫 블록이거나 모두 keep이면 rep 없음)
        else:
            # 가중 합 (Weighted Sum)
            x_rep = (x_ph * placeholder_scores_norm).sum(dim=1, keepdim=True) # (NT, 1, C)

        # >> 2-4. [SPATIAL_CLS] 토큰 준비
        cls_token = self.spatial_cls_token.repeat(NT, 1, 1) # (NT, 1, C)
        
        # >> 2-5. Slow 경로 입력 준비: [CLS, Informative, Representative]
        x_slow_in = torch.cat([cls_token, x_inf, x_rep], dim=1) # (NT, 1+k+1 또는 1+k, C)

        # --- 3. Slow 경로 연산 (MSA + FFN) ---
        
        # >> 3-1. MSA
        x_norm1 = self.spatial_norm1(x_slow_in)
        # average_attn_weights=False (기본값) -> (NT, nhead, L, S) 반환
        attn_output, new_attn_weights = self.spatial_msa(
            x_norm1, x_norm1, x_norm1, average_attn_weights=False
        )
        x_slow_msa_out = x_slow_in + attn_output # (잔차 연결 1)
        
        # >> 3-2. FFN
        x_norm2 = self.spatial_norm2(x_slow_msa_out)
        ffn_output = self.spatial_ffn(x_norm2)
        x_slow_final = x_slow_msa_out + ffn_output # (잔차 연결 2)

        # --- 4. 다음 레이어를 위한 어텐션 맵 추출 ---
        # (NT, nhead, 1+k+..., 1+k+...)
        # CLS 토큰(idx 0)이 다른 토큰(idx 1:)에 부여한 어텐션
        next_cls_attn_to_others = new_attn_weights[:, :, 0, 1:].mean(dim=1) # (NT, k+1 또는 k)

        next_spatial_attn = torch.zeros(NT, J, device=x.device, dtype=x.dtype)
        
        # 중요 토큰(informative)에 대한 어텐션 점수
        cls_attn_to_inf = next_cls_attn_to_others[:, :indices_to_keep.shape[1]]
        next_spatial_attn.scatter_(1, indices_to_keep, cls_attn_to_inf)
        
        if prev_spatial_attn_scores is not None and k != J:
            # 대표 토큰(representative)에 대한 어텐션 점수
            cls_attn_to_rep = next_cls_attn_to_others[:, k:] # (NT, 1)
            # 대표 점수를 모든 자리 표시자(placeholder) 토큰에 전파
            next_spatial_attn.scatter_(1, indices_to_drop, cls_attn_to_rep.expand(-1, J-k))


        # --- 5. Fast 경로 연산 (확장)  ---
        
        if prev_spatial_attn_scores is None or k == J:
            x_ph_out = x_ph # (첫 블록이거나 모두 keep이면 fast update 없음)
        else:
            # >> 5-1. Slow 경로에서 대표 토큰의 '잔차'만 추출
            msa_res_rep = attn_output[:, 1+k:, :]
            ffn_res_rep = ffn_output[:, 1+k:, :]
            x_rep_res_combined = msa_res_rep + ffn_res_rep # (NT, 1, C)

            # >> 5-2. 잔차를 모든 자리 표시자 토큰에 더함
            x_ph_out = x_ph + x_rep_res_combined.expand(-1, J-k, -1) # (NT, J-k, C)
            
        # --- 6. 전체 텐서 재구성 (Scatter) ---
        num_inf_tokens = indices_to_keep.shape[1]
        x_inf_out = x_slow_final[:, 1:1 + num_inf_tokens, :]
        
        x_spatial_out = torch.zeros_like(x_spatial_in)
        x_spatial_out.scatter_(1, indices_to_keep.unsqueeze(-1).expand(-1, -1, C), x_inf_out)
        x_spatial_out.scatter_(1, indices_to_drop.unsqueeze(-1).expand(-1, -1, C), x_ph_out)
        
        # >> (N*T, J, C) -> (N, T, J, C)
        x_spatial_out = x_spatial_out.view(N, T, J, C_out)
        x_spatial_out = x_spatial_out + res_spatial # GCN 이후 잔차 연결


        # >> 7. 시간 Transformer 어텐션
        res_temporal = x_spatial_out
        x_reshaped_temporal = x_spatial_out.permute(0, 2, 1, 3).contiguous().view(N * J, T, C_out)
        x_temporal_attn = self.temporal_transformer_encoder(x_reshaped_temporal)
        x_temporal_attn = self.norm_temporal(x_temporal_attn)


        # >> 원래 형태로 복원
        x_temporal_out = x_temporal_attn.view(N, J, T, C_out).permute(0, 2, 1, 3).contiguous()
        x_temporal_out = x_temporal_out + res_temporal # 시간 내 잔차 연결


        # >> 8. 최종 잔차 연결
        x_final = x_temporal_out + res

        # >> (최종 결과, 다음 블록용 어텐션 맵) 반환
        return x_final, next_spatial_attn
    
    

# ## -------------------------------------------------------------------------
# SlowFast GCN-Transformer
# GCN-Transformer 컴포넌트를 재사용하여 비대칭적인 2-스트림 모델을 구현한다.
# - Fast Pathway: 가벼운 네트워크로 높은 시간 해상도를 처리 (2프레임 간격)
# - Slow Pathway: 무거운 네트워크로 낮은 시간 해상도를 처리 (4프레임 간격)
# ## -------------------------------------------------------------------------
class SlowFast_GCNTransformer(nn.Module):
    def __init__(self,
                 num_joints=config.NUM_JOINTS,
                 num_coords=config.NUM_COORDS,
                 num_classes=config.NUM_CLASSES,
                 fast_dims=config.FAST_DIMS,
                 slow_dims=config.SLOW_DIMS,
                 num_subjects=config.NUM_SUBJECTS
                 ):
        super().__init__()
        
        # >> config.py에서 정의된 채널 수를 가져옵니다.
        self.fast_dims = fast_dims
        self.slow_dims = slow_dims
        
        if len(self.fast_dims) != len(self.slow_dims):
            raise ValueError("Fast와 Slow 경로의 블록 수가 동일해야 합니다.")

        # --- 1. Fast Pathway (가벼움) ---
        
        # >> Fast 경로 입력 프로젝션
        self.fast_input_projection = nn.Linear(num_coords, fast_dims[0])
        # >> Fast 경로 위치 인코딩 (T_fast = max_frames)
        self.fast_pos_encoder = PositionalEncoding(
            d_model=fast_dims[0],
            max_len=config.MAX_FRAMES * num_joints
        )
        # >> Fast 경로 GCN-Transformer 블록 리스트
        self.fast_blocks = nn.ModuleList()
        for i in range(len(fast_dims) - 1):
            self.fast_blocks.append(
                ST_Transformer_Block(
                    in_features=fast_dims[i],
                    out_features=fast_dims[i+1],
                    num_joints=num_joints
                )
            )

        # --- 2. Slow Pathway (무거움) ---
        
        # >> Slow 경로 입력 프로젝션
        self.slow_input_projection = nn.Linear(num_coords, slow_dims[0])
        # >> Slow 경로 위치 인코딩 (T_slow = max_frames / 2)
        self.slow_pos_encoder = PositionalEncoding(
            d_model=slow_dims[0],
            max_len=(config.MAX_FRAMES // 2) * num_joints # Fast의 절반
        )
        # >> Slow 경로 GCN-Transformer 블록 리스트
        self.slow_blocks = nn.ModuleList()
        for i in range(len(slow_dims) - 1):
            self.slow_blocks.append(
                ST_Transformer_Block(
                    in_features=slow_dims[i],
                    out_features=slow_dims[i+1],
                    num_joints=num_joints
                )
            )

        # --- 3. Lateral Connection (Fast -> Slow 융합) ---
        
        # >> nn.ModuleList로 변경하여 다단계 융합을 지원합니다.
        self.lateral_connections = nn.ModuleList()

        for i in range(len(self.fast_dims)-1):
            # i=0: in=32, out=128 (Block 0 입력 전 1차 융합)
            # i=1: in=32, out=128 (Block 1 입력 전 2차 융합)
            self.lateral_connections.append(
                nn.Conv2d(
                    in_channels=fast_dims[i],    # Fast 경로의 [i]번째 채널 수
                    out_channels=slow_dims[i],   # Slow 경로의 [i]번째 채널 수
                    kernel_size=(5, 1),      # (Temporal_Kernel=5, Spatial_Kernel=1)
                    stride=(2, 1),           # (Temporal_Stride=2, Spatial_Stride=1)
                    padding=(2, 0)           # (Temporal_Padding=2, Spatial_Padding=0)
                )
            )

        # --- 4. Final Classification (최종 분류) ---
        
        # >> 각 경로의 최종 특징 차원
        final_fast_dim = fast_dims[-1]
        final_slow_dim = slow_dims[-1]

        # >> 각 경로에 대한 어텐션 풀링
        self.fast_pool = AttentionPooling(d_model=final_fast_dim)
        self.slow_pool = AttentionPooling(d_model=final_slow_dim)

       # >> 각 헤드는 과적합 방지를 위해 config의 드롭아웃을 포함합니다
        self.slow_head = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(final_slow_dim, num_classes)
        )
        self.fast_head = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(final_fast_dim, num_classes)
        )

        
        
        
    
    def forward(self, x_fast, x_slow):
        # x_fast shape: (N, C, T_fast, J)
        # x_slow shape: (N, C, T_slow, J)
        
        N, _, T_fast, J = x_fast.shape
        _, _, T_slow, _ = x_slow.shape

        # --- 1. 입력 Permute (N, C, T, J) -> (N, T, J, C) ---
        x_fast = x_fast.permute(0, 2, 3, 1).contiguous()
        x_slow = x_slow.permute(0, 2, 3, 1).contiguous()
        
        # --- 2. Fast 경로 처리 (PE 추가) ---
        x_fast = self.fast_input_projection(x_fast) # (N, T_fast, J, C_fast0)
        
        x_fast_flat = x_fast.view(N, T_fast * J, self.fast_dims[0])
        x_fast_flat_pe = self.fast_pos_encoder(x_fast_flat)
        x_fast = x_fast_flat_pe.view(N, T_fast, J, self.fast_dims[0])

        # --- 3. Slow 경로 처리 (PE 없음 - 융합 후 적용) ---
        x_slow = self.slow_input_projection(x_slow) # (N, T_slow, J, C_slow0)

        x_slow_flat = x_slow.view(N, T_slow * J, self.slow_dims[0])
        x_slow_flat_pe = self.slow_pos_encoder(x_slow_flat)
        x_slow = x_slow_flat_pe.view(N, T_slow, J, self.slow_dims[0])
        
        # --- 4. GCN-Transformer 블록 및 다단계 융합 처리 ---
        fast_attn_map = None # (N*T_fast, J)
        slow_attn_map = None # (N*T_slow, J)

        # len(self.fast_blocks)는 2 (i=0, i=1)
        for i in range(len(self.fast_blocks)):
            
            # --- A. 융합 수행 (Lateral Connection) ---
            # (N, T, J, C_fast) -> (N, C_fast, T, J)
            x_fast_for_lat = x_fast.permute(0, 3, 1, 2).contiguous()
            
            # [Conv2D] (N, C_fast_i, T_fast, J) -> (N, C_slow_i, T_slow, J)
            # i=0일 때: lateral_connections[0] (1차 융합)
            # i=1일 때: lateral_connections[1] (2차 융합)
            x_fused = self.lateral_connections[i](x_fast_for_lat)
            
            # (N, C_slow_i, T_slow, J) -> (N, T_slow, J, C_slow_i)
            x_fused = x_fused.permute(0, 2, 3, 1).contiguous()

            # [핵심 융합] Slow 경로에 Fast 경로의 정보를 더합니다.
            x_slow = x_slow + x_fused
            
            
            # --- B. GCN-Transformer 블록 실행 ---
            x_fast, fast_attn_map = self.fast_blocks[i](x_fast, fast_attn_map)
            x_slow, slow_attn_map = self.slow_blocks[i](x_slow, slow_attn_map)
            
        # --- 5. 최종 풀링 및 분류 ---
        summary_fast = self.fast_pool(x_fast) # (N, C_fast_final)
        summary_slow = self.slow_pool(x_slow) # (N, C_slow_final)

        
        # >> 각 전문가 헤드가 독립적으로 예측을 수행합니다.
        output_slow = self.slow_head(summary_slow) # (N, num_classes)
        output_fast = self.fast_head(summary_fast) # (N, num_classes)
        output_action = output_slow + output_fast

        return output_action
    
