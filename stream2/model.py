# model.py
# ## ------------------------------------------------------------
#
# ## ------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import config
import math



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

        # >> 1. 공간적 어텐션을 위한 Transformer 인코더이다.
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.2
        )
        self.spatial_transformer_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=1)
        self.norm_spatial = RMSNorm(out_features)


        # >> 2. 시간적 어텐션을 위한 Transformer 인코더이다.
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.2
        )
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        self.norm_temporal = RMSNorm(out_features)
        

        # >> 3. 입력과 출력의 채널 수가 다를 경우를 위한 잔차 연결이다.
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
        
        # >> 잔차 연결을 위해 초기 입력을 저장한다.
        res = self.residual(x)
        
        x_processed = self.gcn(x)

        x_gcn = self.norm_gcn(x_processed)
        
        

        # >> 1. 공간 Transformer 어텐션이다.
        # >> 각 프레임(T) 내에서 관절(J)들 간의 관계를 학습한다.
        # >> (N, T, J, C) -> (N*T, J, C) 형태로 변경하여 J를 시퀀스 길이로 취급한다.
        res_spatial = x_gcn
        x_reshaped_spatial = x_gcn.contiguous().view(N * T, J, C_out)
        x_spatial_attn = self.spatial_transformer_encoder(x_reshaped_spatial)
        x_spatial_attn = self.norm_spatial(x_spatial_attn)


        # >> 원래 형태로 복원한다. (N*T, J, C) -> (N, T, J, C)
        x_spatial_out = x_spatial_attn.view(N, T, J, C_out)
        x_spatial_out = x_spatial_out + res_spatial # 공간 내 잔차 연결


        # >> 2. 시간 Transformer 어텐션이다.
        # >> 각 관절(J)의 시간(T)적 흐름에 따른 관계를 학습한다.
        # >> (N, T, J, C) -> (N*J, T, C) 형태로 변경하여 T를 시퀀스 길이로 취급한다.
        res_temporal = x_spatial_out
        x_reshaped_temporal = x_spatial_out.permute(0, 2, 1, 3).contiguous().view(N * J, T, C_out)
        x_temporal_attn = self.temporal_transformer_encoder(x_reshaped_temporal)
        x_temporal_attn = self.norm_temporal(x_temporal_attn)


        # >> 원래 형태로 복원한다. (N*J, T, C) -> (N, T, J, C)
        x_temporal_out = x_temporal_attn.view(N, J, T, C_out).permute(0, 2, 1, 3).contiguous()
        x_temporal_out = x_temporal_out + res_temporal # 시간 내 잔차 연결


        # >> 3. 최종 잔차 연결을 한다.
        x_final = x_temporal_out + res

        return x_final
    
# ## -------------------------------------------------------------------------
# GCN-Transformer 최종 모델
# 정적인 자세 정보와 동적인 움직임 정보를 결합하여 최종 분류를 수행한다.
# ## -------------------------------------------------------------------------
class GCNTransformerModel(nn.Module):
    def __init__(self,
                 num_joints = config.NUM_JOINTS,
                 num_coords = config.NUM_COORDS,
                 num_classes = config.NUM_CLASSES,
                 layer_dims = config.LAYER_DIMS,
                 ):
        super().__init__()

                
        # >> 입력 좌표를 layer_dims의 첫 번째 차원으로 변환한다.
        self.input_projection = nn.Linear(num_coords, layer_dims[0])
        

        # >> 시퀀스에 위치 정보를 더해주는 Positional Encoding이다.
        self.pos_encoder = PositionalEncoding(
            d_model=layer_dims[0],
            max_len=config.MAX_FRAMES * config.NUM_JOINTS
        )
        
        # >> 동적으로 transformer 블록을 생성한다.
        self.blocks = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            self.blocks.append(
                ST_Transformer_Block(
                    in_features=in_dim, 
                    out_features=out_dim, 
                    num_joints=num_joints
                )
            )

        # >> 출력 부분 처리
        final_dim = layer_dims[-1]

        # >> 시퀀스 전체 정보를 하나의 벡터로 요약하는 Attention Pooling
        self.attention_pool = AttentionPooling(d_model=final_dim)

        # >> 움직임 요약 벡터를 결합하기
        self.classification_head = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.GELU(),
            RMSNorm(128)
        )

        
        # >> 과적합 방지를 위한 Dropout이다.
        self.dropout = nn.Dropout(p=config.DROPOUT)
        # >> 최종 클래스를 분류하는 층이다.
        self.fc = nn.Linear(128, num_classes)

    

    def forward(self, motion_features):
        N, _, T, J = motion_features.shape
        C_proj = self.input_projection.out_features
        

        # >> 1. 움직임 특징을 처리한다.
        x = motion_features.permute(0, 2, 3, 1).contiguous()
        
        # >> 각 관절의 좌표 정보를 64차원 벡터로 임베딩한다.
        x = self.input_projection(x) # (N, T, J, 64)


        # >> 2. 위치 인코딩을 추가한다.
        x_flat = x.view(N, T * J, C_proj)
        x_flat_pe = self.pos_encoder(x_flat)

        # >> 원래 형태로 복원한다.
        x = x_flat_pe.view(N, T, J, C_proj)
        

        # >> 3. GCN-Transformer 블록을 통과시킨다.
        for block in self.blocks:
            x = block(x)

        # >> 4. 동적 움직임 요약
        motion_summary = self.attention_pool(x) # (N, final_dim)

        # >> 5. 최종 분류
        final_vector = self.classification_head(motion_summary) # (N, 128)
        final_vector = self.dropout(final_vector)
        output = self.fc(final_vector) # (N, num_classes)
        return output


# ## -------------------------------------------------------------------------
# SlowFast GCN-Transformer (신규 추가)
# GCN-Transformer 컴포넌트를 재사용하여 비대칭적인 2-스트림 모델을 구현한다.
# - Fast Pathway: 가벼운 네트워크로 높은 시간 해상도를 처리 (2프레임 간격)
# - Slow Pathway: 무거운 네트워크로 낮은 시간 해상도를 처리 (4프레임 간격)
# ## -------------------------------------------------------------------------
class SlowFast_GCNTransformer(nn.Module):
    def __init__(self,
                 num_joints=config.NUM_JOINTS,
                 num_coords=config.NUM_COORDS,
                 num_classes=config.NUM_CLASSES,
                 fast_dims=config.FAST_DIMS,  # 예: [32, 32, 64] (config.py에 추가 필요)
                 slow_dims=config.SLOW_DIMS,  # 예: [128, 128, 256] (config.py에 추가 필요)
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
        
        # >> Fast 경로의 (N, C_fast, T_fast, J) 특징을
        # >> Slow 경로의 (N, C_slow, T_slow, J) 특징으로 변환합니다.
        # >> 2D Conv를 사용하여 시간축(T)을 1/2로 줄이고(stride=2),
        # >> 채널(C)을 Fast에서 Slow에 맞게 늘려줍니다.
        self.lateral_connection = nn.Conv2d(
            in_channels=fast_dims[0],    # Fast 경로의 첫 번째 채널 수
            out_channels=slow_dims[0],   # Slow 경로의 첫 번째 채널 수
            kernel_size=(5, 1),      # (Temporal_Kernel=5, Spatial_Kernel=1)
            stride=(2, 1),           # (Temporal_Stride=2, Spatial_Stride=1)
            padding=(2, 0)           # (Temporal_Padding=2, Spatial_Padding=0)
        )

        # --- 4. Final Classification (최종 분류) ---
        
        # >> 각 경로의 최종 특징 차원
        final_fast_dim = fast_dims[-1]
        final_slow_dim = slow_dims[-1]

        # >> 각 경로에 대한 어텐션 풀링
        self.fast_pool = AttentionPooling(d_model=final_fast_dim)
        self.slow_pool = AttentionPooling(d_model=final_slow_dim)

        # >> 두 경로의 요약 벡터를 결합 (비대칭적 헤드)
        self.classification_head = nn.Sequential(
            # 두 벡터를 합친 차원 (fast + slow)
            nn.Linear(final_fast_dim + final_slow_dim, 256),
            nn.GELU(),
            RMSNorm(256)
        )
        
        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.fc = nn.Linear(256, num_classes)

    
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

        # --- 4. Lateral Connection (융합) ---
        # (N, T_fast, J, C_fast0) -> (N, C_fast0, T_fast, J)
        x_fast_for_lat = x_fast.permute(0, 3, 1, 2).contiguous()
        
        # [Conv2D] (N, C_fast0, T_fast, J) -> (N, C_slow0, T_slow, J)
        x_fused = self.lateral_connection(x_fast_for_lat)
        
        # (N, C_slow0, T_slow, J) -> (N, T_slow, J, C_slow0)
        x_fused = x_fused.permute(0, 2, 3, 1).contiguous()

        # [핵심 융합] Slow 경로에 Fast 경로의 정보를 더합니다.
        x_slow = x_slow + x_fused
        
        # --- 5. Slow 경로 PE 적용 (융합 후) ---
        x_slow_flat = x_slow.view(N, T_slow * J, self.slow_dims[0])
        x_slow_flat_pe = self.slow_pos_encoder(x_slow_flat)
        x_slow = x_slow_flat_pe.view(N, T_slow, J, self.slow_dims[0])

        # --- 6. GCN-Transformer 블록 병렬 처리 ---
        # (이 코드는 블록 수가 같다고 가정합니다)
        for i in range(len(self.fast_blocks)):
            x_fast = self.fast_blocks[i](x_fast)
            x_slow = self.slow_blocks[i](x_slow)
            
            # (참고: 더 복잡한 모델은 여기서 2차, 3차 융합을 수행합니다)

        # --- 7. 최종 풀링 및 분류 ---
        summary_fast = self.fast_pool(x_fast) # (N, C_fast_final)
        summary_slow = self.slow_pool(x_slow) # (N, C_slow_final)

        # [최종 결합]
        combined_summary = torch.cat([summary_slow, summary_fast], dim=1)
        
        final_vector = self.classification_head(combined_summary)
        final_vector = self.dropout(final_vector)
        output = self.fc(final_vector)
        
        return output
