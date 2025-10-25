# model.py
# ## ------------------------------------------------------------
# 1. Shift-GCN
# 2-1. RMSNormalization
# 2-2. Attention Pooling
# 3. Standard-Transformer
# 4. Spatial-Temporal Transformer
# 5. Shift-GCN + Standard-Transformer
# 6. Shift-GCN + ST-Transformer
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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.MAX_FRAMES):
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




# ## --------------------------------------------------------------------
# Standard Transformer Block
# Shift-GCN으로 공간적 특징을 추출한 후,
# 시간과 공간을 하나로 합친 시퀀스에 대해 표준 Transformer 어텐션을 적용한다.
# ## --------------------------------------------------------------------
class StandardTransformerBlock(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, dim_feedforward=256, use_gcn=True):
        super().__init__()
        # >> 1. Shift-GCN 사용 여부를 결정한다.
        self.use_gcn = use_gcn
        if self.use_gcn:
            self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)
        
        
        # >> 2. 표준 Transformer 인코더이다.
        # >> 공간과 시간 차원을 합친 전체 시퀀스에 대해 어텐션을 수행한다.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm_gcn = RMSNorm(out_features)
        self.norm_transformer = RMSNorm(out_features)


        # >> 입력과 출력의 채널 수가 다를 경우를 위한 잔차 연결 코드이다.
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
        

        # >> 잔차 연결을 위해 초기 입력을 저장한다.
        res = self.residual(x)


        # >> 1. use_gcn 플래그에 따라 GCN을 실행하거나 건너뛴다.
        if self.use_gcn:
            x_processed = self.gcn(x) 
        else:
            x_processed = res # GCN을 사용하지 않으면 입력을 그대로 사용한다.

        x_gcn = self.norm_gcn(x_processed)
        


        # >> 2. 표준 Transformer 어텐션이다.
        # >> (N, T, J, C) -> (N, T*J, C) 형태로 펼쳐서 하나의 시퀀스로 만든다.
        x_flattened = x_gcn.contiguous().view(N, T * J, C_out)
        

        # >> Transformer 인코더는 (N, SeqLen, C) 형태의 입력을 받는다.
        x_attn = self.transformer_encoder(x_flattened)
        x_attn = self.norm_transformer(x_attn)
        

        # >> 원래 형태로 복원한다. (N, T*J, C) -> (N, T, J, C)
        x_unflattened = x_attn.view(N, T, J, C_out)
        

        # >> 3. 최종 잔차 연결을 한다.
        x_final = x_unflattened + res
        
        return x_final




# ## -------------------------------------------------------------------------
# Spatial-Temporal Transformer Block
# Shift-GCN 이후, 공간(Spatial) 축과 시간(Temporal) 축에 대해
# 순차적으로 Transformer 어텐션을 적용하여 시공간 특징을 분리하여 학습한다.
# ## -------------------------------------------------------------------------
class ST_Transformer_Block(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, dim_feedforward=256, use_gcn=True):
        super().__init__()
        # >> 1. Shift-GCN 사용 여부를 결정한다.
        self.use_gcn = use_gcn
        if self.use_gcn:
            self.gcn = ShiftGraphConvolution(in_features, out_features, num_joints=num_joints)

        self.norm_gcn = RMSNorm(out_features)

        # >> 2. 공간적 어텐션을 위한 Transformer 인코더이다.
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.spatial_transformer_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=1)
        self.norm_spatial = RMSNorm(out_features)


        # >> 3. 시간적 어텐션을 위한 Transformer 인코더이다.
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        self.norm_temporal = RMSNorm(out_features)
        

        # >> 입력과 출력의 채널 수가 다를 경우를 위한 잔차 연결이다.
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
        
        if self.use_gcn:
            x_processed = self.gcn(x)
        else:
            x_processed = res

        x_gcn = self.norm_gcn(x_processed)
        
        

        # >> 2. 공간 Transformer 어텐션이다.
        # >> 각 프레임(T) 내에서 관절(J)들 간의 관계를 학습한다.
        # >> (N, T, J, C) -> (N*T, J, C) 형태로 변경하여 J를 시퀀스 길이로 취급한다.
        res_spatial = x_gcn
        x_reshaped_spatial = x_gcn.contiguous().view(N * T, J, C_out)
        x_spatial_attn = self.spatial_transformer_encoder(x_reshaped_spatial)
        x_spatial_attn = self.norm_spatial(x_spatial_attn)
        

        # >> 원래 형태로 복원한다. (N*T, J, C) -> (N, T, J, C)
        x_spatial_out = x_spatial_attn.view(N, T, J, C_out)
        x_spatial_out = x_spatial_out + res_spatial # 공간 내 잔차 연결


        # >> 3. 시간 Transformer 어텐션이다.
        # >> 각 관절(J)의 시간(T)적 흐름에 따른 관계를 학습한다.
        # >> (N, T, J, C) -> (N*J, T, C) 형태로 변경하여 T를 시퀀스 길이로 취급한다.
        res_temporal = x_spatial_out
        x_reshaped_temporal = x_spatial_out.permute(0, 2, 1, 3).contiguous().view(N * J, T, C_out)
        x_temporal_attn = self.temporal_transformer_encoder(x_reshaped_temporal)
        x_temporal_attn = self.norm_temporal(x_temporal_attn)
        

        # >> 원래 형태로 복원한다. (N*J, T, C) -> (N, T, J, C)
        x_temporal_out = x_temporal_attn.view(N, J, T, C_out).permute(0, 2, 1, 3).contiguous()
        x_temporal_out = x_temporal_out + res_temporal # 시간 내 잔차 연결
        

        # >> 4. 최종 잔차 연결을 한다.
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
                 block_type ='standard',
                 layer_dims = [64, 64, 128],
                 use_gcn=True
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

        # >> layer_dims 리스트를 기반으로 반복문을 실행하여 블록을 쌓는다.
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            
            if block_type == 'standard':
                block = StandardTransformerBlock(
                    in_features=in_dim, 
                    out_features=out_dim, 
                    num_joints=num_joints,
                    use_gcn=use_gcn
                )
            elif block_type == 'st':
                block = ST_Transformer_Block(
                    in_features=in_dim, 
                    out_features=out_dim, 
                    num_joints=num_joints,
                    use_gcn=use_gcn
                )
            else:
                raise ValueError(f"Unknown block_type: {block_type}")
            
            self.blocks.append(block)

        # >> 출력 부분 처리
        final_dim = layer_dims[-1]

        # >> 시퀀스 전체 정보를 하나의 벡터로 요약하는 Attention Pooling이다.
        self.attention_pool = AttentionPooling(d_model=final_dim)
        
        

        # >> 움직임 요약 벡터와 자세 벡터(128)를 결합하는 층이다.
        self.fusion_layer = nn.Sequential(
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
        

                

        # >> 2. 움직임 특징을 처리한다.
        x = motion_features.permute(0, 2, 3, 1).contiguous()
        
        # >> 각 관절의 좌표 정보를 64차원 벡터로 임베딩한다.
        x = self.input_projection(x) # (N, T, J, 64)


        # >> 3. 위치 인코딩을 추가한다.
        x_flat = x.view(N, T * J, C_proj)
        x_flat_pe = self.pos_encoder(x_flat)

        # >> 원래 형태로 복원한다.
        x = x_flat_pe.view(N, T, J, C_proj)
        

        # >> 4. GCN-Transformer 블록을 통과시킨다.
        for block in self.blocks:
            x = block(x)
        

        # >> 5. 동적 움직임을 요약한다.
        motion_summary = self.attention_pool(x)
        
        
        
        

        # >> 7. 최종 분류를 수행한다.
        final_vector = self.fusion_layer(motion_summary)
        final_vector = self.dropout(final_vector)
        output = self.fc(final_vector)
        return output
