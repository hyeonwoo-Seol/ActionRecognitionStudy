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
import warnings

# ## -----------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# ## -----------------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha 
        return input.clone() 

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = -ctx.alpha * grad_output 
        return grad_input, None 

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return GradientReversalFunction.apply(input, self.alpha)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight


class TemporalEmbedding(nn.Module):
    """
    (N, T, J, C_in) 입력을 받아서 시간 축(T)을 Conv1d로 압축하고,
    (N, T_new, J, C_out) 형태로 반환하는 모듈입니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # Conv1d: (Batch, Channel, Length) -> 여기서는 (N*J, C, T) 형태로 처리
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = RMSNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (N, T, J, C)
        N, T, J, C = x.shape
        
        # 1. Conv1d를 위해 (N*J, C, T) 형태로 변환
        x = x.permute(0, 2, 3, 1).contiguous().view(N * J, C, T)
        
        # 2. Conv1d 적용 (시간 축 압축)
        x = self.conv(x)
        
        # 3. 다시 (N, T_new, J, C_out) 형태로 복원
        # x shape: (N*J, C_out, T_new)
        C_out = x.shape[1]
        T_new = x.shape[2]
        x = x.view(N, J, C_out, T_new).permute(0, 3, 1, 2).contiguous()
        
        # 4. Norm & Act
        x = self.norm(x)
        x = self.act(x)
        return x
    

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        N, T, J, C = x.shape
        x_reshaped = x.view(N, T * J, C)
        attention_logits = self.attention_scorer(x_reshaped)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context_vector = (x_reshaped * attention_weights).sum(dim=1)
        return context_vector
    

class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_joints: int, max_frames: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # max_frames 크기만큼 미리 PE를 만들어둠
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_frames, 1, d_model))
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))
        trunc_normal_(self.temporal_pe, std=.02)
        trunc_normal_(self.spatial_pe, std=.02)

    def forward(self, x):
        # 입력 x의 T 길이에 맞춰서 슬라이싱 (유동적 T 지원)
        x = x + self.temporal_pe[:, :x.size(1), :, :]
        x = x + self.spatial_pe[:, :, :x.size(2), :]
        return self.dropout(x)


class ST_Transformer_Block(nn.Module):
    def __init__(self, in_features, out_features, num_joints, nhead=4, pos_encoder=None):
        super().__init__()
        self.out_features = out_features
        dim_feedforward = out_features * 2
        self.pos_encoder = pos_encoder
        
        self.input_proj = nn.Linear(in_features, out_features)
        self.norm_proj = RMSNorm(out_features)
        self.num_joints = num_joints

        # Spatial
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

        # Temporal
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, dropout=0.2
        )
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        self.norm_temporal = RMSNorm(out_features)
        
        # Residual
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                RMSNorm(out_features)
            )
        else:
            self.residual = nn.Identity()


    def forward(self, x):
        N, T, J, C_out = x.shape[0], x.shape[1], x.shape[2], self.out_features
        res = self.residual(x)
        x_proj = self.norm_proj(self.input_proj(x)) 
        if self.pos_encoder is not None:
            x_proj = self.pos_encoder(x_proj)

        x_spatial_in = x_proj.contiguous().view(N * T, J, C_out)
        
        # Spatial MSA
        x_norm1 = self.spatial_norm1(x_spatial_in)
        attn_output, _ = self.spatial_msa(x_norm1, x_norm1, x_norm1, average_attn_weights=False)
        x_msa_out = x_spatial_in + attn_output 
        
        # Spatial FFN
        x_norm2 = self.spatial_norm2(x_msa_out)
        ffn_output = self.spatial_ffn(x_norm2)
        x_spatial_out_NT = x_msa_out + ffn_output
        x_spatial_out = x_spatial_out_NT.view(N, T, J, C_out)

        # Temporal
        x_reshaped_temporal = x_spatial_out.permute(0, 2, 1, 3).contiguous().view(N * J, T, C_out)
        x_temporal_attn = self.temporal_transformer_encoder(x_reshaped_temporal)
        x_temporal_out = x_temporal_attn.view(N, J, T, C_out).permute(0, 2, 1, 3).contiguous()

        return x_temporal_out + res
    
    
class SlowFast_Transformer(nn.Module):
    def __init__(self,
                 num_joints=config.NUM_JOINTS,
                 num_coords=config.NUM_COORDS,
                 num_classes=config.NUM_CLASSES,
                 num_aux_classes=config.NUM_SUBJECTS,
                 alpha=1.0,
                 **kwargs
                 ):
        super().__init__()
        
        # 명시적 차원 정의 (Explicit Dimensions)
        # Fast Dims: [64, 64, 64]
        # Slow Dims: [64, 128, 256]
        
        # ==================================================================================
        # 1. Stem (Input Projection & Temporal Embedding)
        # ==================================================================================
        # Fast Stem: Conv1d(stride=2) -> T/2 압축, dim 64
        self.fast_embedding = TemporalEmbedding(
            in_channels=num_coords, out_channels=64, 
            kernel_size=3, stride=2, padding=1
        )

        # Slow Stem: Conv1d(stride=2) -> Conv1d(stride=2) -> T/4 압축, dim 64
        self.slow_embedding = nn.Sequential(
            TemporalEmbedding(in_channels=num_coords, out_channels=64, kernel_size=3, stride=2, padding=1),
            TemporalEmbedding(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        )

        # ==================================================================================
        # 2. Layer 1 (Block 1 + Lateral Connection)
        # ==================================================================================
        # [Lateral 1] Fast(64) <-> Slow(64) 교환
        # Fusion이 Block 입력 전에 일어난다고 가정 (기존 로직 유지)
        self.lat_layer1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.inv_lat_layer1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        
        # [Block 1]
        fast_pe_1 = SpatioTemporalPositionalEncoding(d_model=64, num_joints=num_joints, max_frames=config.MAX_FRAMES // 2)
        slow_pe_1 = SpatioTemporalPositionalEncoding(d_model=128, num_joints=num_joints, max_frames=config.MAX_FRAMES // 4)
        
        # Fast: 64 -> 64
        self.fast_block1 = ST_Transformer_Block(in_features=64, out_features=64, num_joints=num_joints, pos_encoder=fast_pe_1)
        # Slow: 64 -> 128 (Dim 확장)
        self.slow_block1 = ST_Transformer_Block(in_features=64, out_features=128, num_joints=num_joints, pos_encoder=slow_pe_1)


        # ==================================================================================
        # 3. Layer 2 (Block 2 + Lateral Connection)
        # ==================================================================================
        # [Lateral 2] Fast(64) <-> Slow(128) 교환
        self.lat_layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.inv_lat_layer2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))

        # [Block 2]
        fast_pe_2 = SpatioTemporalPositionalEncoding(d_model=64, num_joints=num_joints, max_frames=config.MAX_FRAMES // 2)
        slow_pe_2 = SpatioTemporalPositionalEncoding(d_model=256, num_joints=num_joints, max_frames=config.MAX_FRAMES // 4)

        # Fast: 64 -> 64
        self.fast_block2 = ST_Transformer_Block(in_features=64, out_features=64, num_joints=num_joints, pos_encoder=fast_pe_2)
        # Slow: 128 -> 256 (Dim 확장)
        self.slow_block2 = ST_Transformer_Block(in_features=128, out_features=256, num_joints=num_joints, pos_encoder=slow_pe_2)


        # ==================================================================================
        # 4. Heads (Pooling & Classification)
        # ==================================================================================
        final_fast_dim = 64
        final_slow_dim = 256
        combined_dim = final_fast_dim + final_slow_dim

        self.fast_pool = AttentionPooling(d_model=final_fast_dim)
        self.slow_pool = AttentionPooling(d_model=final_slow_dim)

        # Main Task
        self.action_head = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(combined_dim, num_classes)
        )
        
        # Aux Task
        self.grad_reversal = GradientReversalLayer(alpha=alpha) 
        self.aux_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            RMSNorm(256),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(256, num_aux_classes)
        )

    
    def forward(self, x_fast, x_slow):
        # x_fast, x_slow: (N, C, T, J) Input expected from Loader
        # Permute to (N, T, J, C) for Embeddings
        x_fast = x_fast.permute(0, 2, 3, 1).contiguous()
        x_slow = x_slow.permute(0, 2, 3, 1).contiguous()
        
        # -----------------------------
        # 1. Embeddings (Hierarchical Compression)
        # -----------------------------
        x_fast = self.fast_embedding(x_fast) # T -> T/2
        x_slow = self.slow_embedding(x_slow) # T -> T/4
        
        # -----------------------------
        # 2. Layer 1 Execution
        # -----------------------------
        # Lateral Fusion 1
        x_fast_perm = x_fast.permute(0, 3, 1, 2).contiguous() # (N, C, T, J)
        x_slow_perm = x_slow.permute(0, 3, 1, 2).contiguous() # (N, C, T, J)

        fast2slow_1 = self.lat_layer1(x_fast_perm).permute(0, 2, 3, 1).contiguous()
        slow2fast_1 = self.inv_lat_layer1(x_slow_perm).permute(0, 2, 3, 1).contiguous()

        x_slow_in_1 = x_slow + fast2slow_1
        x_fast_in_1 = x_fast + slow2fast_1

        # Block 1
        x_fast = self.fast_block1(x_fast_in_1) # Output: 64
        x_slow = self.slow_block1(x_slow_in_1) # Output: 128

        # -----------------------------
        # 3. Layer 2 Execution
        # -----------------------------
        # Lateral Fusion 2
        x_fast_perm = x_fast.permute(0, 3, 1, 2).contiguous()
        x_slow_perm = x_slow.permute(0, 3, 1, 2).contiguous()

        fast2slow_2 = self.lat_layer2(x_fast_perm).permute(0, 2, 3, 1).contiguous()
        slow2fast_2 = self.inv_lat_layer2(x_slow_perm).permute(0, 2, 3, 1).contiguous()

        x_slow_in_2 = x_slow + fast2slow_2
        x_fast_in_2 = x_fast + slow2fast_2

        # Block 2
        x_fast = self.fast_block2(x_fast_in_2) # Output: 64
        x_slow = self.slow_block2(x_slow_in_2) # Output: 256

        # -----------------------------
        # 4. Final Classification
        # -----------------------------
        summary_fast = self.fast_pool(x_fast)
        summary_slow = self.slow_pool(x_slow)

        combined_summary = torch.cat([summary_slow, summary_fast], dim=1)
        
        output_action = self.action_head(combined_summary)

        reversed_features = self.grad_reversal(combined_summary)
        output_aux = self.aux_classifier(reversed_features)

        return output_action, output_aux
