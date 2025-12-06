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
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_frames, 1, d_model))
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))
        trunc_normal_(self.temporal_pe, std=.02)
        trunc_normal_(self.spatial_pe, std=.02)

    def forward(self, x):
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
                 fast_dims=config.FAST_DIMS,
                 slow_dims=config.SLOW_DIMS,
                 num_aux_classes=config.NUM_SUBJECTS, # [수정] 범용적인 보조 클래스 수
                 alpha=1.0
                 ):
        super().__init__()
        
        self.fast_dims = fast_dims
        self.slow_dims = slow_dims
        
        if len(self.fast_dims) != len(self.slow_dims):
            raise ValueError("Fast와 Slow 경로의 블록 수가 동일해야 합니다.")

        # --- Fast Pathway ---
        self.fast_input_projection = nn.Linear(num_coords, fast_dims[0])
        self.fast_blocks = nn.ModuleList()
        for i in range(len(fast_dims) - 1):
            fast_pe = SpatioTemporalPositionalEncoding(
                d_model=fast_dims[i+1], num_joints=num_joints, max_frames=config.MAX_FRAMES
            )
            self.fast_blocks.append(
                ST_Transformer_Block(
                    in_features=fast_dims[i], out_features=fast_dims[i+1],
                    num_joints=num_joints, pos_encoder=fast_pe
                )
            )

        # --- Slow Pathway ---
        self.slow_input_projection = nn.Linear(num_coords, slow_dims[0])
        self.slow_blocks = nn.ModuleList()
        for i in range(len(slow_dims) - 1):
            slow_pe = SpatioTemporalPositionalEncoding(
                d_model=slow_dims[i+1], num_joints=num_joints, max_frames=(config.MAX_FRAMES // 2)
            )
            self.slow_blocks.append(
                ST_Transformer_Block(
                    in_features=slow_dims[i], out_features=slow_dims[i+1],
                    num_joints=num_joints, pos_encoder=slow_pe
                )
            )

        # --- Lateral Connection ---
        self.lateral_connections = nn.ModuleList()
        for i in range(len(self.fast_dims)-1):
            self.lateral_connections.append(
                nn.Conv2d(
                    in_channels=fast_dims[i], out_channels=slow_dims[i],
                    kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)
                )
            )

        self.inv_lateral_connections = nn.ModuleList()
        for i in range(len(self.fast_dims)-1):
            self.inv_lateral_connections.append(
                nn.ConvTranspose2d(
                    in_channels=slow_dims[i], out_channels=fast_dims[i],
                    kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)
                )
            )

        # --- Final Classification ---
        final_fast_dim = fast_dims[-1]
        final_slow_dim = slow_dims[-1]
        combined_dim = final_fast_dim + final_slow_dim

        self.fast_pool = AttentionPooling(d_model=final_fast_dim)
        self.slow_pool = AttentionPooling(d_model=final_slow_dim)

        # [Head A] 행동 분류 (Main Task)
        self.action_head = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(combined_dim, num_classes)
        )
        
        # [Head B] 보조 분류 (Auxiliary Task via GRL)
        # X-Sub: Subject Classification / X-View: Camera Classification
        self.grad_reversal = GradientReversalLayer(alpha=alpha) 
        
        self.aux_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            RMSNorm(256),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(256, num_aux_classes) # [수정] 동적 클래스 개수 사용
        )

    
    def forward(self, x_fast, x_slow):
        N, _, T_fast, J = x_fast.shape
        _, _, T_slow, _ = x_slow.shape

        x_fast = x_fast.permute(0, 2, 3, 1).contiguous()
        x_slow = x_slow.permute(0, 2, 3, 1).contiguous()
        
        x_fast = self.fast_input_projection(x_fast)
        x_slow = self.slow_input_projection(x_slow)
        
        for i in range(len(self.fast_blocks)):
            # Bi-directional Lateral Fusion
            x_fast_permuted = x_fast.permute(0, 3, 1, 2).contiguous()
            fast_to_slow = self.lateral_connections[i](x_fast_permuted)
            fast_to_slow = fast_to_slow.permute(0, 2, 3, 1).contiguous()

            x_slow_permuted = x_slow.permute(0, 3, 1, 2).contiguous()
            slow_to_fast = self.inv_lateral_connections[i](x_slow_permuted)
            slow_to_fast = slow_to_fast.permute(0, 2, 3, 1).contiguous()

            x_slow_combined = x_slow + fast_to_slow
            x_fast_combined = x_fast + slow_to_fast
            
            x_fast = self.fast_blocks[i](x_fast_combined)
            x_slow = self.slow_blocks[i](x_slow_combined)
            
        summary_fast = self.fast_pool(x_fast)
        summary_slow = self.slow_pool(x_slow)

        combined_summary = torch.cat([summary_slow, summary_fast], dim=1)
        
        output_action = self.action_head(combined_summary)

        reversed_features = self.grad_reversal(combined_summary)
        output_aux = self.aux_classifier(reversed_features) # [수정] 이름 변경 (subject -> aux)

        return output_action, output_aux
