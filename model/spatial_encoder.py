# model/spatial_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialPositionalEncoding(nn.Module):
    """Generates encodings for geographical coordinates (lat, lon) using Fourier features."""
    def __init__(self, d_model: int, num_frequencies: int = 32):
        super().__init__()
        self.proj = nn.Linear(2 * 2 * num_frequencies, d_model)
        freq_bands = 2.0 ** torch.linspace(0, 8, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, node_coords: torch.Tensor) -> torch.Tensor:
        # node_coords: [N, 2]
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(node_coords * freq))
            out.append(torch.cos(node_coords * freq))
        fourier_encoding = torch.cat(out, dim=-1)
        return self.proj(fourier_encoding)

class DynamicGraphBias(nn.Module):
    """
    Core Innovation: Generates a dynamic graph bias based on the time series of a node's dynamic features.
    It uses a CNN and an Attention mechanism to capture the temporal evolution of features and maps them
    into an attention bias.
    """
    def __init__(self, d_feature: int, num_heads: int, d_hidden: int = 32):
        super().__init__()
        self.temporal_conv = nn.Conv1d(d_feature, d_hidden, kernel_size=3, padding=1)
        self.time_attention = nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True)
        self.attn_pair_proj = nn.Linear(d_hidden, num_heads)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T, N, d_feature]
        B, T, N, _ = features.shape
        ts_features = features.permute(0, 2, 3, 1).reshape(B * N, -1, T)
        ts_conv_out = F.relu(self.temporal_conv(ts_features)).transpose(1, 2)
        attn_out, _ = self.time_attention(ts_conv_out, ts_conv_out, ts_conv_out)
        ts_node_rep = attn_out.mean(dim=1).view(B, N, -1)
        ts_pair_diff = ts_node_rep.unsqueeze(2) - ts_node_rep.unsqueeze(1)
        bias = self.attn_pair_proj(ts_pair_diff).permute(0, 3, 1, 2)
        return bias

class SpatialGraphAttention(nn.Module):
    """
    The Spatial Graph Attention mechanism for WD-SGformer.
    It fuses standard self-attention with a static geographic bias and a dynamic graph bias.
    """
    def __init__(self, d_model: int, num_heads: int, d_feature: int, dropout: float = 0.1):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.geo_bias_linear = nn.Linear(d_model, num_heads) if d_model > 0 else None
        self.dynamic_graph_bias = DynamicGraphBias(d_feature, num_heads) if d_feature > 0 else None

    def forward(self, x: torch.Tensor, geo_embedding: torch.Tensor, dynamic_features: torch.Tensor) -> torch.Tensor:
        B_T, N, _ = x.size()
        B = B_T // dynamic_features.size(1) if B_T > 0 and dynamic_features.size(1) > 0 else B_T

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim)
        
        # --- Core Innovation ---
        # Static Geographic Bias
        if self.geo_bias_linear is not None and geo_embedding is not None:
            geo_bias = self.geo_bias_linear(geo_embedding)
            geo_bias = (geo_bias.unsqueeze(1) - geo_bias.unsqueeze(2)).permute(0, 3, 1, 2)
            attn_logits = attn_logits + geo_bias.repeat(B, 1, 1, 1)

        # Dynamic Graph Bias
        if self.dynamic_graph_bias is not None and dynamic_features is not None:
            dyn_bias = self.dynamic_graph_bias(dynamic_features)
            attn_logits = attn_logits + dyn_bias.repeat_interleave(dynamic_features.size(1), dim=0)
        # --- End of Core Innovation ---

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.einsum('bhnm,bhmd->bhnd', attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B_T, N, self.d_model)
        return self.out_proj(attn_output)

class SpatialEncoderLayer(nn.Module):
    """The Spatial Encoder Layer for WD-SGformer."""
    def __init__(self, d_model: int, num_heads: int, d_feature: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SpatialGraphAttention(d_model, num_heads, d_feature, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, geo_embedding: torch.Tensor, dynamic_features: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm
        src_norm = self.norm1(src)
        attn_out = self.attn(src_norm, geo_embedding, dynamic_features)
        src = src + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(src))
        src = src + self.dropout(ffn_out)
        return src