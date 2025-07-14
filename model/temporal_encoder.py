# model/temporal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalPositionalEncoding(nn.Module):
    """Adds sine/cosine positional encodings to a time series."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WeatherDifferentiatedAttention(nn.Module):
    """
    Core Innovation: Generates a differentiated weather context representation for each turbine node.
    It calculates attention scores from each turbine to all weather nodes and then aggregates
    weather information based on these weights.
    """
    def __init__(self, d_model: int, num_turbines: int, num_weather: int, d_hidden: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_turbines = num_turbines
        self.num_weather = num_weather
        
        self.weather_transform = nn.Linear(d_model, d_hidden)
        self.turbine_transform = nn.Linear(d_model, d_hidden)
        
        self.context_attn = nn.MultiheadAttention(d_hidden, num_heads, batch_first=True)
        self.output_proj = nn.Linear(d_hidden, d_model)
        
        self.norm_t = nn.LayerNorm(d_hidden)
        self.norm_w = nn.LayerNorm(d_hidden)

    def forward(self, turbine_feats: torch.Tensor, weather_feats: torch.Tensor) -> torch.Tensor:
        # turbine_feats: [B, T, num_turbines, d_model]
        # weather_feats: [B, T, num_weather, d_model]
        B, T, Nt, C = turbine_feats.shape
        _, _, Nw, _ = weather_feats.shape

        h_t = self.norm_t(self.turbine_transform(turbine_feats)) # [B, T, Nt, d_hidden]
        h_w = self.norm_w(self.weather_transform(weather_feats)) # [B, T, Nw, d_hidden]

        # Calculate attention weights using dot-product attention: [B, T, Nt, Nw]
        attn_logits = torch.einsum('btnd,btmd->btnm', h_t, h_w) / math.sqrt(h_t.size(-1))
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Aggregate weather features based on attention weights: [B, T, Nt, d_hidden]
        context = torch.einsum('btnm,btmd->btnd', attn_weights, h_w)

        # Enhance the context with self-attention
        context_flat = context.reshape(B * T, Nt, -1)
        enhanced_context_flat, _ = self.context_attn(context_flat, context_flat, context_flat)
        enhanced_context = enhanced_context_flat.view(B, T, Nt, -1)
        
        # Return the enhanced turbine features (with a residual connection)
        enhanced_turbine_feats = turbine_feats + self.output_proj(enhanced_context)
        return enhanced_turbine_feats


class TemporalEncoderLayer(nn.Module):
    """
    The Temporal Encoder Layer for WD-SGformer.
    It processes standard temporal self-attention and weather-differentiated attention 
    in parallel and fuses their outputs using a gating mechanism.
    """
    def __init__(self, d_model: int, num_heads: int, num_turbines: int, num_weather: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.num_turbines = num_turbines
        self.num_weather = num_weather
        self.total_nodes = num_turbines + num_weather
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.weather_diff_attn = WeatherDifferentiatedAttention(d_model, num_turbines, num_weather)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B * total_nodes, T, d_model]
        B_N, T, C = src.shape
        B = B_N // self.total_nodes
        
        # 1. Standard Self-Attention
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout(attn_out)
        
        # 2. Weather-Differentiated Attention
        x_reshaped = src.view(B, self.total_nodes, T, C)
        turbine_feats = x_reshaped[:, :self.num_turbines].permute(0, 2, 1, 3)
        weather_feats = x_reshaped[:, self.num_turbines:].permute(0, 2, 1, 3)
        
        enhanced_turbine_feats = self.weather_diff_attn(turbine_feats, weather_feats).permute(0, 2, 1, 3)
        
        # 3. Gating and Fusion
        gate = self.fusion_gate(enhanced_turbine_feats)
        fused_turbine_feats = gate * enhanced_turbine_feats + (1 - gate) * x_reshaped[:, :self.num_turbines]
        
        # Combine and reshape back
        x_fused = torch.cat([fused_turbine_feats, x_reshaped[:, self.num_turbines:]], dim=1)
        src = x_fused.view(B_N, T, C)

        # 4. Feed-Forward Network
        ffn_out = self.ffn(self.norm2(src))
        src = src + self.dropout(ffn_out)
        
        return src