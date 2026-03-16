from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BoundaryNet(nn.Module):
    def __init__(self, n_mels: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64 * n_mels,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x).squeeze(-1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBoundaryNet(nn.Module):
    def __init__(self, n_mels: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64 * n_mels, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.fc(x).squeeze(-1)
        return x


class MultiScaleTransformerBoundaryNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        long_pool: int = 4,
        num_sources: int = 0,
    ):
        super().__init__()
        self.long_pool = int(max(1, long_pool))
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj_short = nn.Linear(64 * n_mels, d_model)
        self.proj_long = nn.Linear(64 * n_mels, d_model)
        self.pos_short = PositionalEncoding(d_model)
        self.pos_long = PositionalEncoding(d_model)
        encoder_layer_short = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        encoder_layer_long = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder_short = nn.TransformerEncoder(encoder_layer_short, num_layers=num_layers)
        self.encoder_long = nn.TransformerEncoder(encoder_layer_long, num_layers=max(1, num_layers // 2))
        self.pool = nn.AvgPool1d(kernel_size=self.long_pool, stride=self.long_pool, ceil_mode=False)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(d_model, 1)
        self.source_emb = nn.Embedding(int(num_sources), d_model) if int(num_sources) > 0 else None

    def forward(self, mel: torch.Tensor, source_id: torch.Tensor = None) -> torch.Tensor:
        x = mel.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        s = self.proj_short(x)
        l = self.proj_long(x)

        if source_id is not None and self.source_emb is not None:
            sid = source_id.to(device=mel.device).view(-1).clamp(min=0, max=self.source_emb.num_embeddings - 1)
            emb = self.source_emb(sid).unsqueeze(1)
            s = s + emb
            l = l + emb

        s = self.pos_short(s)
        s = self.encoder_short(s)

        l_t = l.transpose(1, 2)
        l_t = self.pool(l_t)
        l = l_t.transpose(1, 2)
        l = self.pos_long(l)
        l = self.encoder_long(l)
        l_up = l.repeat_interleave(self.long_pool, dim=1)
        if l_up.size(1) < s.size(1):
            pad = s.size(1) - l_up.size(1)
            l_up = torch.nn.functional.pad(l_up, (0, 0, 0, pad))
        l_up = l_up[:, : s.size(1), :]

        fused = torch.cat([s, l_up], dim=-1)
        fused = self.fuse(fused)
        out = self.fc(fused).squeeze(-1)
        return out


class MultiScaleTransformerBoundaryNetV2(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        long_pool: int = 4,
        num_sources: int = 0,
    ):
        super().__init__()
        self.long_pool = int(max(1, long_pool))
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj_short = nn.Linear(64 * n_mels, d_model)
        self.proj_long = nn.Linear(64 * n_mels, d_model)
        self.pos_short = PositionalEncoding(d_model)
        self.pos_long = PositionalEncoding(d_model)
        self.pre_short = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pre_long = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_short = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        enc_long = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder_short = nn.TransformerEncoder(enc_short, num_layers=int(num_layers))
        self.encoder_long = nn.TransformerEncoder(enc_long, num_layers=max(1, int(num_layers) // 2))
        self.pool = nn.AvgPool1d(kernel_size=self.long_pool, stride=self.long_pool, ceil_mode=False)
        self.source_emb = nn.Embedding(int(num_sources), d_model) if int(num_sources) > 0 else None
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Sigmoid(),
        )
        self.smoother = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model),
        )

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn_fwd = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_bwd = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # x: (B, T, D), context: (B, S, D)
        # Forward attention: x queries context (Global guidance)
        out_fwd, _ = self.attn_fwd(x, context, context)
        
        # Simplified SOTA Fusion: Just use the forward attention (Context injection)
        # The backward/self reinforcement was causing instability and noise in v26
        return self.norm(x + self.dropout(out_fwd))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiScaleTransformerBoundaryNetV3(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        downsample_kernel: int = 3,
        downsample_stride: int = 3,
        downsample_dropout: float = 0.1,
        num_sources: int = 0,
        drop_path_rate: float = 0.1, # Added DropPath rate
    ):
        super().__init__()
        self.d_model = d_model
        self.source_emb = nn.Embedding(int(num_sources), d_model) if int(num_sources) > 0 else None
        
        # Input projection
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64 * n_mels, d_model)

        # Reverted to Sinusoidal PE (More stable than Learnable PE for this task)
        self.pos = PositionalEncoding(d_model, max_len=20000)
        
        self.pre = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # SOTA: Deep Network with Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(4, num_layers) + max(2, num_layers // 2) * 2)]
        
        # Short branch: 4 layers
        self.encoder_short = nn.Sequential(*[
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, drop_path=dpr[i])
            for i in range(max(4, num_layers))
        ])

        # Multi-scale Downsampling
        self.downsample_stride = int(max(1, downsample_stride))
        self.downsample = TimeDownsample1D(
            d_model, 
            kernel_size=int(downsample_kernel), 
            stride=self.downsample_stride, 
            padding=0, 
            dropout=downsample_dropout
        )
        self.pos_ds = PositionalEncoding(d_model, max_len=20000)
        
        # DS branch: 2+ layers
        start_idx = max(4, num_layers)
        self.encoder_ds = nn.Sequential(*[
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, drop_path=dpr[start_idx + i])
            for i in range(max(2, num_layers // 2))
        ])

        self.downsample_stride2 = int(max(1, self.downsample_stride * 3)) # 3x stride for global
        self.downsample2 = TimeDownsample1D(
            d_model, 
            kernel_size=int(downsample_kernel), 
            stride=self.downsample_stride2, 
            padding=0, 
            dropout=downsample_dropout
        )
        self.pos_ds2 = PositionalEncoding(d_model, max_len=20000)
        
        # DS2 branch: 2+ layers
        start_idx = max(4, num_layers) + max(2, num_layers // 2)
        self.encoder_ds2 = nn.Sequential(*[
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, drop_path=dpr[start_idx + i])
            for i in range(max(2, num_layers // 2))
        ])

        # SOTA: Bidirectional Fusion (Replacing simple FPN/Gate)
        # Fusion Level 1: Fine <-> Coarse
        self.bi_attn_1 = BidirectionalCrossAttention(d_model, nhead, dropout)
        # Fusion Level 2: Coarse <-> Global
        self.bi_attn_2 = BidirectionalCrossAttention(d_model, nhead, dropout)
        
        # Retain FPN blocks for feature projection
        self.fpn_1 = FPNBlock(d_model, d_model)
        self.fpn_2 = FPNBlock(d_model, d_model)
        self.fpn_3 = FPNBlock(d_model, d_model)
        
        # Temporal Gating (Retained from v18)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Sigmoid(),
        )
        self.smoother = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
        )
        
        # Contrastive Head (for Structure Contrastive Loss)
        self.proj_contrast = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128) # Project to lower dim for contrastive loss
        )
        
        # Multi-head Prediction (Retained from v18)
        self.head_short = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_ds = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_ds2 = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_fuse = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_weight = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model, 5, kernel_size=1),
        )
        self.head_temp = nn.Parameter(torch.tensor(1.0))
        self.boundary_scale = nn.Parameter(torch.tensor(1.0))
        self.boundary_bias = nn.Parameter(torch.tensor(0.0))
        
        # Temporal Attention Context (v18 feature)
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, mel: torch.Tensor, source_id: torch.Tensor = None) -> torch.Tensor:
        # Input processing (Reverted to v18 style)
        x = mel.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        s = self.proj(x)
        
        if source_id is not None and self.source_emb is not None:
            sid = source_id.to(device=mel.device).view(-1).clamp(min=0, max=self.source_emb.num_embeddings - 1)
            emb = self.source_emb(sid).unsqueeze(1)
            s = s + emb
            
        s = self.pos(s)
        
        s = self.pre(s)
        s = self.encoder_short(s)
        
        # Multi-scale encoding
        ds = self.downsample(s)
        ds = self.pos_ds(ds)
        ds = self.encoder_ds(ds)
        
        ds2 = self.downsample2(ds)
        ds2 = self.pos_ds2(ds2)
        ds2 = self.encoder_ds2(ds2)
        
        # SOTA: Bidirectional Fusion
        # 1. Enhance Coarse (ds) with Global (ds2)
        # Fix: Use linear interpolation for upsampling to match dimensions exactly
        ds2_up = F.interpolate(ds2.transpose(1, 2), size=ds.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        ds_enhanced = self.bi_attn_2(ds, ds2_up)
        
        # 2. Enhance Fine (s) with Enhanced Coarse (ds_enhanced)
        ds_enhanced_up = F.interpolate(ds_enhanced.transpose(1, 2), size=s.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        s_enhanced = self.bi_attn_1(s, ds_enhanced_up)
        
        # FPN Projection (on enhanced features)
        p3 = self.fpn_3(ds2)
        p2 = self.fpn_2(ds_enhanced) + F.interpolate(p3.transpose(1, 2), size=ds.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        p1 = self.fpn_1(s_enhanced) + F.interpolate(p2.transpose(1, 2), size=s.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        
        # Upsample for final fusion
        ds_up = F.interpolate(ds.transpose(1, 2), size=s.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        ds2_up = F.interpolate(ds2.transpose(1, 2), size=s.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        
        # Gating Fusion
        ds_sum = 0.5 * (ds_up + ds2_up)
        g = self.gate(torch.cat([s_enhanced, ds_sum], dim=-1))
        fused = g * s_enhanced + (1.0 - g) * ds_sum
        
        # Add FPN context
        fused = fused + 0.1 * p1
        
        fused = self.fuse(fused)
        fused = fused.transpose(1, 2)
        fused = fused * self.temporal_gate(fused)
        fused = fused + self.smoother(fused)
        fused = fused.transpose(1, 2)
        
        # Temporal Attention Context
        attn = self.temporal_attn(fused).squeeze(-1)
        attn = torch.softmax(attn, dim=-1).unsqueeze(-1)
        context = (fused * attn).sum(dim=1, keepdim=True)
        fused = fused + context
        
        # Contrastive Feature Projection (for auxiliary loss)
        contrast_feat = self.proj_contrast(fused) # (B, T, 128)
        
        # Multi-head prediction
        logit_short = self.head_short(s).squeeze(-1)
        logit_ds = self.head_ds(ds_up).squeeze(-1)
        logit_ds2 = self.head_ds2(ds2_up).squeeze(-1)
        logit_fuse = self.head_fuse(fused).squeeze(-1)
        
        global_feat = fused.mean(dim=1, keepdim=True).expand_as(fused)
        logit_global = self.head_fuse(global_feat).squeeze(-1)
        
        w = self.head_weight(fused.transpose(1, 2)).transpose(1, 2)
        temp = self.head_temp.clamp(min=0.2)
        weights = torch.softmax(w / temp, dim=-1)
        
        out = (
            weights[..., 0] * logit_short
            + weights[..., 1] * logit_ds
            + weights[..., 2] * logit_ds2
            + weights[..., 3] * logit_fuse
            + weights[..., 4] * logit_global
        )
        
        out = out * self.boundary_scale + self.boundary_bias
        
        if self.training:
            return {
                "out": out,
                "contrast": contrast_feat,
                "short": logit_short,
                "ds": logit_ds,
                "ds2": logit_ds2,
                "fuse": logit_fuse,
                "global": logit_global
            }
        return out


class TVLoss1D(nn.Module):
    def __init__(
        self, beta=1.0, lambda_tv=0.4, boundary_threshold=0.01, reduction_weight=0.1
    ):
        """
        Args:
            beta: Exponential parameter for TV loss (recommended 0.5~1.0)
            lambda_tv: Overall weight for TV loss
            boundary_threshold: Label threshold to determine if a region is a "boundary area" (e.g., 0.01)
            reduction_weight: Scaling factor for TV penalty within boundary regions (e.g., 0.1, meaning only 10% penalty)
        """
        super().__init__()
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.boundary_threshold = boundary_threshold
        self.reduction_weight = reduction_weight

    def forward(self, pred, target=None):
        """
        Args:
            pred: (B, T) or (B, T, 1), float boundary scores output by the model
            target: (B, T) or (B, T, 1), ground truth labels (optional, used for spatial weighting if provided)

        Returns:
            scalar: weighted TV loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target is not None and target.dim() == 3:
            target = target.squeeze(-1)

        diff = pred[:, 1:] - pred[:, :-1]
        tv_base = torch.pow(torch.abs(diff) + 1e-8, self.beta)

        if target is None:
            return self.lambda_tv * tv_base.mean()

        left_in_boundary = target[:, :-1] > self.boundary_threshold
        right_in_boundary = target[:, 1:] > self.boundary_threshold
        near_boundary = left_in_boundary | right_in_boundary
        weight_mask = torch.where(
            near_boundary,
            self.reduction_weight * torch.ones_like(tv_base),
            torch.ones_like(tv_base),
        )
        tv_weighted = (tv_base * weight_mask).mean()
        return self.lambda_tv * tv_weighted


class SoftmaxFocalLoss(nn.Module):
    """
    Softmax Focal Loss for single-label multi-class classification.
    Suitable for mutually exclusive classes.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C] (for classification) or [B, T, C] (for sequence), raw logits
            targets: [B] (hard label) or [B, C] (soft) or [B, T, C]
        Returns:
            loss: scalar
        """
        # Handle different input shapes, ensure pred is logits
        if pred.ndim == 2:  # [B, C]
            log_probs = F.log_softmax(pred, dim=-1)
        elif pred.ndim == 3:  # [B, T, C]
            log_probs = F.log_softmax(pred, dim=-1)
        else:
            raise ValueError(f"Unsupported pred shape: {pred.shape}")

        probs = torch.exp(log_probs)

        # Handle targets
        if targets.dtype == torch.long and targets.ndim == pred.ndim - 1:
            targets_onehot = F.one_hot(targets, num_classes=pred.size(-1)).float()
        else:
            targets_onehot = targets

        p_t = (probs * targets_onehot).sum(dim=-1)
        p_t = p_t.clamp(min=1e-8, max=1.0 - 1e-8)

        if self.alpha > 0:
            alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (
                1 - targets_onehot
            )
            alpha_weight = (alpha_t * targets_onehot).sum(dim=-1)
        else:
            alpha_weight = 1.0

        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -log_probs * targets_onehot
        ce_loss = ce_loss.sum(dim=-1)

        loss = alpha_weight * focal_weight * ce_loss

        if pred.ndim == 3:
            # Masking logic could go here if needed, but for now just mean
            pass

        return loss.mean()


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class TimeDownsample1D(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int = None,
        kernel_size: int = 3,
        stride: int = 3,
        padding: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_out = dim_out or dim_in
        
        # Depthwise Conv (SOTA: kernel_size=5, stride=5 in SongFormer)
        self.depthwise_conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim_in,
            bias=False,
        )
        # Pointwise Conv
        self.pointwise_conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=self.dim_out,
            kernel_size=1,
            bias=False,
        )
        # SOTA: Residual Connection in Downsample
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=padding)
        self.residual_conv = nn.Conv1d(dim_in, self.dim_out, kernel_size=1, bias=False) if dim_in != self.dim_out else None
        
        self.norm = nn.LayerNorm(self.dim_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        
        # SOTA: TimeDownsample
        # 1. Depthwise -> Pointwise
        x_c = x.transpose(1, 2) # (B, C, T)
        x_c = self.depthwise_conv(x_c)
        x_c = self.pointwise_conv(x_c)
        
        # 2. Residual: AvgPool + Linear
        res = self.pool(residual.transpose(1, 2)) # (B, C, T_down)
        if self.residual_conv is not None:
            res = self.residual_conv(res)
            
        x_c = x_c + res
        x_c = x_c.transpose(1, 2) # (B, T_down, C_out)
        
        # 3. Norm -> Act -> Drop
        x_c = self.norm(x_c)
        x_c = self.act(x_c)
        x_c = self.drop(x_c)
        return x_c





class RoPEMultiHeadAttention(nn.Module):
    """Rotary Position Embedding based Multi-Head Attention (SOTA: SongFormer style)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :].to(dtype), emb.sin()[None, None, :, :].to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cache(t, x.device, q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        dp = float(self.drop.p) if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class TransformerBlockV4(nn.Module):
    """Pre-norm Transformer block with RoPE attention and DropPath."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BoundaryHead(nn.Module):
    """SOTA-style MLP head for boundary detection."""
    def __init__(self, d_model: int, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 8]
        dims = [d_model] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def reset_bias(self, prior: float = 0.01):
        bias_val = -torch.log(torch.tensor((1 - prior) / prior))
        self.net[-1].bias.data.fill_(bias_val.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BoundaryNetV4(nn.Module):
    """
    V4 Boundary Detection Model - SOTA-aligned clean architecture.
    
    Design principles (from SongFormer analysis):
    1. Strong Conv frontend for local feature extraction
    2. TimeDownsample for temporal compression (like SOTA)
    3. RoPE-based Transformer encoder (like SOTA)
    4. Single boundary head with MLP (like SOTA)
    5. Higher capacity (d_model=256) vs V3's 192
    6. No multi-branch/multi-head complexity
    """
    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        downsample_kernel: int = 3,
        downsample_stride: int = 3,
        num_sources: int = 0,
    ):
        super().__init__()
        self.d_model = d_model

        # Strong Conv frontend: extract local spectral features
        # 3 Conv layers with BatchNorm (more expressive than V3's 2 Conv layers)
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Project flattened conv features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(64 * n_mels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )

        # SOTA: TimeDownsample for temporal compression
        self.downsample = TimeDownsample1D(
            dim_in=d_model,
            dim_out=d_model,
            kernel_size=downsample_kernel,
            stride=downsample_stride,
            padding=0,
            dropout=dropout,
        )

        # Source embedding (optional, for multi-source training)
        self.source_emb = nn.Embedding(max(1, int(num_sources)), d_model) if int(num_sources) > 0 else None

        # RoPE Transformer encoder with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.encoder = nn.Sequential(*[
            TransformerBlockV4(d_model, nhead, dim_feedforward, dropout, drop_path=dpr[i])
            for i in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # SOTA: Single boundary head with MLP (no multi-head weighted fusion)
        self.boundary_head = BoundaryHead(d_model, hidden_dims=[128, 64, 8])

        # Contrastive projection (for optional auxiliary loss)
        self.proj_contrast = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
        )

    def forward(self, mel: torch.Tensor, source_id: torch.Tensor = None) -> torch.Tensor:
        # mel: (B, n_mels, T) -> need (B, 1, n_mels, T) for Conv2d
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.frontend(x)  # (B, 64, n_mels, T)

        # Reshape: (B, 64, n_mels, T) -> (B, T, 64*n_mels)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # SOTA: TimeDownsample
        x = self.downsample(x)  # (B, T_down, d_model)

        # Source embedding
        if source_id is not None and self.source_emb is not None:
            b = x.size(0)
            sid = source_id.to(device=mel.device).view(-1).clamp(min=0, max=self.source_emb.num_embeddings - 1)
            emb = self.source_emb(sid).unsqueeze(1)  # (B, 1, d_model)
            x = x + emb

        # RoPE Transformer encoder
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Boundary prediction
        logits = self.boundary_head(x)  # (B, T_down)

        if self.training:
            contrast_feat = self.proj_contrast(x)
            return {"out": logits, "contrast": contrast_feat}
        return logits


class SegmentClassifier(nn.Module):
    def __init__(self, n_mels: int, labels: List[str], hidden: int = 256):
        super().__init__()
        self.labels = labels
        self.net = nn.Sequential(
            nn.LayerNorm(n_mels),
            nn.Linear(n_mels, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, len(labels)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class SegmentMelCNN(nn.Module):
    def __init__(self, n_mels: int, labels: List[str], channels: int = 64):
        super().__init__()
        self.labels = labels
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(channels * 2, len(labels))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        x = self.features(mel)
        x = x.view(x.size(0), -1)
        return self.head(x)


class SegmentMelAttn(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Mixup/CutMix regularization
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, D)
        if self.training:
            # Apply Mixup or CutMix with small probability
            if torch.rand(1).item() < 0.2:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                index = torch.randperm(x.size(0)).to(x.device)
                x = lam * x + (1 - lam) * x[index, :]
        
        q = x.mean(dim=1, keepdim=True) # (B, 1, D)
        out, _ = self.attn(q, x, x, key_padding_mask=mask)
        out = self.dropout(out)
        out = self.norm(q + out)
        return out.squeeze(1)


class StatisticPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1) # (B, 2*C)

class SegmentMelAttnGate(nn.Module):
    def __init__(self, n_mels: int, labels: List[str], channels: int = 64, attn_dim: int = 128, heads: int = 4):
        super().__init__()
        self.labels = labels
        self.heads = int(max(1, heads))
        
        # SOTA-aligned Feature Extraction:
        # Instead of MaxPool2d which destroys time/freq info, use Conv2d with no pooling 
        # followed by TimeDownsample1D (SongFormer style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64 * n_mels, channels * 2) # Project to d_model
        
        # SongFormer-style Time Downsampling
        self.time_downsample = TimeDownsample1D(
            dim_in=channels * 2,
            dim_out=channels * 2,
            kernel_size=5,
            stride=5, # Aggressive downsampling to reduce length (like MaxPool(2,2)*2 but better)
            padding=0,
            dropout=0.1
        )
        
        # Squeeze-and-Excitation (Retained for channel attention)
        self.se = SEBlock1D(channels * 2, reduction=4)
        
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2),
            nn.Sigmoid(),
        )
        # AST-style: Statistic Pooling
        self.stat_pool = StatisticPooling()
        
        self.attn = nn.Sequential(
            nn.Conv1d(channels * 2, attn_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(attn_dim, self.heads, kernel_size=1),
        )
        # Optimized fusion gate
        self.fuse_gate = nn.Sequential(
            nn.Linear(channels * 2 * self.heads * 2, channels * 2 * self.heads // 2), 
            nn.GELU(),
            nn.Linear(channels * 2 * self.heads // 2, channels * 2 * self.heads),
            nn.Sigmoid(),
        )
        # Global projection
        self.global_proj = nn.Linear(channels * 2 * 2, channels * 2 * self.heads)
        self.global_gate = nn.Sequential(
            nn.Linear(channels * 2 * self.heads * 2, channels * 2 * self.heads // 2),
            nn.GELU(),
            nn.Linear(channels * 2 * self.heads // 2, channels * 2 * self.heads),
            nn.Sigmoid(),
        )
        self.pool_drop = nn.Dropout(0.3)
        
        # Learnable PE
        self.pos = LearnablePositionalEncoding(channels * 2, dropout=0.1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels * 2,
            nhead=heads,
            dim_feedforward=channels * 8,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.out_norm = nn.LayerNorm(channels * 2 * self.heads)
        self.head = nn.Linear(channels * 2 * self.heads, len(labels))
        
        # Mixup/CutMix regularization params
        self.mixup_alpha = 0.2
        self.grad_checkpointing = True 

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
            
        # Mixup during training
        if self.training and torch.rand(1).item() < 0.2:
             lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
             index = torch.randperm(mel.size(0)).to(mel.device)
             mel = lam * mel + (1 - lam) * mel[index]
        
        # Use gradient checkpointing for feature extraction
        if self.grad_checkpointing and self.training:
            if not mel.requires_grad:
                mel.requires_grad_(True)
            x = torch.utils.checkpoint.checkpoint(self.features, mel, use_reentrant=False)
        else:
            x = self.features(mel)
        
        # x: (B, 64, F, T) -> Permute -> (B, T, 64*F)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        # Project to d_model
        x = self.proj(x) # (B, T, C)
        
        # SOTA: TimeDownsample
        x = self.time_downsample(x) # (B, T_down, C)
        
        # SE Block
        x = x.transpose(1, 2) # (B, C, T)
        x = self.se(x)
        x = x.transpose(1, 2) # (B, T, C)

        # Temporal Gating
        gate = self.temporal_gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * gate
        
        # Add PE + Dropout
        x = self.pos(x)
        
        # Transformer
        x = self.encoder(x)
        x = x.transpose(1, 2) # (B, C, T)
        
        attn = self.attn(x)
        attn = torch.softmax(attn, dim=-1)
        pooled = torch.einsum("bct,bht->bhc", x, attn)
        pooled = self.pool_drop(pooled.reshape(x.size(0), -1))
        
        x2 = F.avg_pool1d(x, kernel_size=2, stride=2)
        attn2 = self.attn(x2)
        attn2 = torch.softmax(attn2, dim=-1)
        pooled2 = self.pool_drop(torch.einsum("bct,bht->bhc", x2, attn2).reshape(x.size(0), -1))
        
        fuse = torch.cat([pooled, pooled2], dim=-1)
        gate_f = self.fuse_gate(fuse)
        fused = gate_f * pooled + (1.0 - gate_f) * pooled2
        
        # Statistic Pooling for global features (Robustness)
        pooled_global = self.stat_pool(x) # (B, 2*C)
        pooled_global = self.global_proj(pooled_global) # (B, H)
        
        fuse_g = torch.cat([fused, pooled_global], dim=-1)
        gate_g = self.global_gate(fuse_g)
        fused = gate_g * fused + (1.0 - gate_g) * pooled_global
        return self.head(self.out_norm(fused))


class MultiResolutionBoundaryNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        local_window_sec: float = 30.0,
        global_window_sec: float = 420.0,
        hop_length: int = 512,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.local_window_sec = local_window_sec
        self.global_window_sec = global_window_sec
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.local_frames = int(local_window_sec * sample_rate / hop_length)
        self.global_frames = int(global_window_sec * sample_rate / hop_length)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj_local = nn.Linear(64 * n_mels, d_model)
        self.proj_global = nn.Linear(64 * n_mels, d_model)
        self.ln_local = nn.LayerNorm(d_model)
        self.ln_global = nn.LayerNorm(d_model)
        self.pos_local = PositionalEncoding(d_model, max_len=20000)
        self.pos_global = PositionalEncoding(d_model, max_len=20000)
        enc_local = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        enc_global = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder_local = nn.TransformerEncoder(enc_local, num_layers=num_layers)
        self.encoder_global = nn.TransformerEncoder(enc_global, num_layers=max(1, num_layers // 2))
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, mel: torch.Tensor, mel_global: torch.Tensor = None) -> torch.Tensor:
        x = mel.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        local_feat = self.proj_local(x)
        local_feat = self.ln_local(local_feat)
        local_feat = self.pos_local(local_feat)
        local_feat = self.encoder_local(local_feat)
        if mel_global is not None and mel_global.shape[2] > 0:
            xg = mel_global.unsqueeze(1)
            xg = self.conv(xg)
            xg = xg.permute(0, 3, 1, 2).contiguous()
            xg = xg.view(xg.size(0), xg.size(1), -1)
            global_feat = self.proj_global(xg)
            global_feat = self.ln_global(global_feat)
            global_feat = global_feat.transpose(1, 2)
            global_feat = self.downsample(global_feat)
            global_feat = global_feat.transpose(1, 2)
            global_feat = self.pos_global(global_feat)
            global_feat = self.encoder_global(global_feat)
            target_len = local_feat.size(1)
            src_len = global_feat.size(1)
            if src_len < target_len:
                repeat_factor = max(1, target_len // src_len)
                global_up = global_feat.repeat_interleave(repeat_factor, dim=1)
                if global_up.size(1) < target_len:
                    pad = target_len - global_up.size(1)
                    global_up = F.pad(global_up, (0, 0, 0, pad))
                global_up = global_up[:, :target_len, :]
            else:
                global_up = global_feat[:, :target_len, :]
            g = self.gate(torch.cat([local_feat, global_up], dim=-1))
            fused = g * local_feat + (1.0 - g) * global_up
        else:
            fused = local_feat
        fused = self.fuse(fused)
        out = self.fc(fused).squeeze(-1)
        return out


def save_boundary(path: str, model: nn.Module, cfg: Dict):
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)


def save_classifier(path: str, model: SegmentClassifier, cfg: Dict):
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)
