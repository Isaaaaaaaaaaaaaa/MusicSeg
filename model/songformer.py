from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SongFormerConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    downsample_factor: int = 3
    global_pool: int = 8
    num_sources: int = 1
    rope_base: float = 10000.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = -x2
    y2 = x1
    y = torch.stack([y1, y2], dim=-1)
    return y.flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _rope_cache(
    seq_len: int, head_dim: int, device: torch.device, base: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = int(d_model // nhead)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = _rope_cache(t, self.head_dim, x.device, self.rope_base)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        dropout_p = float(self.drop.p) if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class SongFormerEncoderLayer(nn.Module):
    def __init__(self, cfg: SongFormerConfig):
        super().__init__()
        self.attn = RoPEAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, rope_base=cfg.rope_base)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ResidualDownsample(nn.Module):
    def __init__(self, d_model: int, factor: int = 3):
        super().__init__()
        f = int(factor)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, stride=f, padding=1, groups=d_model)
        self.dw_pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.avg = nn.AvgPool1d(kernel_size=3, stride=f, ceil_mode=True)
        self.avg_pw = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.dw_pw(self.dw(x))
        y2 = self.avg_pw(self.avg(x))
        return y1 + y2


class SongFormerNet(nn.Module):
    def __init__(self, n_mels: int, num_labels: int, cfg: Optional[SongFormerConfig] = None):
        super().__init__()
        self.cfg = cfg or SongFormerConfig()
        self.num_labels = int(num_labels)
        self.in_proj = nn.Sequential(
            nn.Conv1d(int(n_mels), self.cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.cfg.d_model, self.cfg.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.cfg.d_model),
        )
        self.global_pool = nn.AvgPool1d(kernel_size=int(self.cfg.global_pool), stride=int(self.cfg.global_pool), ceil_mode=True)
        self.global_proj = nn.Sequential(
            nn.Conv1d(int(n_mels), self.cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.cfg.d_model, self.cfg.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.cfg.d_model),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(self.cfg.d_model * 2, self.cfg.d_model, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(1, self.cfg.d_model),
        )
        self.downsample = ResidualDownsample(self.cfg.d_model, factor=self.cfg.downsample_factor)
        self.post_down_norm = nn.GroupNorm(1, self.cfg.d_model)
        self.source_emb = nn.Embedding(int(self.cfg.num_sources), int(self.cfg.d_model))
        self.layers = nn.ModuleList([SongFormerEncoderLayer(self.cfg) for _ in range(int(self.cfg.num_layers))])
        self.boundary_head = nn.Linear(self.cfg.d_model, 1)
        self.function_head = nn.Linear(self.cfg.d_model, self.num_labels)

    def forward(self, mel: torch.Tensor, source_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mel.dim() != 3:
            raise ValueError("mel must be [B, n_mels, T]")
        b, _, t = mel.shape
        local = self.in_proj(mel)
        pooled = self.global_pool(mel)
        glob = self.global_proj(pooled)
        factor = int(self.cfg.global_pool)
        glob = glob.repeat_interleave(factor, dim=-1)[..., :t]
        x = self.fuse(torch.cat([local, glob], dim=1))
        x = self.downsample(x)
        x = self.post_down_norm(x)
        td = x.shape[-1]
        if source_id is None:
            source_id = torch.zeros((b,), device=mel.device, dtype=torch.long)
        s = self.source_emb(source_id).unsqueeze(-1).expand(b, self.cfg.d_model, td)
        x = x + s
        x = x.transpose(1, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        boundary_logits = self.boundary_head(x).squeeze(-1)
        function_logits = self.function_head(x)
        return {"boundary_logits": boundary_logits, "function_logits": function_logits}


def save_songformer(path: str, model: nn.Module, cfg: Dict):
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)
