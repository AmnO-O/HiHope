"""Reusable Transformer cross-attention fusion block."""

from __future__ import annotations
import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """A single Transformer Fusion block (Cross-Attention + LayerNorm + FFN)."""
    def __init__(self, hidden: int, num_heads: int, ffn_expansion: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.alpha_attn = nn.Parameter(torch.ones(1))

        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ffn_expansion, hidden),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.alpha_ffn = nn.Parameter(torch.ones(1))

    def forward(self, q: torch.Tensor, kv_toks: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query=q, key=kv_toks, value=kv_toks)
        x = self.norm1(q + self.alpha_attn * attn_out)
        out = self.norm2(x + self.alpha_ffn * self.ffn(x))
        return out
