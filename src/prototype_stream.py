"""Two-Stream Prototype & Semantic Shift Representation Learning.

This module implements the components for the Two-Stream Bi-Encoder architecture:
    Stream 1 (Prototype): [CLS] target_word [SEP] -> h_proto
    Stream 2 (Context):   [CLS] context_sentence [SEP] -> h_ctx

Features:
    1. pool_prototype: Pools lexical subwords of the isolated target word,
       excluding special tokens [CLS] and [SEP] to extract the pure prototype.
    2. SemanticShiftFusion: Attention-based Cross-Fusion Transformer between h_ctx,
       h_proto, and their directional displacement (h_ctx - h_proto) using FusionBlock.
    3. prototype_rank_loss: Optional auxiliary margin ranking loss aligning
       cosine similarities with human compositionality ratings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_block import FusionBlock


def pool_prototype(
    hidden: torch.Tensor, 
    proto_mask: torch.Tensor,
    span_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool lexical subword representations from Stream 1.

    If span_mask is provided and has active tokens, pools ONLY the target word
    subwords within the template prompt (excluding prompt boilerplate).
    Otherwise, for an isolated target word sequence [CLS] w_1 ... w_K [SEP],
    this excludes position 0 ([CLS]) and the last active position ([SEP]) whenever K >= 1
    so that only the true lexical subword tokens are pooled.
    Degrades gracefully to masked mean over all active tokens if sequence length < 3.

    Args:
        hidden: Tensor of shape (B, L, H) from the prototype forward pass.
        proto_mask: Attention mask of shape (B, L) where 1 indicates active tokens.
        span_mask: Optional boolean or float mask of shape (B, L) marking target word tokens.

    Returns:
        Tensor of shape (B, H) containing the unpolluted prototype vector.
    """
    B, L, H = hidden.shape
    default_mask = proto_mask.clone().bool()
    lengths = proto_mask.sum(dim=-1).long()

    for i, length in enumerate(lengths):
        l_int = int(length.item())
        if l_int >= 3:
            default_mask[i, 0] = False           # Exclude [CLS]
            default_mask[i, l_int - 1] = False   # Exclude [SEP]

    if span_mask is not None:
        sm = span_mask.bool()
        has_span = sm.any(dim=-1, keepdim=True)
        word_mask = torch.where(has_span, sm, default_mask)
    else:
        word_mask = default_mask

    mask_float = word_mask.unsqueeze(-1).float()
    denom = mask_float.sum(dim=1).clamp(min=1.0)
    return (hidden * mask_float).sum(dim=1) / denom


def corrected_cosine_similarity(
    u: torch.Tensor, 
    v: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """Cosine similarity after per-vector mean-centering (anisotropy correction).

    Removes the dominant shared positive direction that causes raw Transformer
    embeddings to cluster in a narrow cone with high positive cosine similarities
    (anisotropy). Spreads the similarity range across [-1, 1], making the margin
    ranking loss far more effective.

    Args:
        u: First embedding tensor of shape (B, H) or (..., H).
        v: Second embedding tensor of shape (B, H) or (..., H).
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor of shape (B,) containing corrected cosine similarities.
    """
    u_c = u - u.mean(dim=-1, keepdim=True)
    v_c = v - v.mean(dim=-1, keepdim=True)
    return F.cosine_similarity(u_c, v_c, dim=-1, eps=eps)


class SemanticShiftFusion(nn.Module):
    """Symmetrical Attention-based Cross-Fusion between Context and Prototype.

    Compositionality assesses two complementary linguistic directions:
        1. Context-queried shift (h_ctx -> h_proto): How much does the context
           displace or modulate the literal prototype meaning?
        2. Prototype-queried literality (h_proto -> h_ctx): How much of the
           isolated literal prototype's semantics survives intact inside the context?

    Architecture:
        - Stream A (Context Query): q_ctx = h_ctx attends over [h_ctx, h_proto, h_ctx - h_proto].
        - Stream B (Prototype Query): q_proto = h_proto attends over [h_proto, h_ctx, h_proto - h_ctx].
        - Symmetrical Combiner: Merges both contextualized shift and literalness
          preservation vectors into a calibrated hidden representation.
        - Residual Connection: Preserves h_ctx baseline with LayerNorm.
    """

    def __init__(
        self, 
        hidden_size: int = 768, 
        num_layers: int = 2,
        num_heads: int = 4, 
        dropout: float = 0.1,
        use_adaptive_gate: bool = False,
        gate_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaptive_gate = use_adaptive_gate
        heads = num_heads if (hidden_size % num_heads == 0) else 1

        # Role embeddings for Context-Queried Stream: [0: Context, 1: Prototype, 2: Forward Displacement]
        self.ctx_type_emb = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.ctx_type_emb, std=0.02)

        # Role embeddings for Prototype-Queried Stream: [0: Prototype, 1: Context, 2: Reverse Displacement]
        self.proto_type_emb = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.proto_type_emb, std=0.02)

        # Multi-Head Cross-Attention Transformer blocks
        n_layers = max(1, num_layers)
        self.ctx_layers = nn.ModuleList([
            FusionBlock(hidden_size, num_heads=heads, ffn_expansion=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.proto_layers = nn.ModuleList([
            FusionBlock(hidden_size, num_heads=heads, ffn_expansion=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Symmetrical Merger combining both directional representations
        self.combiner = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.out_norm = nn.LayerNorm(hidden_size)

        if self.use_adaptive_gate:
            state_emb_dim = 16
            self.state_emb = nn.Embedding(3, state_emb_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_size + state_emb_dim, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
            # Init gate bias to -1.0 so initial g is ~0.27
            nn.init.constant_(self.gate_net[-2].bias, -1.0)
            nn.init.xavier_uniform_(self.gate_net[-2].weight, gain=0.1)

        self.last_cos: Optional[torch.Tensor] = None
        self.last_raw_cos: Optional[torch.Tensor] = None
        self.last_g: Optional[torch.Tensor] = None

    def forward(
        self, 
        h_ctx: torch.Tensor, 
        h_proto: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse contextual and prototype vectors via symmetrical cross-attention.

        Args:
            h_ctx: Contextual vector of shape (B, H).
            h_proto: Prototype vector of shape (B, H).
            state: Optional deterministic alignment state tensor of shape (B,).

        Returns:
            Fused vector of shape (B, H).
        """
        # Anisotropy-corrected and raw cosine similarities for ranking and diagnostics
        self.last_cos = corrected_cosine_similarity(h_ctx, h_proto)
        self.last_raw_cos = F.cosine_similarity(h_ctx, h_proto, dim=-1, eps=1e-8)

        # --- Stream 1: Context-queried Forward Shift (h_ctx -> h_proto) ---
        diff_fwd = h_ctx - h_proto
        kv_ctx = torch.stack([h_ctx, h_proto, diff_fwd], dim=1)  # (B, 3, H)
        type_ids = torch.tensor([0, 1, 2], device=h_ctx.device, dtype=torch.long)
        kv_ctx = kv_ctx + self.ctx_type_emb[type_ids].to(dtype=kv_ctx.dtype)

        q_ctx = h_ctx.unsqueeze(1)  # (B, 1, H)
        for layer in self.ctx_layers:
            q_ctx = layer(q_ctx, kv_ctx)
        z_ctx = q_ctx.squeeze(1)  # (B, H)

        # --- Stream 2: Prototype-queried Literality Survival (h_proto -> h_ctx) ---
        diff_rev = h_proto - h_ctx
        kv_proto = torch.stack([h_proto, h_ctx, diff_rev], dim=1)  # (B, 3, H)
        kv_proto = kv_proto + self.proto_type_emb[type_ids].to(dtype=kv_proto.dtype)

        q_proto = h_proto.unsqueeze(1)  # (B, 1, H)
        for layer in self.proto_layers:
            q_proto = layer(q_proto, kv_proto)
        z_proto = q_proto.squeeze(1)  # (B, H)

        # --- Symmetrical Integration with Context Residual ---
        fused_shift = self.combiner(torch.cat([z_ctx, z_proto], dim=-1))  # (B, H)

        if not self.use_adaptive_gate or state is None:
            self.last_g = None
            return self.out_norm(h_ctx + fused_shift)

        s_feat = self.state_emb(state)
        g = self.gate_net(torch.cat([h_ctx, h_proto, s_feat], dim=-1))  # (B, 1)
        self.last_g = g.detach().float()
        alpha = 0.5  # Max context attenuation
        beta = 1.0   # Max prototype shift amplification
        ctx_scaled = (1.0 - alpha * g) * h_ctx
        shift_scaled = (1.0 + beta * g) * fused_shift
        return self.out_norm(ctx_scaled + shift_scaled)


def prototype_rank_loss(
    cos_sim_or_h_ctx: torch.Tensor,
    ratings_or_h_proto: torch.Tensor,
    ratings: Optional[torch.Tensor] = None,
    margin: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise margin ranking loss with anisotropy correction.

    Supports either:
        1. prototype_rank_loss(cos_sim, ratings, margin=..., mask=...)
        2. prototype_rank_loss(h_ctx, h_proto, ratings, margin=..., mask=...)

    Encourages cosine similarity to correlate positively with compositionality:
    samples with higher human ratings should have higher prototype-context similarity.

    Args:
        cos_sim_or_h_ctx: Cosine similarities (B,) or context embeddings (B, H).
        ratings_or_h_proto: Ratings (B,) or prototype embeddings (B, H).
        ratings: Ground truth ratings if embeddings were provided as first 2 args.
        margin: Hinge loss margin (default: 0.2).
        mask: Valid sample mask of shape (B,).

    Returns:
        Scalar loss tensor.
    """
    if ratings is not None:
        # Called as (h_ctx, h_proto, ratings, ...)
        h_ctx = cos_sim_or_h_ctx
        h_proto = ratings_or_h_proto
        cos_sim = corrected_cosine_similarity(h_ctx, h_proto)
        target_ratings = ratings
    else:
        # Called as (cos_sim, ratings, ...)
        cos_sim = cos_sim_or_h_ctx
        target_ratings = ratings_or_h_proto

    if mask is not None:
        cos_sim = cos_sim[mask]
        target_ratings = target_ratings[mask]

    valid = torch.isfinite(target_ratings)
    cos_sim = cos_sim[valid]
    target_ratings = target_ratings[valid]

    n = cos_sim.size(0)
    if n < 2:
        return cos_sim.sum() * 0.0

    target_diff = target_ratings[:, None] - target_ratings[None, :]
    cos_diff = cos_sim[:, None] - cos_sim[None, :]

    pos_mask = target_diff > 0.5  # Only compare pairs with meaningful score gap
    if not pos_mask.any():
        return cos_sim.sum() * 0.0

    return F.relu(margin - cos_diff[pos_mask]).mean()
