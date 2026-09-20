"""Two-Stream Bi-Encoder Architecture for Compositionality Assessment.

This module implements the canonical Two-Stream architecture:
    Stream 1 (Target Word):   [CLS] target_word [SEP]       -> h_word (out-of-context prototype)
    Stream 2 (Context):       [CLS] context_sentence [SEP] -> h_context (in-context span representation)

Interaction & Fusion:
    1. Directional semantic displacement: Delta_h = h_context - h_word
    2. Element-wise multiplicative interaction: h_context * h_word
    3. Calibrated cosine similarity: cos(h_context, h_word)
    4. Lightweight semantic fusion layer -> GaussHead -> (mu, sigma)

mmBERT-base (~110M parameters) is trained with full fine-tuning or top-layer unfreezing.
No LoRA bottleneck is used, ensuring full cross-lingual gradient propagation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .constants import SCORE_MAX, SCORE_MIN
from .heads import GaussHead
from .prototype_stream import pool_prototype


class TwoStreamBiEncoderModel(nn.Module):
    """Two-Stream Bi-Encoder for Compositionality and Uncertainty Prediction."""

    def __init__(
        self,
        backbone: str = "jhu-clsp/mmBERT-base",
        hidden_size: int = 768,
        head_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 0.05,
    ):
        super().__init__()
        self.lm = AutoModel.from_pretrained(backbone)
        self.hidden_size = hidden_size
        self.sigma_floor = sigma_floor

        # Semantic interaction projection:
        # Concatenates: [h_context (H), h_word (H), Delta_h (H), h_context * h_word (H), cos_sim (1)] = 4H + 1
        fusion_in_dim = 4 * hidden_size + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Single unified GaussHead predicting (mu, sigma)
        self.head = GaussHead(
            in_features=hidden_size,
            hidden=head_hidden,
            dropout=dropout,
            floor=sigma_floor,
        )

        # Cache last computed cosine and displacement magnitude for metrics/inspection
        self.last_cos_sim: Optional[torch.Tensor] = None
        self.last_displacement_norm: Optional[torch.Tensor] = None

    @property
    def mod_gauss(self) -> nn.Module:
        return self.head

    @property
    def head_gauss(self) -> nn.Module:
        return self.head

    @property
    def pv_gauss(self) -> nn.Module:
        return self.head

    def pred_heads(self) -> List[nn.Module]:
        """Return the prediction heads and fusion layers for Phase 1 unfreezing."""
        return [self.fusion, self.head]

    def pool_active_context(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked-mean pooling over the target constituent subwords in context.
        
        Args:
            hidden_states: [B, L, H] from Stream 2 (context forward pass).
            target_mask: [B, L] binary indicator of target subwords.
            
        Returns:
            [B, H] in-context constituent representation h_context.
        """
        mask = target_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def forward_stream_word(
        self,
        word_input_ids: torch.Tensor,
        word_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Stream 1: Encode isolated target word/lemma out-of-context.
        
        Returns:
            h_word: [B, H] prototype representation.
        """
        outputs = self.lm(
            input_ids=word_input_ids,
            attention_mask=word_attention_mask,
            return_dict=True,
        )
        return pool_prototype(outputs.last_hidden_state, word_attention_mask)

    def forward_stream_context(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Stream 2: Encode the full context sentence.
        
        Returns:
            h_context: [B, H] in-context span representation.
        """
        outputs = self.lm(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask,
            return_dict=True,
        )
        return self.pool_active_context(outputs.last_hidden_state, target_mask)

    def _forward_pair(
        self,
        h_context: torch.Tensor,
        h_word: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction signals, fuse representations, and predict (mu, sigma)."""
        diff = h_context - h_word
        prod = h_context * h_word
        cos_sim = F.cosine_similarity(h_context, h_word, dim=-1, eps=1e-8).unsqueeze(-1)

        self.last_cos_sim = cos_sim.squeeze(-1)
        self.last_displacement_norm = torch.norm(diff, p=2, dim=-1)

        combined = torch.cat([h_context, h_word, diff, prod, cos_sim], dim=-1)
        fused = self.fusion(combined)
        mu, sigma = self.head(fused)
        return mu, sigma

    def _get_prototype_from_emb(
        self,
        input_ids: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback prototype extraction from lexical embedding lookup."""
        emb = self.lm.get_input_embeddings()(input_ids)
        mask = span_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (emb * mask).sum(dim=1) / denom

    def forward(
        self,
        ctx_input_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        word_input_ids: Optional[torch.Tensor] = None,
        word_attention_mask: Optional[torch.Tensor] = None,
        with_logits: bool = False,
        with_pv: bool = False,
    ):
        """Forward pass supporting both explicit 2-stream arguments and batch dict."""
        # 1. Direct explicit Two-Stream API (Tensors passed as positional args)
        if isinstance(ctx_input_ids, torch.Tensor) and word_input_ids is not None:
            h_word = self.forward_stream_word(word_input_ids, word_attention_mask)
            h_context = self.forward_stream_context(ctx_input_ids, ctx_attention_mask, target_mask)
            mu, sigma = self._forward_pair(h_context, h_word)
            if not self.training:
                mu = mu.clamp(SCORE_MIN, SCORE_MAX)
            return mu, sigma

        # 2. Batch dictionary passed (from DataLoader / train / evaluate harness)
        batch = ctx_input_ids if isinstance(ctx_input_ids, dict) else {}
        if not batch:
            raise ValueError("Expected batch dictionary or explicit tensor arguments to forward()")

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        mod_span_mask = batch.get('mod_span_mask')
        head_span_mask = batch.get('head_span_mask')
        if mod_span_mask is None:
            mod_span_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if head_span_mask is None:
            head_span_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        pv_span_mask = mod_span_mask | head_span_mask

        # Stream 2: Full context sentence encoding
        ctx_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        ctx_hidden = ctx_outputs.last_hidden_state

        h_ctx_mod = self.pool_active_context(ctx_hidden, mod_span_mask)
        h_ctx_head = self.pool_active_context(ctx_hidden, head_span_mask)
        h_ctx_pv = self.pool_active_context(ctx_hidden, pv_span_mask)

        # Single-target mode check (batch has 'target' and 'proto_ids')
        if 'target' in batch and 'proto_ids' in batch:
            proto_ids = batch['proto_ids']
            proto_mask = batch['proto_mask']
            proto_outputs = self.lm(input_ids=proto_ids, attention_mask=proto_mask, return_dict=True)
            h_word = pool_prototype(proto_outputs.last_hidden_state, proto_mask)

            tgt = batch['target']
            # Select active context according to target (0: mod, 1: head, 2: pv)
            h_ctx = torch.where(
                (tgt == 0).unsqueeze(-1),
                h_ctx_mod,
                torch.where((tgt == 1).unsqueeze(-1), h_ctx_head, h_ctx_pv)
            )

            mu, sigma = self._forward_pair(h_ctx, h_word)
            if not self.training:
                mu = mu.clamp(SCORE_MIN, SCORE_MAX)

            mod_pred = head_pred = pv_pred = mu
            mod_sigma = head_sigma = pv_sigma = sigma
        else:
            # Multi-target / Joint mode (score each target with lexical prototypes)
            if 'proto_ids' in batch and 'proto_mask' in batch:
                proto_out = self.lm(input_ids=batch['proto_ids'], attention_mask=batch['proto_mask'], return_dict=True)
                h_word_default = pool_prototype(proto_out.last_hidden_state, batch['proto_mask'])
                h_proto_mod = h_word_default
                h_proto_head = h_word_default
            else:
                h_proto_mod = self._get_prototype_from_emb(input_ids, mod_span_mask)
                h_proto_head = self._get_prototype_from_emb(input_ids, head_span_mask)

            h_proto_pv = 0.5 * (h_proto_mod + h_proto_head)

            mod_mu, mod_sigma = self._forward_pair(h_ctx_mod, h_proto_mod)
            head_mu, head_sigma = self._forward_pair(h_ctx_head, h_proto_head)
            pv_mu, pv_sigma = self._forward_pair(h_ctx_pv, h_proto_pv)

            if not self.training:
                mod_mu = mod_mu.clamp(SCORE_MIN, SCORE_MAX)
                head_mu = head_mu.clamp(SCORE_MIN, SCORE_MAX)
                pv_mu = pv_mu.clamp(SCORE_MIN, SCORE_MAX)

            mod_pred, head_pred, pv_pred = mod_mu, head_mu, pv_mu

        if with_logits:
            return (mod_pred, head_pred, pv_pred, mod_sigma, head_sigma, pv_sigma) if with_pv \
                else (mod_pred, head_pred, mod_sigma, head_sigma)
        if with_pv:
            return mod_pred, head_pred, pv_pred
        return mod_pred, head_pred
