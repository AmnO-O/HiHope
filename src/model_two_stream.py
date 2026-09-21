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
from .prototype_stream import (
    SemanticShiftFusion,
    corrected_cosine_similarity,
    pool_prototype,
)


class LearnedLayerWeights(nn.Module):
    """Softmax-normalized scalar layer weighting across extracted hidden layers."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, stacked_hidden: torch.Tensor) -> torch.Tensor:
        """Args: stacked_hidden of shape [K, B, L, H]. Returns [B, L, H]."""
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (stacked_hidden * w).sum(dim=0)


class TwoStreamBiEncoderModel(nn.Module):
    """Two-Stream Bi-Encoder for Compositionality and Uncertainty Prediction."""

    def __init__(
        self,
        backbone: str = "jhu-clsp/mmBERT-base",
        hidden_size: int = 768,
        head_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 0.05,
        fusion_type: str = "cross_attention",
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        extract_layers: Optional[Tuple[int, ...]] = (14, 15, 16, 17, 18),
        extract_mode: str = "mean",
    ):
        super().__init__()
        self.lm = AutoModel.from_pretrained(backbone)
        self.hidden_size = hidden_size
        self.sigma_floor = sigma_floor
        self.fusion_type = fusion_type
        self.extract_layers = extract_layers
        self.extract_mode = extract_mode

        if extract_mode == "learned" and extract_layers:
            self.layer_agg = LearnedLayerWeights(len(extract_layers))
        else:
            self.layer_agg = None

        # Fusion: Cross-Attention Semantic Shift Transformer or Linear projection
        if fusion_type == "cross_attention":
            self.fusion = SemanticShiftFusion(
                hidden_size=hidden_size,
                num_layers=fusion_layers,
                num_heads=fusion_heads,
                dropout=dropout,
            )
        else:
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
    def shift_fuse(self) -> nn.Module:
        return self.fusion

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
        """Return the prediction heads, fusion layers, and learned layer weights for Phase 1 unfreezing."""
        modules = [self.fusion, self.head]
        if self.layer_agg is not None:
            modules.append(self.layer_agg)
        return modules

    def _extract_hidden(self, outputs) -> torch.Tensor:
        """Extract hidden states from configured layers (e.g. upper-middle layers 14-18).
        
        If extract_layers is specified and hidden_states are available, combines
        the selected layer representations (via learned weights or uniform mean).
        Otherwise falls back to last_hidden_state.
        """
        if self.extract_layers and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # outputs.hidden_states has 23 entries for a 22-layer model (idx 0 is embedding)
            selected = [outputs.hidden_states[idx] for idx in self.extract_layers if idx < len(outputs.hidden_states)]
            if selected:
                stacked = torch.stack(selected, dim=0)
                if self.layer_agg is not None:
                    return self.layer_agg(stacked)
                return stacked.mean(dim=0)
        return outputs.last_hidden_state

    def pool_active_context(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        fallback_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked-mean pooling over the target constituent subwords in context.

        Rows whose target span could not be aligned (all-false ``target_mask``)
        fall back to whole-sentence pooling over ``fallback_mask`` (the context
        ``attention_mask``) instead of an empty mean, so an unaligned row still
        contributes its overall-context representation rather than being
        silently dropped.

        Args:
            hidden_states: [B, L, H] from Stream 2 (context forward pass).
            target_mask: [B, L] binary indicator of target subwords.
            fallback_mask: [B, L] binary mask used when target_mask is empty.

        Returns:
            [B, H] in-context constituent representation h_context.
        """
        mask = target_mask.unsqueeze(-1).float()
        has_span = mask.sum(dim=1) > 0.0
        if fallback_mask is not None:
            fb = fallback_mask.unsqueeze(-1).float()
            mask = torch.where(
                has_span.unsqueeze(-1),
                mask,
                fb,
            )
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
            output_hidden_states=bool(self.extract_layers),
            return_dict=True,
        )
        hidden = self._extract_hidden(outputs)
        return pool_prototype(hidden, word_attention_mask)

    def forward_stream_context(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        fallback_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stream 2: Encode the full context sentence.
        
        Returns:
            h_context: [B, H] in-context span representation.
        """
        outputs = self.lm(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask,
            output_hidden_states=bool(self.extract_layers),
            return_dict=True,
        )
        hidden = self._extract_hidden(outputs)
        return self.pool_active_context(hidden, target_mask, fallback_mask)

    def _forward_pair(
        self,
        h_context: torch.Tensor,
        h_word: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction signals, fuse representations, and predict (mu, sigma)."""
        diff = h_context - h_word
        cos_sim = corrected_cosine_similarity(h_context, h_word, eps=1e-8).unsqueeze(-1)

        self.last_cos_sim = cos_sim.squeeze(-1)
        self.last_displacement_norm = torch.norm(diff, p=2, dim=-1)

        if isinstance(self.fusion, SemanticShiftFusion):
            fused = self.fusion(h_context, h_word)
        else:
            prod = h_context * h_word
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
            h_context = self.forward_stream_context(
                ctx_input_ids, ctx_attention_mask, target_mask, ctx_attention_mask)
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
        ctx_outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=bool(self.extract_layers),
            return_dict=True,
        )
        ctx_hidden = self._extract_hidden(ctx_outputs)

        h_ctx_mod = self.pool_active_context(ctx_hidden, mod_span_mask, attention_mask)
        h_ctx_head = self.pool_active_context(ctx_hidden, head_span_mask, attention_mask)
        h_ctx_pv = self.pool_active_context(ctx_hidden, pv_span_mask, attention_mask)

        # Single-target mode check (batch has 'target' and 'proto_ids')
        if 'target' in batch and 'proto_ids' in batch:
            proto_ids = batch['proto_ids']
            proto_mask = batch['proto_mask']
            proto_outputs = self.lm(
                input_ids=proto_ids,
                attention_mask=proto_mask,
                output_hidden_states=bool(self.extract_layers),
                return_dict=True,
            )
            h_word = pool_prototype(self._extract_hidden(proto_outputs), proto_mask)

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
            if 'mod_proto_ids' in batch and 'head_proto_ids' in batch:
                proto_mod_out = self.lm(
                    input_ids=batch['mod_proto_ids'],
                    attention_mask=batch['mod_proto_mask'],
                    output_hidden_states=bool(self.extract_layers),
                    return_dict=True,
                )
                h_proto_mod = pool_prototype(self._extract_hidden(proto_mod_out), batch['mod_proto_mask'])

                proto_head_out = self.lm(
                    input_ids=batch['head_proto_ids'],
                    attention_mask=batch['head_proto_mask'],
                    output_hidden_states=bool(self.extract_layers),
                    return_dict=True,
                )
                h_proto_head = pool_prototype(self._extract_hidden(proto_head_out), batch['head_proto_mask'])

                # For PV rows, proto_ids encodes the full compound verb (e.g. "abziehen" or "give up")
                if 'proto_ids' in batch and 'proto_mask' in batch:
                    proto_pv_out = self.lm(
                        input_ids=batch['proto_ids'],
                        attention_mask=batch['proto_mask'],
                        output_hidden_states=bool(self.extract_layers),
                        return_dict=True,
                    )
                    h_proto_pv = pool_prototype(self._extract_hidden(proto_pv_out), batch['proto_mask'])
                else:
                    h_proto_pv = 0.5 * (h_proto_mod + h_proto_head)
            elif 'proto_ids' in batch and 'proto_mask' in batch:
                proto_out = self.lm(
                    input_ids=batch['proto_ids'],
                    attention_mask=batch['proto_mask'],
                    output_hidden_states=bool(self.extract_layers),
                    return_dict=True,
                )
                h_word_default = pool_prototype(self._extract_hidden(proto_out), batch['proto_mask'])
                h_proto_mod = h_word_default
                h_proto_head = h_word_default
                h_proto_pv = h_word_default
            else:
                h_proto_mod = self._get_prototype_from_emb(input_ids, mod_span_mask)
                h_proto_head = self._get_prototype_from_emb(input_ids, head_span_mask)
                h_proto_pv = 0.5 * (h_proto_mod + h_proto_head)

            mod_mu, mod_sigma = self._forward_pair(h_ctx_mod, h_proto_mod)
            self.last_mod_cos = getattr(self, 'last_cos_sim', None)

            head_mu, head_sigma = self._forward_pair(h_ctx_head, h_proto_head)
            self.last_head_cos = getattr(self, 'last_cos_sim', None)

            pv_mu, pv_sigma = self._forward_pair(h_ctx_pv, h_proto_pv)
            self.last_pv_cos = getattr(self, 'last_cos_sim', None)

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
