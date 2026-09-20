"""Two-Stream Bi-Encoder Architecture for Compositionality Assessment.

This module implements the user's core architecture:
    Stream 1 (Target Word):   [CLS] target_word [SEP]       -> h_word (out-of-context prototype)
    Stream 2 (Context):       [CLS] context_sentence [SEP] -> h_context (in-context span representation)

Interaction & Fusion:
    1. Directional semantic displacement: Delta_h = h_context - h_word
    2. Element-wise multiplicative interaction: h_context * h_word
    3. Calibrated cosine similarity: cos(h_context, h_word)
    4. Lightweight semantic fusion layer -> GaussHead -> (mu, sigma)

No LoRA is required: mmBERT-base (~110M parameters) is trained with standard full
fine-tuning or top-layer unfreezing, providing stable and direct gradient propagation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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
            hidden_dim=head_hidden,
            dropout=dropout,
            sigma_floor=sigma_floor,
        )

        # Cache last computed cosine and displacement magnitude for metrics/inspection
        self.last_cos_sim: Optional[torch.Tensor] = None
        self.last_displacement_norm: Optional[torch.Tensor] = None

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
        # Pool lexical subwords excluding [CLS] and [SEP]
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

    def forward(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        word_input_ids: torch.Tensor,
        word_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complete Two-Stream Bi-Encoder forward pass.
        
        Args:
            ctx_input_ids: Context token IDs [B, L_ctx]
            ctx_attention_mask: Context attention mask [B, L_ctx]
            target_mask: Binary mask of target word inside context [B, L_ctx]
            word_input_ids: Isolated target word token IDs [B, L_word]
            word_attention_mask: Isolated target word attention mask [B, L_word]
            
        Returns:
            mu: Predicted compositionality mean rating [B, 1]
            sigma: Predicted annotator disagreement std [B, 1]
        """
        # Stream 1: h_word out of context
        h_word = self.forward_stream_word(word_input_ids, word_attention_mask)  # [B, H]

        # Stream 2: h_context in sentence context
        h_context = self.forward_stream_context(ctx_input_ids, ctx_attention_mask, target_mask)  # [B, H]

        # Interaction signals
        diff = h_context - h_word  # Semantic displacement Delta_h [B, H]
        prod = h_context * h_word  # Element-wise interaction [B, H]
        cos_sim = F.cosine_similarity(h_context, h_word, dim=-1, eps=1e-8).unsqueeze(-1)  # [B, 1]

        self.last_cos_sim = cos_sim.squeeze(-1)
        self.last_displacement_norm = torch.norm(diff, p=2, dim=-1)

        # Fused semantic representation: [B, 4H + 1] -> [B, H]
        combined = torch.cat([h_context, h_word, diff, prod, cos_sim], dim=-1)
        fused = self.fusion(combined)

        # Output calibrated Gaussian distribution
        mu, sigma = self.head(fused)

        if not self.training:
            mu = mu.clamp(SCORE_MIN, SCORE_MAX)

        return mu, sigma
