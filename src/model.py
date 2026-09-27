"""Target-Aware Query Cross-Attention & Component Interaction Model (NC & PV).

Architecture:
  - Layer 0: Encoder (mmBERT) & Explicit Role Injection (E_role)
             Role IDs: 0: Context, 1: Mod / Base Verb, 2: Head / Particle
             H_final = H_mmBERT + E_role  [B, S, 768]
  - Layer 1: Target-Aware Multi-Head Cross-Attention (MHCA)
             Queries: Q_base = [q_Slot0, q_Slot1, q_Slot2] in [3, 768]
             Attention map: A = Softmax(Q * K^T / sqrt(d)) in [B, 3, S]
             Z = LayerNorm(Q + Dropout(MHCA(Q, K, V))) in [B, 3, 768]
  - Layer 2: Component Interaction Layer (MHSA)
             Z' = LayerNorm(Z + Dropout(MHSA(Z))) in [B, 3, 768]
  - Layer 3: Shared GaussHead Regression
             Slot 0 (Mod/Verb): (mu_0, sigma_0), Slot 1 (Head/Part): (mu_1, sigma_1), Slot 2 (Expr): (mu_2, sigma_2)
             Continuous prediction: Score = mu * 5.0
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
from .model_query import (
    ComponentInteractionLayer,
    TargetAwareCrossAttention,
    TargetAwareQueryAttentionModel,
)


def build_model(cfg, device, load_from: Optional[str | Path] = None) -> nn.Module:
    """Build the Target-Aware Query Attention model."""
    model = TargetAwareQueryAttentionModel(
        backbone=cfg.backbone,
        hidden_size=cfg.hidden_size,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        shared_head=bool(getattr(cfg, 'shared_head', True)),
        sigma_floor=float(getattr(cfg, 'sigma_floor', 0.04)),
        use_pre_ln=bool(getattr(cfg, 'use_pre_ln', True)),
        use_rms_norm=bool(getattr(cfg, 'use_rms_norm', True)),
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        if any(k.startswith('lm.') for k in state.keys()):
            model.load_state_dict(state, strict=False)
        else:
            model.lm.load_state_dict(state, strict=False)
    return model.to(device)


__all__ = [
    'TargetAwareCrossAttention',
    'ComponentInteractionLayer',
    'TargetAwareQueryAttentionModel',
    'build_model',
]
