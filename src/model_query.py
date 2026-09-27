"""Target-Aware Query Attention & Component Interaction Architecture.

End-to-End Pipeline for Noun Compound (NC) & Particle Verb (PV) Compositionality Rating (1.0 -> 5.0):

[INPUT SEQUENCE - DUAL MODALITY SUPPORT]
Case A: Noun Compound (Adjacent or Fused)
  Context:         "This was soon thrown out through the back door, never to be seen again."
  Target Metadata: Mod (Slot 0) = "back", Head (Slot 1) = "door", Compound (Slot 2) = "back door"

Case B: Particle Verb (Continuous or Discontinuous/Separable in En/De)
  Context (En):    "He refused to give up on reaching the summit."
  Target Metadata: Base Verb (Slot 0) = "give", Particle (Slot 1) = "up", Expression (Slot 2) = "give up"
  Context (De):    "Er haute sofort durch die Hintertür ab."
  Target Metadata: Base Verb (Slot 0) = "haute", Particle (Slot 1) = "ab", Expression (Slot 2) = "abhauen"
                                    |
                                    v
+-------------------------------------------------------------------------------+
| LAYER 0: ENCODER & EXPLICIT ROLE INJECTION                                    |
|                                                                               |
|  Input Token IDs  [B, S] ------> mmBERT Encoder ----> H_mmBERT  [B, S, 768]   |
|  Role IDs         [B, S] ------> Role Embeddings ---> E_role    [B, S, 768]   |
|  (0: Context, 1: Mod / Base Verb, 2: Head / Particle)                         |
|                               H_final = H_mmBERT + E_role ---+-> [B, S, 768]  |
+--------------------------------------|----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| LAYER 1: TARGET-AWARE MULTI-HEAD CROSS-ATTENTION (MHCA)                       |
|                                                                               |
|  Learned Queries: Q_base = [q_Slot0, q_Slot1, q_Slot2] in [3, 768]            |
|  (Slot 0: Mod/Verb, Slot 1: Head/Particle, Slot 2: Whole Compound/PV)         |
|  Batch Expand  --------> Q in [B, 3, 768]                                     |
|                                                                               |
|  Queries (Q)  = Q               in [B, 3, 768]                                 |
|  Keys (K)     = H_final . W_k   in [B, S, 768]                                 |
|  Values (V)   = H_final . W_v   in [B, S, 768]                                 |
|                                                                               |
|  A = Softmax(Q.K^T / sqrt(d_h)) in [B, 3, S]  (Interpretability Attention Map)|
|  Z_attn = MultiHeadAttention(Q, K, V)                                         |
|  Z = LayerNorm(Q + Dropout(Z_attn))                                           |
|  --> Tensor Z in [B, 3, 768] (Context-Conditioned Target Representations)     |
+--------------------------------------|----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| LAYER 2: COMPONENT INTERACTION LAYER (MHSA)                                   |
|                                                                               |
|  Self-Attention between constituents (Mod/Verb <-> Head/Part <-> Whole Expr): |
|  Z_self = MultiHeadSelfAttention(Q=Z, K=Z, V=Z)                               |
|  Z' = LayerNorm(Z + Dropout(Z_self))                                          |
|  --> Tensor Z' in [B, 3, 768] (Boundary-Aware Compositional Representations)  |
+--------------------------------------|----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| LAYER 3: GAUSSHEAD REGRESSION & CONTINUOUS PREDICTION                         |
|                                                                               |
|  z'_Slot0 [B, 768] --+                                                        |
|  z'_Slot1 [B, 768] --+--> Shared GaussHead --> Slot 0: (mu_0, sigma_0)        |
|  z'_Slot2 [B, 768] --+      (Shared MLP)       Slot 1: (mu_1, sigma_1)        |
|                                                Slot 2: (mu_2, sigma_2)        |
|  (NC: Mod, Head, Comp | PV: Base Verb, Particle, Whole Phrasal Verb)          |
+--------------------------------------|----------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| CCC LOSS & TARGET NORMALIZATION                                               |
|                                                                               |
|  Targets normalized: y_norm = y_raw / 5.0 in [0.2, 1.0]                       |
|  Loss = L_CCC(mu_1, y_1_norm) + L_CCC(mu_0, y_0_norm)                         |
|  Continuous Prediction: Score = mu * 5.0                                      |
+-------------------------------------------------------------------------------+
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .constants import SCORE_MAX, SCORE_MIN
from .heads import GaussHead


class TargetAwareCrossAttention(nn.Module):
    """Layer 1: Target-Aware Multi-Head Cross-Attention (MHCA).
    
    Probes in-context representation H_final using 3 learned query vectors
    [q_Mod, q_Head, q_Compound]. Produces interpretability attention map A in [B, 3, S].
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 3 Learned Probing Queries: [q_Mod, q_Head, q_Compound]
        self.q_base = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.q_base, std=0.02)
        
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        h_final: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_final: [B, S, H] encoder output with role injection
            attention_mask: [B, S] boolean or 0/1 mask (1 for valid token, 0 for pad)
        Returns:
            Z: [B, 3, H] context-conditioned target representations
            attn_weights: [B, 3, S] attention map over sequence tokens
        """
        b_size = h_final.size(0)
        # Expand learned queries for the batch: [B, 3, H]
        q = self.q_base.unsqueeze(0).expand(b_size, -1, -1)

        # MultiheadAttention expects key_padding_mask as True for padded positions
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Multi-Head Cross-Attention
        z_attn, attn_weights = self.mha(
            query=q,
            key=h_final,
            value=h_final,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # [B, 3, S]
        )

        # Residual connection + LayerNorm: Z = LayerNorm(Q + Dropout(Z_attn))
        z = self.norm(q + self.dropout(z_attn))
        return z, attn_weights


class ComponentInteractionLayer(nn.Module):
    """Layer 2: Component Interaction Layer (MHSA).
    
    Performs Multi-Head Self-Attention among the 3 target representations
    (Mod <-> Head <-> Compound) to capture compositional and boundary interactions.
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, 3, H] from Layer 1
        Returns:
            z_prime: [B, 3, H] updated compositional representations
            self_attn_weights: [B, 3, 3] interaction weights
        """
        z_self, self_attn_weights = self.self_mha(
            query=z,
            key=z,
            value=z,
            need_weights=True,
            average_attn_weights=True,
        )
        z_prime = self.norm(z + self.dropout(z_self))
        return z_prime, self_attn_weights


class TargetAwareQueryAttentionModel(nn.Module):
    """Full End-to-End Pipeline for Compositionality Rating via Target-Aware Query Cross-Attention."""

    def __init__(
        self,
        backbone: str = "jhu-clsp/mmBERT-base",
        hidden_size: int = 768,
        head_hidden: int = 128,
        dropout: float = 0.1,
        num_roles: int = 4,  # 0: Context, 1: Mod, 2: Head, 3: Compound/PV
        num_cross_heads: int = 8,
        num_self_heads: int = 4,
        shared_head: bool = True,
        sigma_floor: float = 0.04,
        scale_target_5x: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.hidden_size = hidden_size
        self.scale_target_5x = scale_target_5x
        self.sigma_floor = sigma_floor
        self.shared_head = shared_head

        # Layer 0: Encoder & Role Embeddings
        self.lm = AutoModel.from_pretrained(backbone)
        self.role_embeddings = nn.Embedding(num_roles, hidden_size)
        # Small initialization for role embeddings so they blend smoothly with mmBERT
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)

        # Layer 1: Target-Aware Multi-Head Cross-Attention
        self.cross_attn = TargetAwareCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_cross_heads,
            dropout=dropout,
        )

        # Layer 2: Component Interaction Layer
        self.component_interaction = ComponentInteractionLayer(
            hidden_size=hidden_size,
            num_heads=num_self_heads,
            dropout=dropout,
        )

        # Layer 3: GaussHead Regression
        if shared_head:
            self._head = GaussHead(
                in_features=hidden_size,
                hidden=head_hidden,
                dropout=dropout,
                floor=sigma_floor,
            )
        else:
            self._head_mod = GaussHead(in_features=hidden_size, hidden=head_hidden, dropout=dropout, floor=sigma_floor)
            self._head_head = GaussHead(in_features=hidden_size, hidden=head_hidden, dropout=dropout, floor=sigma_floor)
            self._head_comp = GaussHead(in_features=hidden_size, hidden=head_hidden, dropout=dropout, floor=sigma_floor)

        # Interpretability caches
        self.last_cross_attn_map: Optional[torch.Tensor] = None
        self.last_component_attn_map: Optional[torch.Tensor] = None
        self.last_slot_representations: Optional[torch.Tensor] = None

    @property
    def mod_gauss(self) -> GaussHead:
        return self._head if self.shared_head else self._head_mod

    @property
    def head_gauss(self) -> GaussHead:
        return self._head if self.shared_head else self._head_head

    @property
    def comp_gauss(self) -> GaussHead:
        return self._head if self.shared_head else self._head_comp

    def pred_heads(self) -> List[nn.Module]:
        """Return non-backbone modules for phase 1 unfreezing."""
        modules = [
            self.role_embeddings,
            self.cross_attn,
            self.component_interaction,
        ]
        if self.shared_head:
            modules.append(self._head)
        else:
            modules.extend([self._head_mod, self._head_head, self._head_comp])
        return modules

    def _build_role_ids(
        self,
        input_ids: torch.Tensor,
        mod_span_mask: Optional[torch.Tensor] = None,
        head_span_mask: Optional[torch.Tensor] = None,
        compound_span_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct role IDs tensor [B, S]: 0: Context, 1: Mod, 2: Head, 3: Compound."""
        role_ids = torch.zeros_like(input_ids, dtype=torch.long)
        if mod_span_mask is not None:
            role_ids = torch.where(mod_span_mask.bool(), torch.full_like(role_ids, 1), role_ids)
        if head_span_mask is not None:
            role_ids = torch.where(head_span_mask.bool(), torch.full_like(role_ids, 2), role_ids)
        if compound_span_mask is not None:
            role_ids = torch.where(compound_span_mask.bool(), torch.full_like(role_ids, 3), role_ids)
        return role_ids

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mod_span_mask: Optional[torch.Tensor] = None,
        head_span_mask: Optional[torch.Tensor] = None,
        compound_span_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Layers 0, 1, and 2.
        Returns:
            z_prime: [B, 3, H] (Boundary-Aware Compositional Representations)
            cross_attn_map: [B, 3, S] (Layer 1 Attention Map)
            comp_attn_map: [B, 3, 3] (Layer 2 Component Interaction Map)
        """
        # --- LAYER 0: ENCODER & EXPLICIT ROLE INJECTION ---
        encoder_out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        h_mmbert = encoder_out.last_hidden_state  # [B, S, 768]

        role_ids = self._build_role_ids(
            input_ids=input_ids,
            mod_span_mask=mod_span_mask,
            head_span_mask=head_span_mask,
            compound_span_mask=compound_span_mask,
        )
        e_role = self.role_embeddings(role_ids)    # [B, S, 768]
        h_final = h_mmbert + e_role                # [B, S, 768]

        # --- LAYER 1: TARGET-AWARE MULTI-HEAD CROSS-ATTENTION ---
        z, cross_attn_map = self.cross_attn(h_final, attention_mask=attention_mask)  # [B, 3, 768], [B, 3, S]

        # --- LAYER 2: COMPONENT INTERACTION LAYER ---
        z_prime, comp_attn_map = self.component_interaction(z)  # [B, 3, 768], [B, 3, 3]

        # Cache for interpretability / visualizers
        self.last_cross_attn_map = cross_attn_map.detach()
        self.last_component_attn_map = comp_attn_map.detach()
        self.last_slot_representations = z_prime.detach()

        return z_prime, cross_attn_map, comp_attn_map

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        with_logits: bool = False,
        with_pv: bool = False,
    ):
        """
        Forward pass predicting continuous Gaussian distributions for Mod, Head, and Compound.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        mod_span_mask = batch.get('mod_span_mask')
        head_span_mask = batch.get('head_span_mask')
        compound_span_mask = batch.get('compound_span_mask')

        z_prime, _, _ = self.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mod_span_mask=mod_span_mask,
            head_span_mask=head_span_mask,
            compound_span_mask=compound_span_mask,
        )

        # Slot 0: Mod, Slot 1: Head, Slot 2: Compound / Phrasal Verb
        z_mod = z_prime[:, 0, :]   # [B, 768]
        z_head = z_prime[:, 1, :]  # [B, 768]
        z_comp = z_prime[:, 2, :]  # [B, 768]

        # --- LAYER 3: GAUSSHEAD REGRESSION ---
        mod_mu, mod_sigma = self.mod_gauss(z_mod)
        head_mu, head_sigma = self.head_gauss(z_head)
        comp_mu, comp_sigma = self.comp_gauss(z_comp)

        # Scale or clamp predictions
        if self.training:
            mod_pred, head_pred, comp_pred = mod_mu, head_mu, comp_mu
        else:
            mod_pred = mod_mu.clamp(SCORE_MIN, SCORE_MAX)
            head_pred = head_mu.clamp(SCORE_MIN, SCORE_MAX)
            comp_pred = comp_mu.clamp(SCORE_MIN, SCORE_MAX)

        if with_logits:
            return (
                (mod_pred, head_pred, comp_pred, mod_sigma, head_sigma, comp_sigma)
                if with_pv
                else (mod_pred, head_pred, mod_sigma, head_sigma)
            )

        if with_pv:
            return mod_pred, head_pred, comp_pred
        return mod_pred, head_pred
