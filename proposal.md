# Proposal: Within-Exit Partitioned Contrastive Regularization (WEP-InfoNCE)

**Target Problem**: Eliminating the Cross-Exit Shortcut & Aligning Metric Geometry for Multi-Task Compositionality & Uncertainty Prediction  
**Author**: Antigravity Research / AI Studio  
**Date**: September 2026  
**Status**: Revised Specification (Incorporating Reviewer Critiques & Uncertainty Analysis)

---

## 1. Executive Summary & Diagnostic Verification

The review confirms the fundamental diagnostic with mathematical and empirical rigor:

> *"Never run the matrix across mixed exits. You already measured the trap: pooled $\rho = 0.42$ vs per-split $\approx 0$. The batch mixes mod/head/pv rows, so off-diagonal negatives become cross-exit pairs (mostly trivial) and the matrix learns exit-identity geometry, not compositionality. Compute InfoNCE per exit-group (mask the matrix to NN rows / same exit), or exclude PV rows from the matrix entirely."*

### Why Cross-Exit InfoNCE Collapses into a Shortcut
1. **Trivial Syntactic Separation**: In an unconstrained $B \times B$ batch, a Noun Modifier (`mod`, e.g. *"crocodile"*) is paired with off-diagonal negatives from Head Nouns (`head`) and Particle Verbs (`pv`, e.g. *"give up"*). For a transformer, separating a noun from a phrasal verb is trivial due to Part-of-Speech and syntactic divergence.
2. **Simpson's Paradox**: The optimizer easily minimizes InfoNCE by clustering exits apart rather than learning compositionality. Global pooled correlation jumps to $\rho \approx 0.42$ solely from cluster-centroid offsets, while within-split correlations stay flat ($\rho_{\text{mod}} \approx 0$).
3. **Solution**: **Within-Exit Partitioned InfoNCE (WEP-InfoNCE)** with block-diagonal masking and PV exclusion.

---

## 2. Addressing Annotator Uncertainty (`std`): Do We Use It in InfoNCE?

### The Question
> *"We didn't add the std right? Do we use it?"*

### Analysis & Recommendation
In compositionality datasets (e.g. REDAC-T, German Comp), each sample has a human mean rating $\mu_{\text{gold}}$ and an annotator standard deviation $\sigma_{\text{gold}}$ (disagreement among annotators).

1. **Where `std` belongs (Primary Role)**:
   `std` is already explicitly modeled by the **`GaussHead`** via negative log-likelihood:
   $$\mathcal{L}_{\text{Gauss}} = \frac{(\mu_{\text{pred}} - y_{\text{gold}})^2}{2 \sigma_{\text{pred}}^2} + \log \sigma_{\text{pred}}$$
   The Gaussian head is specifically tasked with learning aleatoric annotator uncertainty $\sigma$.

2. **Should `std` be used inside InfoNCE?**:
   **Option A (Pure Mean - Recommended Baseline)**: Do not inject $\sigma_{\text{gold}}$ directly into the vector metric space. InfoNCE aligns the expected semantic vectors ($z_{\text{ctx}}$ and $z_{\text{proto}}$). The vector space represents the *point geometry* of the word meaning.
   
   **Option B (Uncertainty-Aware Consensus Attenuation - Advanced Extension)**:
   When annotators strongly disagree (high $\sigma_{\text{gold}}$, e.g. $\sigma > 1.2$), the sample is semantically ambiguous (some annotators saw it as literal, others as idiomatic). Forcing an ambiguous sample to have a rigid contrastive target can induce gradient noise.
   We can apply an optional **Consensus Weighting**:
   $$c_i = \exp\left(-\frac{\sigma_{\text{gold}, i}^2}{2 \sigma_0^2}\right) \quad \text{or} \quad c_i = \frac{1}{1 + \sigma_{\text{gold}, i}}$$
   The final contrastive weight becomes $w_i = c_i \cdot \tilde{y}_i$.
   - High consensus literal ($y=4.8, \sigma=0.3 \implies w_i \approx 1.0$): strong attraction.
   - High controversy sample ($y=3.0, \sigma=1.8 \implies w_i \approx 0.3$): softly regularized, leaving ambiguity modeling to `GaussHead`.

**Conclusion for V1**: Keep InfoNCE focused on mean literalness with a clean `has_label` mask; provide an optional flag `use_std_attenuation: bool = False` to avoid coupling uncertainty noise into metric initialization.

---

## 3. Detailed Fixes to Reviewer Critiques

### Critique 1: Success Metric Tensor ($\cos(h)$ vs $\cos(z)$)
- **Problem**: Existing `[Diag]` lines log raw pre-projector $\cos(h_{\text{ctx}}, h_{\text{proto}})$. WEP trains the projector output $z$. The projector can succeed in organizing $z$-space while raw $h$-space remains unchanged, causing the logger to miss progress.
- **Fix**: The validation engine must compute and log **`cos_z-vs-gold ρ`** (`last_mod_cos_z`, `last_head_cos_z`) alongside the baseline raw cosine.

### Critique 2: Empirical Baseline Numbers in §6
- **Correction**: Replace hypothetical numbers with actual logged baselines from your runs:
  - Raw pre-projector cos-vs-gold $\rho$:
    - Modifier (`mod`): **$+0.177$**
    - Head (`head`): **$-0.093$**
    - Particle Verb (`pv`): **$+0.197$**
  - Targets for projected similarity $\cos(z_{\text{ctx}}, z_{\text{proto}})$:
    - $\rho(z)_{\text{mod}} \ge 0.35$
    - $\rho(z)_{\text{head}} \ge 0.35$ (recovering from negative correlation)

### Critique 3: Weight Formula Inconsistency & Missing Guards
- **Problem**: §3 had $(y - y_{\min})/(y_{\max} - y_{\min})$ while snippet had `rating / 5`, with no per-exit normalization and no `has_label` / NaN check.
- **Fix**: Standardize on exit-isolated, min-max normalized weights with a strict boolean label guard:
  ```python
  # Per-exit min-max normalization with label guard
  valid = has_label & torch.isfinite(ratings)
  # For exit e in {mod, head}:
  y_e = ratings[valid & (exit_ids == e)]
  w_e = (y_e - y_min_e) / (y_max_e - y_min_e + 1e-6)
  ```

### Critique 4: Integration Gap (Exporting $h$ / $z$ to Trainer)
- **Problem**: `train_epoch` only received `(mu, sigma)` and `last_*_cos`. WEP requires gradient-bearing representations.
- **Fix**: Add a flag-gated feature export in `TwoStreamBiEncoderModel`:
  ```python
  # In TwoStreamBiEncoderModel.forward:
  if self.use_wep_infonce:
      self.last_z_ctx_mod = z_ctx_mod
      self.last_z_proto_mod = z_proto_mod
      self.last_z_ctx_head = z_ctx_head
      self.last_z_proto_head = z_proto_head
  ```
  In `train.py`, place `within_exit_infonce_loss` right next to line 147.

### Critique 5: Loss Competition (`proto_rank_loss` vs `WEP-InfoNCE`)
- **Problem**: Both losses penalize the same alignment axis. Running `proto_rank_loss` at $\lambda=0.15$ alongside WEP at $\lambda=0.08$ produces competing gradient directions.
- **Fix**: When `use_wep_infonce=True`, set `proto_rank_weight = 0.0`. WEP subsumes and strictly supersedes the pairwise ranking loss.

### Additional Adjustments:
- **Temperature $\tau$**: Set default $\tau = 0.10$ (configurable via `cfg.wep_tau`) rather than over-aggressive $0.07$.
- **L2 Normalization**: Enforce `F.normalize(..., p=2, dim=-1)` inside the loss function defensively.
- **Backbone Gradients**: Clarify that Phase 1 trains `W_proj`; Phase 2 propagates gradients through `W_proj` to Layer 19 via LoRA or `unfreeze_from_layer`.

---

## 4. Architectural Implementation & Shared Projector

```text
Stream 1 (In-Context Sentence)
  BERT Layer 19 -> Span Pool -> h_ctx (768d) ──► [ Shared Projector W_proj ] ──► z_ctx (768d)
                                                                                       │
                                                                           F.normalize(p=2)
                                                                                       │
                                                                                       ▼
                                                                           Exit-Masked Sim Matrix
                                                                           (B_NN x B_NN, Tau = 0.10)
                                                                                       ▲
                                                                                       │
                                                                           F.normalize(p=2)
                                                                                       │
Stream 2 (Prototype Prompt)                                                            │
  BERT Layer 19 -> Word Pool -> h_proto (768d) ─► [ Shared Projector W_proj ] ──► z_proto (768d)
```

The projector is a 2-layer MLP with residual connection or bottleneck:
```python
class SharedProjector(nn.Module):
    def __init__(self, hidden_size: int = 768, proj_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))
```

---

## 5. Algorithmic Formulation: `within_exit_infonce_loss`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def within_exit_infonce_loss(
    z_ctx: torch.Tensor,         # [B, D] context embeddings (unnormalized or normalized)
    z_proto: torch.Tensor,       # [B, D] prototype embeddings
    ratings: torch.Tensor,       # [B] human compositionality ratings
    has_label: torch.Tensor,     # [B] boolean mask of valid ground-truth labels
    exit_ids: torch.Tensor,      # [B] exit indicator: 0=mod, 1=head, 2=pv
    tau: float = 0.10,
    exclude_pv: bool = True,
    std_ratings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes Within-Exit Partitioned InfoNCE with continuous rating weights and label guards."""
    device = z_ctx.device
    
    # 1. Defensive normalization
    z_ctx = F.normalize(z_ctx, p=2, dim=-1)
    z_proto = F.normalize(z_proto, p=2, dim=-1)

    # 2. Strict validity filtering (valid label, finite, non-PV if requested)
    valid = has_label.bool() & torch.isfinite(ratings)
    if exclude_pv:
        valid = valid & (exit_ids != 2)

    if valid.sum() < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    z_ctx = z_ctx[valid]
    z_proto = z_proto[valid]
    ratings = ratings[valid]
    exit_ids = exit_ids[valid]
    if std_ratings is not None:
        std_ratings = std_ratings[valid]

    B = z_ctx.size(0)

    # 3. Normalized weights per exit group
    weights = torch.zeros(B, device=device)
    for e in (0, 1) if exclude_pv else (0, 1, 2):
        mask_e = (exit_ids == e)
        if mask_e.any():
            r_e = ratings[mask_e]
            r_min = r_e.min()
            r_max = r_e.max()
            denom = (r_max - r_min).clamp(min=1e-5)
            w_e = (r_e - r_min) / denom
            if std_ratings is not None:
                # Optional consensus attenuation: downweight high annotator variance
                consensus = 1.0 / (1.0 + std_ratings[mask_e])
                w_e = w_e * consensus
            weights[mask_e] = w_e

    # 4. Pairwise similarity matrix
    sim = torch.matmul(z_ctx, z_proto.T) / tau  # [B, B]

    # 5. Block-diagonal exit mask: M[i, j] = 1 if exit_ids[i] == exit_ids[j]
    exit_mask = (exit_ids.unsqueeze(1) == exit_ids.unsqueeze(0)).float()

    # Mask off-diagonal cross-exit negatives with -1e9 (applied AFTER / tau)
    masked_sim = sim.masked_fill(exit_mask == 0.0, -1e9)

    # 6. Numerator & Denominator
    pos_sim = torch.diagonal(sim)
    log_denom = torch.logsumexp(masked_sim, dim=1)

    # 7. Final loss
    loss = - (weights * pos_sim - log_denom).mean()
    return loss
```

---

## 6. Empirical Validation & Real Baseline Targets

To accurately capture the impact of WEP-InfoNCE, metrics must distinguish between raw $h$ representations and learned $z$ projections:

| Metric | Real Starting Baseline (Raw $\cos(h)$) | Target Success Threshold (Projected $\cos(z)$) | Description |
| :--- | :--- | :--- | :--- |
| **Modifier $\rho(z)_{\text{mod}}$** | $+0.177$ | $\ge \mathbf{0.350}$ | Significant boost in modifier literalness ranking |
| **Head Noun $\rho(z)_{\text{head}}$** | $-0.093$ (Negative) | $\ge \mathbf{0.300}$ | Inverts and recovers corrupted head semantic axis |
| **Particle Verb $\rho_{\text{pv}}$** | $+0.197$ | $\ge \mathbf{0.350}$ | Supervised via dedicated GaussHead (PV excluded from InfoNCE) |
| **Pooled $\rho(z)_{\text{pooled}}$** | $\approx 0.10$ (Within-domain) | $\ge \mathbf{0.450}$ | Genuine cross-constituent Spearman correlation |

---

## 7. Step-by-Step Implementation Roadmap

1. **`src/config.py`**:
   - Add `use_wep_infonce: bool = True`
   - Add `wep_tau: float = 0.10`
   - Add `wep_weight: float = 0.08`
   - Set `proto_rank_weight = 0.0` when `use_wep_infonce` is enabled.
2. **`src/model_two_stream.py`**:
   - Instantiate `SharedProjector` in `__init__`.
   - In forward pass, project pooled Layer 19 vectors: $z_{\text{ctx}} = \text{Projector}(h_{\text{ctx}})$, $z_{\text{proto}} = \text{Projector}(h_{\text{proto}})$.
   - Cache `last_z_ctx_mod`, `last_z_proto_mod`, `last_z_ctx_head`, `last_z_proto_head`.
   - Add `self.projector` to `pred_heads()` so it trains throughout Phase 1.
3. **`src/losses.py`**:
   - Add `within_exit_infonce_loss` with full guards and per-exit min-max scaling.
4. **`src/train.py` & `src/trainer.py`**:
   - In `train_epoch`, invoke `within_exit_infonce_loss` using cached $z$ vectors.
   - In `evaluate`, log both `[Diag] Raw CosSim ρ` and `[Diag] Proj CosSim ρ (z)`.
