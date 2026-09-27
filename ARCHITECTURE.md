# Compositionality Assessment Architectures

This codebase supports two primary paradigms for Noun Compound and Particle Verb compositionality rating ($1.0 \to 5.0$):
1. **Target-Aware Query Cross-Attention & Component Interaction** (`src/model_query.py` - `model_backend: query_attention`)
2. **Two-Stream Bi-Encoder Architecture** (`src/model_two_stream.py` - `model_backend: twostream`)

---

# 1. Target-Aware Query Cross-Attention Pipeline

```text
[INPUT SEQUENCE]
Context: "This was soon thrown out through the back door, never to be seen again."
Target Metadata: Mod = "back", Head = "door", Compound = "back door"
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ LAYER 0: ENCODER & EXPLICIT ROLE INJECTION                                    │
│                                                                               │
│  Input Token IDs  [B, S] ──────► mmBERT Encoder ────► H_mmBERT  [B, S, 768]   │
│  Role IDs         [B, S] ──────► Role Embeddings ───► E_role    [B, S, 768]   │
│  (0: Context, 1: Mod, 2: Head)                                │               │
│                               H_final = H_mmBERT + E_role ───┴─► [B, S, 768]  │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TARGET-AWARE MULTI-HEAD CROSS-ATTENTION (MHCA)                       │
│                                                                               │
│  Learned Queries: Q_base = [q_Mod, q_Head, q_Compound] ∈ [3, 768]             │
│  Batch Expand  ────────► Q ∈ [B, 3, 768]                                      │
│                                                                               │
│  Queries (Q)  = Q               ∈ [B, 3, 768]                                 │
│  Keys (K)     = H_final · W_k   ∈ [B, S, 768]                                 │
│  Values (V)   = H_final · W_v   ∈ [B, S, 768]                                 │
│                                                                               │
│  A = Softmax(Q·Kᵀ / √d_h)       ∈ [B, 3, S]  (Interpretability Attention Map) │
│  Z_attn = MultiHeadAttention(Q, K, V)                                         │
│  Z = LayerNorm(Q + Dropout(Z_attn))                                           │
│  ──► Tensor Z ∈ [B, 3, 768] (Context-Conditioned Target Representations)     │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: COMPONENT INTERACTION LAYER (MHSA)                                   │
│                                                                               │
│  Self-Attention tương tác giữa các thành tố (Mod ↔ Head ↔ Compound):           │
│  Z_self = MultiHeadSelfAttention(Q=Z, K=Z, V=Z)                               │
│  Z' = LayerNorm(Z + Dropout(Z_self))                                          │
│  ──► Tensor Z' ∈ [B, 3, 768] (Boundary-Aware Compositional Representations)   │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: GAUSSHEAD REGRESSION & CONTINUOUS PREDICTION                         │
│                                                                               │
│  z'_Mod      [B, 768] ──┐                                                     │
│  z'_Head     [B, 768] ──┼──► Shared GaussHead ──►  Mod: (μ_mod, σ_mod)          │
│  z'_Compound [B, 768] ──┘      (Shared MLP)        Head: (μ_head, σ_head)     │
│                                                   Comp: (μ_comp, σ_comp)      │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ CCC LOSS & TARGET NORMALIZATION                                               │
│                                                                               │
│  Targets chuẩn hóa: y_norm = y_raw / 5.0 ∈ [0.2, 1.0]                          │
│  Loss = L_CCC(μ_head, y_head_norm) + L_CCC(μ_mod, y_mod_norm)                 │
│  Dự đoán điểm cuối cùng: Score = μ * 5.0                                      │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Layer 0: Encoder & Explicit Role Injection**
   - Natural context tokenized without artificial delimiters.
   - Subtoken roles tagged via `role_ids` ($0$: Context, $1$: Mod, $2$: Head, $3$: Compound/PV).
   - $E_{\text{role}}$ adds directly onto $H_{\text{mmBERT}}$: $H_{\text{final}} = H_{\text{mmBERT}} + E_{\text{role}}$.

2. **Layer 1: Target-Aware Multi-Head Cross-Attention (MHCA)**
   - 3 learned queries $Q_{\text{base}} = [q_{\text{Mod}}, q_{\text{Head}}, q_{\text{Compound}}] \in \mathbb{R}^{3 \times 768}$.
   - Probes $H_{\text{final}}$ to extract context-dependent semantic nuances.
   - Attention map $A \in \mathbb{R}^{B \times 3 \times S}$ provides full token-level interpretability.

3. **Layer 2: Component Interaction Layer (MHSA)**
   - Self-Attention across $Z \in \mathbb{R}^{B \times 3 \times 768}$.
   - Allows modifier and head representations to inform each other before final scoring.

4. **Layer 3: Shared GaussHead Regression**
   - Predicts central tendency $\mu \in [0, 1]$ and uncertainty variance $\sigma \ge 0.04$.
   - Continuous score output: $\text{Score} = \mu \times 5.0$.

5. **Optimization: Lin's CCC Loss**
   - Directly optimizes ranking and concordance on normalized targets $y_{\text{norm}} = y / 5.0$.

---

# 2. Two-Stream Bi-Encoder Architecture

```text
 ┌────────────────────────────────────────────────────────┐
 │ Stream 1: Isolated Target Word (Prototype)             │
 │   [CLS] target_lemma [SEP]                             │
 │   ──> mmBERT Encoder ──> pool_prototype                │
 └───────────────────────────┬────────────────────────────┘
                             │  h_word [B, 768]
                             ▼
                    ┌─────────────────┐
                    │ Semantic Fusion │ <── h_context [B, 768]
                    └────────┬────────┘          │
                             │                   │
 ┌───────────────────────────┴───────────────────┴────────┐
 │ Stream 2: Full Context Sentence                        │
 │   [CLS] sentence_with_compound_in_context [SEP]        │
 │   ──> mmBERT Encoder ──> pool_active_context           │
 └────────────────────────────────────────────────────────┘
```
