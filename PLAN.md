# Architecture & Training Plan: Target-Centric In-Context Probing (Bottleneck Cloze Regression)

## 1. Executive Summary & Core Motivation

### The Breakthrough: Target-Centric Attention Bottleneck
Traditional compositionality models suffer from two extremes:
1. **Isolated Bi-Encoders**: Starve the model of cross-attention between the sentence context and the target constituent, causing metric collapse on modifiers ($\rho = -0.194$).
2. **Heavy Vocabulary MLM Probing**: Forces continuous human ratings into discrete token buckets and incurs massive computational overhead ($768 \to 250{,}000$ vocabulary projection matrix).

### Our Solution: In-Context Bottleneck Regression
We formulate compositionality scoring as an **In-Context Bottleneck Query**:
$$\text{"\{sentence\} Target: \"\{target\}\". Score: [MASK] / 5"}$$

1. **Information Bottleneck**: The `[MASK]` token acts as a cross-attentive query probe. Across mmBERT's 22 transformer layers, self-attention aggregates interactions between the sentence context and the target expression into a single 768-dimensional vector $h_{\text{mask}}$.
2. **Direct Continuous Score Prediction**: $h_{\text{mask}}$ is fed directly into `GaussHead`, predicting the continuous Gaussian distribution $(\mu, \sigma)$ on the true human scale $[0.0, 5.0]$.
3. **Ultra-Compact Prompts**: Adds only ~7 tokens per sample, preserving 95%+ of sequence capacity for long linguistic contexts while eliminating 250k-token vocabulary projection overhead.

---

## 2. Architecture & Data Flow

```
                      +-------------------------------------------------------------+
                      |                        INPUT SAMPLE                         |
                      |  Sentence: "They bought an old clock at the flea market."   |
                      |  Target: "flea" | Compound: "flea market" | Lang: en        |
                      +-------------------------------------------------------------+
                                                     |
                                                     v
                                      [ Ultra-Short Prompt Carrier ]
                     "They bought an old clock at the flea market. Target: \"flea\". Score: [MASK] / 5"
                                                     |
                                                     v
                                      +-----------------------------+
                                      |    mmBERT-base (22 layers)  |
                                      |  Layers 0-11: Frozen        |
                                      |  Layers 12-21: LoRA Tuned   |
                                      +-----------------------------+
                                                     |
                                                     v
                                      [ Multi-Layer Aggregation ]
                                   h_mask = Agg(L6, L14, L20)[mask_pos]
                                              ∈ ℝ⁷⁶⁸
                                                     |
                                                     v
                                      +-----------------------------+
                                      |          GaussHead          |
                                      |    Linear(768 -> 256)       |
                                      |    LayerNorm + GELU         |
                                      |    Linear(256 -> 1) -> μ    |
                                      |    Linear(256 -> 1) -> σ    |
                                      +-----------------------------+
                                                     |
                                                     v
                                  Predicted Normal: 𝒩(μ̂, σ̂²)  where:
                                  μ̂ ∈ [0.0, 5.0] (Compositionality Mean)
                                  σ̂ ∈ [0.04, 2.0] (Rater Uncertainty)
                                                     |
                                                     v
                                      +-----------------------------+
                                      |     MULTI-OBJECTIVE LOSS    |
                                      |  ℒ = ℒ_Gauss(y, σ_human)    |
                                      |      + 0.8 * ℒ_CCC          |
                                      +-----------------------------+
```

---

## 3. Semantic-Aware Prompt Formulation

Prompts are designed to be ultra-compact, language-specific, and target-agnostic (handling Single Words, Compounds, Particle Verbs, and Bare Lemmas):

### A. Real Noun Compounds (`en-nn`, `de-nn`, `nctti_en_scored.tsv`)
* **English**:
  ```text
  {sentence} Target: "{word}". Score: [MASK] / 5
  ```
* **German**:
  ```text
  {sentence} Ziel: "{word}". Bewertung: [MASK] / 5
  ```

### B. Particle Verbs (`en-pv`, `de-pv`)
* **English**:
  ```text
  {sentence} Target: "{verb_phrase}". Score: [MASK] / 5
  ```
* **German**:
  ```text
  {sentence} Ziel: "{verb_phrase}". Bewertung: [MASK] / 5
  ```

### C. Bare Lemmas (`de-litnlit-aux.tsv`)
* **German**:
  ```text
  {sentence} Ziel: "{lemma}". Bewertung: [MASK] / 5
  ```

---

## 4. Multi-Layer Extraction & Attention Calibration

From empirical probing dynamics and Miletić & Schulte im Walde (2023, 2025):
- **Layer 6 (Early Semantic)**: Captures base lexical semantics and subword grounding.
- **Layer 14 (Intermediate Syntax)**: Syntactic binding between compound constituents and verb particles.
- **Layer 20 (Deep Compositionality)**: High-level contextual semantic shift and idiomaticity representation.

$$\mathbf{w} = \text{Softmax}([w_6, w_{14}, w_{20}])$$
$$h_{\text{mask}} = w_6 \cdot h_{\text{mask}}^{(6)} + w_{14} \cdot h_{\text{mask}}^{(14)} + w_{20} \cdot h_{\text{mask}}^{(20)}$$

---

## 5. Optimization & Training Schedule

### Layer Allocation across 22-Layer mmBERT-base
* **Layers 0 – 11 (12 Layers)**: **Permanently Frozen**. Retains multilingual backbone integrity and prevents catastrophic forgetting.
* **Layers 12 – 21 (10 Layers)**: **LoRA Fine-Tuned** ($r=8, \alpha=16$, dropout $0.1$) on `attn.Wqkv` and `attn.Wo` (20 total adapters).

### Two-Phase Training Schedule
1. **Phase 1: Warmup & Alignment (Epochs 1 – 10)**:
   - Encoder frozen (`encoder_lr = 0`).
   - Train `GaussHead` and `MultiLayerAggregator` (`head_lr = 3e-4`, `layer_agg_lr = 1e-3`).
   - Aligns random projection weights into the $[0.0, 5.0]$ Gaussian density manifold.
2. **Phase 2: Full In-Context Adaptation (Epochs 11 – 35+)**:
   - Unfreeze top-10 LoRA adapters (`encoder_lr = 1e-5` with Cosine Annealing decay).
   - Head LR decays smoothly from $3\times 10^{-4} \to 2\times 10^{-5}$.
   - EMA shadow weights ($\beta = 0.999$) enabled to stabilize validation evaluation.

---

## 6. Loss Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Gauss}} + \lambda_{\text{CCC}} \cdot \mathcal{L}_{\text{CCC}}$$

### 1. Heteroscedastic Gaussian NLL with Human Uncertainty
$$\mathcal{L}_{\text{Gauss}} = \frac{1}{2} \sum_{i=1}^{B} \left[ \ln(\hat{\sigma}_i^2 + \sigma_{i,\text{human}}^2) + \frac{(\hat{\mu}_i - y_{i,\text{human}})^2}{\hat{\sigma}_i^2 + \sigma_{i,\text{human}}^2} \right]$$

### 2. Lin's Concordance Correlation Coefficient (CCC) Loss
$$\text{CCC} = \frac{2 s_{xy}}{s_x^2 + s_y^2 + (\bar{x} - \bar{y})^2}, \qquad \mathcal{L}_{\text{CCC}} = 1 - \text{CCC}$$

---

## 7. Performance & Comparison Matrix

| Property | Old Two-Stream Bi-Encoder | Heavy MLM Token Guessing | Target-Centric Bottleneck Probing (Our Plan) |
| :--- | :--- | :--- | :--- |
| **Cross-Attention** | None (Isolated streams) | Full Bidirectional | **Full Bidirectional Contextualization** |
| **Target Representation** | Raw token diff $(h_{\text{ctx}} - h_{\text{proto}})$ | Vocabulary token distribution | **Learned Attention Bottleneck ($h_{\text{mask}}$)** |
| **Memory / VRAM Overhead** | Medium | Extreme ($250\text{k} \times 768$ matrix) | **Minimal (direct $768 \to 256 \to 2$ projection)** |
| **Prompt Length** | N/A | Long & verbose (20+ tokens) | **Ultra-Compact (~7 tokens)** |
| **Prediction Space** | Heuristic bounds | Discrete tokens (`"1"`..`"5"`) | **Continuous Calibrated Gaussian $\mathcal{N}(\mu, \sigma^2)$ on $[0.0, 5.0]$** |
| **Optimization Target** | Unsupervised InfoNCE | Cross-Entropy | **Gaussian NLL + Lin's CCC Loss** |
