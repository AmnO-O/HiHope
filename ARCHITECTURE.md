# Two-Stream Bi-Encoder Architecture for Compositionality Assessment

## 1. Core Architecture Overview

Unlike architectures that rely solely on in-context representations or static Layer 0 lookup tables, MoTune uses a **True Two-Stream Bi-Encoder Architecture** with `jhu-clsp/mmBERT-base` (22 layers, $H=768$):

```
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

---

## 2. Mathematical Formalism

### Stream 1: Prototype Vector $h_{\text{word}}$
The target word or compound constituent (e.g., *"flea"* or *"flea market"*) is passed through mmBERT in isolation:
$$h_{\text{word}} = \text{pool\_prototype}\left(\text{mmBERT}(\text{word\_tokens})\right) \in \mathbb{R}^{H}$$
Special tokens $[CLS]$ and $[SEP]$ are excluded to isolate the pure lexical prototype representation.

### Stream 2: In-Context Vector $h_{\text{context}}$
The full sentence context is tokenized without artificial marker delimiters:
$$H_{\text{ctx}} = \text{mmBERT}(\text{sentence\_tokens}) \in \mathbb{R}^{L \times H}$$
Using character offset alignments, the subwords corresponding to the target constituent are extracted via masked-mean pooling:
$$h_{\text{context}} = \frac{\sum_{i=1}^L m_i \cdot H_{\text{ctx}, i}}{\sum_{i=1}^L m_i} \in \mathbb{R}^{H}$$

### Semantic Interaction & Directional Displacement
The interaction between context and prototype captures whether the word retains its literal sense or undergoes idiomatic drift:
1. **Directional Semantic Displacement:**
   $$\Delta h = h_{\text{context}} - h_{\text{word}} \in \mathbb{R}^{H}$$
2. **Multiplicative Constituent Interaction:**
   $$h_{\text{prod}} = h_{\text{context}} \odot h_{\text{word}} \in \mathbb{R}^{H}$$
3. **Calibrated Cosine Similarity:**
   $$\cos\_sim = \frac{h_{\text{context}} \cdot h_{\text{word}}}{\|h_{\text{context}}\|_2 \|h_{\text{word}}\|_2} \in [-1, 1]$$

### Fusion & Calibrated Uncertainty Prediction
The features are concatenated into a unified interaction vector of dimension $4H + 1 = 3073$:
$$z = \text{GELU}\left(\text{LayerNorm}\left(W_f [h_{\text{context}}; h_{\text{word}}; \Delta h; h_{\text{prod}}; \cos\_sim] + b_f\right)\right) \in \mathbb{R}^{H}$$
$$(\mu, \sigma) = \text{GaussHead}(z)$$
where:
- $\mu \in [0.0, 5.0]$: Central degree of compositionality (literalness).
- $\sigma \ge 0.05$: Modeled annotator disagreement variance ($\text{softplus}(\cdot) + \text{floor}$).

---

## 3. Training Strategy (No LoRA Needed)
With modern GPUs, mmBERT-base (~110M parameters) is fully fine-tuned or trained with top-layer unfreezing directly:
- **Optimizer:** AdamW with linear warmup and cosine decay.
- **Learning Rate:** $1\text{e-}5$ to $2\text{e-}5$ for the encoder, $1\text{e-}4$ for the fusion and Gaussian heads.
- **Loss Function:**
  $$\mathcal{L} = \mathcal{D}_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(y, \sigma_y^2)\right) + \lambda_{\text{ccc}} \mathcal{L}_{\text{CCC}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}$$
