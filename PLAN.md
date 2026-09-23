# Implementation Plan: Prompt-Based Masked Probing (Cloze-Head Architecture)

## 1. Executive Summary & Core Motivation

### Why the Shift?
In our previous **Two-Stream Bi-Encoder** setup:
- Stream 1 encoded the context sentence $S$.
- Stream 2 encoded the isolated prototype word $W$ (e.g. `"apple"` or `"flea"`).
- **The Failure Point**: mmBERT encoded the naked word in an unnatural 3-token sequence (`[CLS] flea [SEP]`), while the contextual modifier in the sentence had blended into the head noun's domain. Unsupervised InfoNCE failed on modifiers ($\rho = -0.194$), and simple vector displacement $(h_{\text{ctx}} - h_{\text{proto}})$ starved the model of holistic sentence-to-compound interactions.

### The Breakthrough: Pattern-Exploiting Cloze-Probing
Instead of two isolated streams, we formulate the task as an **in-context cloze-style evaluation**:
$$\text{[CLS] Target: "\{word\}" in "\{compound\}" | Sentence: "\{Sentence\}" | In this context, "\{word\}" is used in a [MASK] sense. [SEP]}$$

1. **Pre-training Alignment**: mmBERT was pre-trained for millions of steps specifically to infer semantic properties at `[MASK]`.
2. **Full Cross-Attention**: Every token of the sentence, the compound, and the target word cross-attends across all 22 layers.
3. **Dual Prediction Streams**:
   - **Continuous Metric Head**: Extracts the hidden state $h_{\text{mask}}$ at the `[MASK]` position and passes it into `GaussHead` to predict $(\mu, \sigma)$ optimized by **NLL + CCC Loss**.
   - **Verbalizer Prior (MLM Logits)**: Leverages mmBERT's pre-trained LM head at `[MASK]` between `"literal"` (*wörtlich*) vs `"figurative"` (*übertragen*), providing an inductive bias that anchors the regression.

---

## 2. Architecture & Data Flow

```
                      +-------------------------------------------------------------+
                      |                        INPUT SAMPLE                         |
                      |  Sentence: "They bought an old clock at the flea market."   |
                      |  Target Word: "flea" | Compound: "flea market" | Lang: en   |
                      +-------------------------------------------------------------+
                                                     |
                                                     v
                                       [ Template Construction ]
      "[CLS] Target: \"flea\" in \"flea market\" | Sentence: \"...\" | The word \"flea\" is [MASK]. [SEP]"
                                                     |
                                                     v
                                      +-----------------------------+
                                      |    mmBERT-base (22 layers)  |
                                      |     with LoRA Adapters      |
                                      +-----------------------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v                                                       v
         Hidden state at [MASK] token                              Logits at [MASK] token
                 h_mask in R^768                                      from MLM Head
                         |                                                       |
                         v                                                       v
         +-------------------------------+                     Extract verbalizer logit diff:
         |  Feature Concatenation Block  |                     Delta = z("literal") - z("figurative")
         |  [h_mask, h_target, h_compound]                      (Zero-shot linguistic prior)
         +-------------------------------+                                       |
                         |                                                       |
                         +---------------------------+---------------------------+
                                                     |
                                                     v
                                        +-------------------------+
                                        |    GaussHead (Linear)   |
                                        |  Input: [h_mask, Delta] |
                                        +-------------------------+
                                                     |
                                                     v
                                   Predicted Normal: N(mu, sigma^2)
                                                     |
                                                     v
                                      +-----------------------------+
                                      |     MULTI-OBJECTIVE LOSS    |
                                      |  L = L_Gauss + 0.8 * L_CCC  |
                                      +-----------------------------+
```

---

## 3. Semantic-Aware Cloze Template Router

To prevent artificial syntactic framing and semantic vacuity across the real auxiliary files:
1. **Never rely strictly on table column shapes (`is_pv`)**:
   - `nctti_en_scored.tsv` has `Base`/`Particle` columns but contains real Noun Compounds (`"academy award"`).
   - `de-litnlit-aux.tsv` contains single bare lemmas (`"abbiegen"`), which are neither multi-word compounds nor phrasal verbs.
2. **Deterministic Semantic Classification Rule**:
   - If `filename` indicates `nn` (or compound has multiple words separated by space/hyphen and `compound != word`): classify as **Compound**.
   - If `compound == word` (or single bare token without constituents): classify as **Bare Lemma**.
   - If true Particle Verb (`head` is a spatial/directional particle like *off, up, ab, an*): classify as **Particle Verb**.

### A. Real Noun Compounds (`en-nn`, `de-nn`, and `nctti_en_scored.tsv`):
When a distinct compound exists (`compound != word` and `len(compound.split()) > 1`):
* **English**:
  ```text
  Target: "{word}" in "{compound}". Sentence: "{sentence}". In this sentence, the word "{word}" is [MASK].
  ```
* **German**:
  ```text
  Ziel: "{word}" in "{compound}". Satz: "{sentence}". In diesem Satz ist das Wort "{word}" [MASK].
  ```
  *(Note: For German solid compounds like "Flohmarkt", `compound` is the full word and `word` is the constituent lemma "Floh".)*

### B. True Particle Verbs (`en-pv`, `de-pv`):
When the target is a multi-word phrasal verb or particle-verb construction:
* **English**:
  ```text
  Target: "{verb_phrase}". Sentence: "{sentence}". In this sentence, the expression "{verb_phrase}" is [MASK].
  ```
* **German**:
  ```text
  Ziel: "{verb_phrase}". Satz: "{sentence}". In diesem Satz ist der Ausdruck "{verb_phrase}" [MASK].
  ```

### C. Bare Lemmas (`de-litnlit-aux.tsv`):
When `compound == word` or only a single lemma exists:
* **German**:
  ```text
  Ziel: "{word}". Satz: "{sentence}". In diesem Satz wird das Wort "{word}" im [MASK] Sinne verwendet.
  ```
* **English**:
  ```text
  Target: "{word}". Sentence: "{sentence}". In this sentence, the word "{word}" is used in a [MASK] sense.
  ```
*(Omit the redundant `in "{compound}"` clause entirely to prevent vacuous self-nesting like `Target: "abbiegen" in "abbiegen"`.)*

---

## 4. Multi-Layer Extraction & The Organizers' Discovery

From Miletić & Schulte im Walde (2023, 2025):
- Compound semantics is strongest in **Layers 4 to 8** (lexical/semantic space).
- Sentence-level syntax and context stabilization occurs in **Layers 14 to 18**.
- **Strategy**: Instead of extracting only the final layer, $h_{\text{mask}}$ is extracted as a weighted concatenation or projection of:
  $$h_{\text{mask}} = W_p \left[ h_{\text{mask}}^{(\text{layer 6})} \,;\, h_{\text{mask}}^{(\text{layer 16})} \,;\, h_{\text{mask}}^{(\text{layer 22})} \right]$$

---

## 5. Detailed Step-by-Step Implementation Roadmap

### Phase 1: Dataset & Tokenizer Adaptation (`src/prompt_dataset.py`)
1. Create a `ClozeCompositionalityDataset` that maps each row `(sentence, word, compound, label)` to:
   - Formatted prompt string.
   - Exact index position of the `[MASK]` token.
   - Indices of the `word` and `compound` token spans in the prompt.
2. Ensure dynamic padding and attention masks handle `[MASK]` position tracking cleanly.

### Phase 2: Cloze Model Architecture (`src/model_cloze.py`)
1. Wrap `AutoModelForMaskedLM` (mmBERT).
2. Attach LoRA adapters to attention projections (`q_proj`, `v_proj`, `k_proj`, `out_proj`).
3. Add `ClozeHead`:
   - Extracts $h_{\text{mask}}$ from intermediate and top layers.
   - Computes verbalizer logit scalar:
     $$\Delta_{\text{verb}} = \text{logit}(w_{\text{literal}}) - \text{logit}(w_{\text{figurative}})$$
   - Passes $[h_{\text{mask}} \,;\, \Delta_{\text{verb}}]$ through an MLP into `GaussHead` $\to (\mu, \sigma)$.

### Phase 3: Losses & Optimization (`src/losses.py` & `src/trainer.py`)
1. **Primary Loss**: Gaussian NLL Loss:
   $$\mathcal{L}_{\text{Gauss}} = \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\ln(\sigma^2)$$
2. **Correlation Anchor**: Lin's Concordance Correlation Coefficient (CCC):
   $$\mathcal{L}_{\text{CCC}} = 1 - \frac{2\rho\sigma_x\sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$
3. **Optional MLM Auxiliary Regularizer**: Cross-Entropy at `[MASK]` against the verbalizer target for extreme cases ($y > 4.2 \to \text{"literal"}$, $y < 1.8 \to \text{"figurative"}$).

### Phase 4: Training & Evaluation Pipeline
1. **Curriculum**:
   - Epochs 1–5: Warmup `ClozeHead` with backbone frozen.
   - Epochs 6–25: Joint fine-tuning of LoRA + `ClozeHead` with cosine decay and EMA ($0.999$).
2. **Validation**: Direct evaluation of Spearman $\rho$ against ground-truth human ratings on:
   - `en-nn` (ModAvg, HeadAvg)
   - `de-nn` (ModAvg, HeadAvg)
   - `en-pv` (Avg)
   - Macro Mean $\rho$

---

## 6. Comparison: What Makes This Superior?

| Property | Old Two-Stream Approach | New Cloze-Prompt Approach |
| :--- | :--- | :--- |
| **Cross-Attention** | None (Isolated Bi-Encoder) | Full bidirectional cross-attention across sentence & target |
| **Input Structure** | Unnatural 2-word input vs 40-word input | Natural cloze sentence identical to pre-training distribution |
| **Feature Richness** | Only 2 vectors + cosine scalar | Entire sentence context focused into $h_{\text{mask}}$ + MLM logits |
| **Language Handling** | Rigid vector arithmetic | Native natural prompts in English and German |
| **Optimization Target** | Unsupervised InfoNCE (indirect proxy) | Direct NLL + CCC Correlation Loss on human scores |
