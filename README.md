# SemEval-2027 Task 5: Multilingual Compositionality in Context (NC & PV)

An advanced pipeline for predicting contextual compositionality ratings (1.0 → 5.0) of multi-word expressions (Noun Compounds and Particle Verbs) in English and German.

## 🌟 Architecture Overview

```
[Input Sequence] 
  Context + Target (Noun Compound: Mod/Head or Particle Verb: Base/Particle)
        │
        ▼
[Layer 0: Encoder & Explicit Role Injection]
  H_final = Backbone(Input_IDs) + E_role (0: Context, 1: Mod/Verb, 2: Head/Particle, 3: Whole)
        │
        ▼
[Layer 1: Target-Aware Multi-Head Cross-Attention (Pre-LN + RMSNorm)]
  Learned Queries: [q_Mod, q_Head, q_Compound] probe context embeddings H_final
        │
        ▼
[Layer 2: Component Interaction Layer (Pre-LN + RMSNorm)]
  Multi-Head Self-Attention between constituents (Mod ↔ Head ↔ Whole Compound/PV)
        │
        ▼
[Layer 3: Shared Gaussian Regression Head]
  Predicts continuous mean (mu ∈ [1.0, 5.0]) and annotator uncertainty (sigma)
        │
        ▼
[GaussLoss & Evaluation]
  Hybrid Loss: Lin's CCC Loss + Gaussian Distribution KL Divergence Loss
```

## 🚀 Key Modern Features
- **Backbone Support:** Out-of-the-box support for `EuroBERT/EuroBERT-210m`, `EuroBERT/EuroBERT-610m`, and `jhu-clsp/mmBERT-base`.
- **Pre-LN & RMSNorm:** High-stability gradient propagation via Pre-Layer Normalization and Root Mean Square Normalization.
- **Top-N Layer Unfreezing:** Configurable progressive fine-tuning (`unfreeze_layers: 4`, `-1` for full backbone).
- **Dual Modality:** Unified processing of Noun Compounds (adjacent/fused) and Particle Verbs (separable/discontinuous in German & English).

## 🛠️ Quickstart (Kaggle & Local)

### 1. Training with EuroBERT-210m
```bash
python run.py \
  --config config/target_aware_query_attention.json \
  --set backbone=EuroBERT/EuroBERT-210m \
  --set use_pre_ln=True \
  --set use_rms_norm=True \
  --set unfreeze_layers=4 \
  --set freeze_epochs=3 \
  --set unfreeze_epochs=9
```

### 2. Interactive Notebook
Open and run `notebooks/kaggle_run.ipynb` on GPU (T4 / P100) for full visual analysis, attention maps, and trial evaluations.
