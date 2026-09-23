# Comprehensive Audit & Bug Analysis Report

This document provides an end-to-end audit of the training and data pipeline for the Compositionality Probing & SemEval-2022 Task 2 system (incorporating Two-Stream, Target-Centric, and Cloze Probing backends).

---

## 1. End-to-End Pipeline Architecture & Flow

```
[TSV Raw Datasets]
 (en/de-nn, en/de-pv, de-litnlit-aux)
       │
       ▼  (1) TSV Loading & Standardization (src/data.py)
 [Unified Row Dicts] ─── Normalized keys: mod, head, compound, is_pv, mod_avg, head_avg
       │
       ▼  (2) Target Expansion (src/data.py: expand_targets)
 [Target-Specific Rows] ─── NN split into (mod, ModAvg) & (head, HeadAvg); PV -> (pv, Avg)
       │
       ▼  (3) Semantic Prompt Synthesis (src/cloze_prompts.py)
 [Cloze Prompt String] ─── "Target: \"flea\" in \"flea market\". Score: [MASK] / 5"
       │
       ▼  (4) Mask-Guaranteed Tokenization (src/cloze_dataset.py)
 [Tokenized Tensors] ─── input_ids, attention_mask, mask_pos, labels, stds
       │
       ▼  (5) Multi-Depth Layer Aggregation (src/model_cloze.py)
 [Hidden Extraction] ─── mmBERT Layers [6, 14, 20] -> Softmax weighted sum -> GaussHead (μ, σ)
       │
       ▼  (6) Gaussian KL + CCC Objective (src/losses.py & src/train.py)
 [Loss Computation] ─── KL(N(μ, σ²) || N(y, σ_t²)) + 0.7 * (1 - CCC)
       │
       ▼  (7) Evaluation & Correlation (src/trainer.py & src/train.py)
 [Validation & Metrics] ─── Spearman ρ computed per exit group (nn_mod, nn_head, pv)
```

---

## 2. Catalog of Identified Bugs, Root Causes & Fixes

### Bug #1: Python String Expression Evaluated as Boolean in Cloze Prompts
* **Severity**: 🔴 Critical (Directly crippled Head and Modifier probe learning)
* **Location**: `src/cloze_prompts.py`
* **Root Cause**:
  ```python
  # ❌ BUGGY LINE:
  target = f'{w}' in '{c}'
  ```
  Because `in` was outside the quotes, Python evaluated `'market' in 'flea market'` as a boolean expression returning `True`.
* **Symptom in Logs**:
  Prompts contained `Target: "True". Score: [MASK] / 5` instead of `"market" in "flea market"`. `Train Head ρ` and `Val Head ρ` hovered near random chance ($0.02 - 0.09$).
* **Fix**:
  ```python
  # ✅ FIXED:
  if c and w and c != w:
      target = f'"{w}" in "{c}"'
  else:
      target = f'"{w}"' if w else f'"{c}"'
  ```

---

### Bug #2: Cloze Backend Missing Automatic Target-Centric Row Expansion
* **Severity**: 🔴 Critical
* **Location**: `src/trainer.py: _build_loaders`
* **Root Cause**:
  When `cfg.targets` was not explicitly passed in the JSON config, `expand_targets` was bypassed. A raw NN compound row with both `(ModAvg, HeadAvg)` remained a single record. In `ClozePromptDataset`, it only queried `mod` and completely discarded `head`.
* **Symptom in Logs**:
  The model only received Modifier supervision during training, and the Head evaluation received missing/NaN targets.
* **Fix**:
  ```python
  # ✅ FIXED:
  active_targets = self.cfg.targets if self.cfg.targets else (['mod', 'head', 'pv'] if backend == 'cloze' else None)
  if active_targets:
      train_rows = expand_targets(train_rows, active_targets)
      val_rows = expand_targets(val_rows, active_targets)
  ```

---

### Bug #3: Validation Evaluation Metric Masking Misalignment
* **Severity**: 🟠 High
* **Location**: `src/trainer.py: line 504 & line 612`
* **Root Cause**:
  Validation evaluation checked `if self.cfg.targets:` instead of detecting whether `_val_rows` had target-expanded records. When `cfg.targets` was None, `nn_head_mask` was assigned `nn_mask` (which includes both `mod` and `head` samples together). Because a `mod` sample has NaN in `head_avg`, the correlation calculation was corrupted.
* **Fix**:
  ```python
  # ✅ FIXED:
  has_target_expanded = bool(self.cfg.targets) or (getattr(self.cfg, 'model_backend', 'twostream') == 'cloze') or any(r.get('target') is not None for r in self._val_rows)
  if has_target_expanded:
      trg = np.array([0 if r.get('target') == 'mod' else (1 if r.get('target') == 'head' else 2) for r in self._val_rows])
      nn_mod_mask = val_mask & (~is_pv_mask) & (trg == 0)
      nn_head_mask = val_mask & (~is_pv_mask) & (trg == 1)
      pv_mask = val_mask & is_pv_mask & (trg == 2)
  ```

---

### Bug #4: Particle Verb Semantic Field Mapping Ambiguity
* **Severity**: 🟡 Medium (Conceptual / Schema clarity)
* **Location**: `src/data.py` & `src/cloze_dataset.py`
* **Explanation**:
  SemEval-2022 Task 2 uses `Avg` for Particle Verbs (as PVs have one overall idiomaticity score) and `ModAvg`/`HeadAvg` for Noun Compounds. The legacy codebase mapped `mod_avg = row['Avg']` as an internal dataloader shortcut.
* **Fix**:
  Unified ground-truth extraction into explicit `label` / `stds` in `ClozePromptDataset`:
  * `target == 'mod'` $\to$ `item['mod_avg']`
  * `target == 'head'` $\to$ `item['head_avg']`
  * `target == 'pv'` $\to$ `item['mod_avg']` (the overall PV `Avg` score)

---

### Bug #5: Bare German Lemma Formatting Redundancy
* **Severity**: 🟢 Minor (Linguistic formatting)
* **Location**: `src/cloze_prompts.py`
* **Root Cause**:
  German auxiliary verbs (e.g. `"abbiegen"` in `de-litnlit-aux.tsv`) have `Base == Compound == "abbiegen"`. Formatting as `"abbiegen in abbiegen"` was grammatically unidiomatic.
* **Fix**:
  Treated bare lemmas as standalone targets: `Ziel: "abbiegen". Bewertung: [MASK] / 5`.

---

## 3. Training & Validation Health Verification

After applying the fixes:
1. **Smoke Tests**: All test suites in `tests/smoke_src.py` pass without errors.
2. **Applet Compilation**: `compile_applet` runs cleanly with zero build errors.
3. **Data Routing**: Every sample correctly routes its constituent to the target prompt, with exact alignment between predicted $\mu$ and human ground-truth labels.
