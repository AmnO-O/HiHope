# Bug Review — HiHope

Full-codebase review (data handling + model + training/eval + tooling/tests), verified by direct
line inspection. Severity: **HIGH** = silent wrong behavior / crash in reachable paths,
**MED** = wrong under specific (mostly non-default) config, **LOW** = latent / cosmetic /
data-dependent.

---

## 1. MODEL code

### H-1. `from_layer` LoRA window is dead for mmBERT/BERT — wraps ALL 22 layers
**`src/lora.py:64-70`** (`_layer_idx`), used at **:96-99** (`_derive_attn_targets`) and the `main()` path.
`_layer_idx` only matches segments named `layers` (`s == 'layers'`) but the backbone modules live under
`encoder.layer.N` (standard BERT). So it returns `None` for every dotted path, the
`layer < from_layer` prune **never fires**, and `apply_lora`/`_derive_attn_targets` wrap **every layer**
regardless of `lora_from_layer` (configs set 14/18). The "layers >= from_layer" log still looks correct.
**Impact:** 2-3x the intended adapter params/compute; the committed warm-LoRA run
(`lora_from_layer: 14`) is not doing what the config says.
**Fix:** match `s in ('layers', 'layer')`.
**Severity: HIGH**

### H-2. `src/model_combined.py` is un-importable
**`src/model_combined.py:6-7`** imports `build_two_stream_model` from `src.model_two_stream`, but that
name is **defined nowhere** in the repo (repo-wide grep: only `model_combined.py` mentions it).
`import src.model_combined` → `ImportError`; `tests/smoke_src.py:816` depends on it.
The alias is also never *used* in production (`config.py:228-229` rewrites `'combined'` → `'twostream'`).
**Fix:** delete the shim or define `build_two_stream_model`.
**Severity: HIGH**

### H-3. Saved checkpoint cannot be loaded by `build_model(load_from=...)` (latent)
**`src/trainer.py:551`** saves `model.state_dict()` (keys `lm.*`, `fusion.*`, `_head_mod.*`,
`layer_agg.*`, LoRA `*.a/b/linear.*`); **`src/model.py:461-462, 475-476`** do
`model.lm.load_state_dict(state)` (strict, backbone-only keys) → `unexpected/missing key` crash.
No caller currently passes `load_from`, so it is **write-only today** — but the flag is broken.
**Fix:** load the `lm.`-prefixed subset (or the full dict into the model).
**Severity: HIGH (latent)**

### M-1. Rank-loss cosine misaligned when `proto_stream=False` with a single target
**`src/train.py:137-144`** uses `model.last_cos_sim`; that tensor is only row-correct (rebuilt by
`torch.where(tgt...)`) in the *single-target + proto_ids* forward path
(`src/model_two_stream.py:363-381`). Without `proto_ids` in the batch the model takes the joint branch
and `last_cos_sim` is last-write-wins → the **PV** cosine on every row
(`src/model_two_stream.py:434-441`), which the margin loss then ranks against mod/head labels.
`last_mod_cos/head_cos/pv_cos` are available and unused here.
*Not hit by the shipped config* (`proto_stream=true`, `target_mask_prob=0.0` → multi-target batch, no
`'target'` key → joint branch at train.py:148-161 which *does* use the per-task caches).
**Fix:** make the trainer's `'target' in batch` branch require `proto_stream`, or select cos by target.
**Severity: MED (needs confirmation of that config space)**

### L-1. `pool_prototype` truncation blanks a real token, not `[SEP]`
**`src/prototype_stream.py:46-50`** — at `max_proto_length` truncation (`src/data.py:411`) position
`l-1` is a real subword; the code zeros it anyway, silently dropping it from the prototype pool.
**Severity: LOW**

### L-2. Empty-span rows produce a zero prototype → NaN cosine poisons the linear fusion path
**`src/model_two_stream.py:283-285, 430-432`**; `src/data.py:414` (`_tok_word` → `zeros(1)` on empty).
`corrected_cosine_similarity(0, v)` = 0/0 = `NaN`; with `fusion_type="linear"` the NaN enters
`combined = cat([..., cos_sim])` → NaN mu. Cross-attention path is safe (cos is only cached).
**Fix:** guard the cosine for all-zero rows.
**Severity: LOW** (requires `fusion_type="linear"`)

### L-3. Cache tensors keep graph references alive
**`src/model_two_stream.py:277-278, 365-369`**, `src/prototype_stream.py:152-153` —
`last_cos_sim`, `last_displacement_norm`, `last_cos`, `last_raw_cos` hold undetached graph tensors until
the next forward; memory retention across `accum_steps`. Use `.detach()`.
**Severity: LOW**

### L-4. Dead `.linear.` skip in `unfreeze_top_layers`
**`src/train.py:349-352`** — `LoRAAdapter` stores the base `nn.Linear` as a bare attribute
(`src/lora.py:27`), so base `weight`/`bias` never appear in `named_parameters()`; the `.linear.`
filter never matches and freezing is actually guaranteed by the adapter. Docstring (`src/train.py:9-10`)
describes a mechanism that isn't real. Also **LOW**: **`src/lora.py:233` `merge_lora`** has no callers.

### Checked and found OK
Per-task GaussHead refactor (`_head_mod/_head_head/_head_pv` vs shared `_head`, properties,
`pred_heads()` dedup) — no key collisions, no double-optimization. `torch.where` tgt selection correct.
`last_cos_sim` captured right after each `_forward_pair` in the single-target path (the earlier
overwrite bug is fixed). MHA additive-mask reshape is functionally equivalent. `GaussHead` sigma-bias
init lands at ≈0.5.

**Duplicate pooling logic (consolidation):** masked-mean implemented 4x with subtly different
fallbacks — `_masked_mean` (`model.py:48`), `pool_active_context` (`model_two_stream.py:193`),
`pool_span`/`pool_active` (`targets.py:68,88`), `_get_prototype_from_emb`
(`model_two_stream.py:291`); anisotropy-corrected cosine computed twice per pair.

---

## 2. DATA code

### D-1. English consonant-doubling rule over-applies → correct `-ed/-ing` forms missing; garbage injected
**`src/marks.py:157-219`** (`_alternate_forms`, `doubles()` at **187-190**, `elif doubles()` at **204-208**).
`doubles()` fires for any CVC-final string without syllable check — true for `water`, `offer`, `open`,
`cover`, `answer` (t[-2] vowel, t[-3] not vowel). The `elif` then shadows the correct `else`
(**209-214**), so `watered/offered/opened` never enter the candidate set and non-words
(`waterred`, `openned`, `offerring`, `stepps`, `dropps`) do. Verified: **all 5 shipped `water down`
rows in `en-pv-train.tsv` fail to align** (`find_spans → found=False`) → silently pushed to
whole-sentence pooling. `crack/step/shop/drop` are fine (last-2 isn't a vowel).
**Fix:** always emit `t + "ed"`/`t + "ing"` (plus y-rules) and only *add* doubling variants for
genuine 1-syllable CVC bases (exclude `-er/-en/-el/-ow/-al` endings).
**Severity: MED** (confirmed wrong output, real leakage into shipped data)

### D-2. German NN row whose head noun is in `_GERMAN_PARTICLES` routed to PV matcher
**`src/marks.py:509-515`** (`is_de_pv` derivation) → `_match_german_pv` (**427-493**).
`is_de_pv` tests only the head string, not whether the row is PV; heads like `weg`/`an`/`aus` can get a
PV pair as first choice. Empirically 0/3298 de-nn rows affected today, so latent.
**Fix:** pass the row's own `is_pv` flag into `find_spans` (already carried by `data.py`)
and only run `_match_german_pv` for real PV rows.
**Severity: LOW**

### D-3. German ablaut table incomplete → root-changing forms never generated
**`src/marks.py:57-102`** `_IRREGULAR_DE` + `_german_base_forms` (**362-424**) — `brechen`-class verbs:
`bricht`, `gebrochen`, `brach` never emitted, so sentences containing them can't align for PV.
Zero frequency in the shipped `de-pv-train.tsv`, so latent.
**Fix:** add the strong verbs + `ge-` participles.
**Severity: LOW**

### D-4. Separated German particle matcher grabs homographic prepositions
**`src/marks.py:476-491`** — boundary-exact particle search with no role disambiguation; `an`/`aus`/
`bei`/`vor`/`zu` used as prepositions (`… an die Wand`) are grabbed as the head span, so the pooled
"head" token is wrong (no unligned-row fallback to flag it).
**Severity: LOW**

### D-5. `has_label` = mod label only, in joint mode
**`src/data.py:108`** — `bool(np.isfinite(mod_avg))`. In joint mode (`targets == []`, rows untouched by
`expand_targets`) a row with `HeadAvg` but no `ModAvg` is marked unlabeled → its head supervision is
dropped and it's excluded from val mod/head rho. `expand_targets` (`:294-296`) recomputes per-target
correctly; shipped NN files have 0 partial-label rows.
**Fix:** `bool(np.isfinite(mod_avg) or np.isfinite(head_avg))`.
**Severity: LOW**

### D-6. `collate_comp` pads with hard-coded `pad_token_id=0`
**`src/data.py:480-499`** — call sites (`trainer.py:170-178`, `run.py:124`) pass no override; silent
breakage for tokenizers whose `[PAD]` != 0. Fine for mmBERT/BERT.
**Fix:** thread `tokenizer.pad_token_id` through (mirror `mask_token_id` at `:332-339`).
**Severity: LOW**

### D-7. Same file loaded twice via different config attrs
**`src/data.py:126-154`** dedups by attribute string only, not file identity — an aux file whose name
collides with a core attr is appended again (`:181-186`, rebased `compound_id`), double-counting
gradient + inflating alignment reports.
**Severity: LOW** (no overlap in default configs)

### D-8. Ragged TSV rows become literal string `"nan"` data
**`src/data.py:70-73`** (`read_tsv`) + `:102` — short rows → float `NaN` cells despite
`keep_default_na=False`; `str(r['Context'])` yields `'nan'` and a garbage-looking labeled row is
trained on. No malformed rows in shipped files.
**Fix:** drop/coerce `df.isna()` cells in `_df_to_rows`.
**Severity: LOW**

### Checked and found OK
Token-offset mapping (`_char_to_token` 251-270, `tok_span` 529-535) — no off-by-one; degenerate
`end == len` case safe. Collation/`[MASK]` masking structurally sound. Split integrity
(`pipeline.py:52-66`, compound-grouped GSS + aux-leak filter) correct. Determinism reproducible.

---

## 3. TRAINING / EVAL code

### T-1. Phase-2 boundary overwrites the embedding group's LR with `encoder_lr`
**`src/trainer.py:375-381`** — the reset branches only on `tag == 'head'`; the embedding group
(`tag='frozen'`, **:229-231**) falls into the `else` and gets `encoder_lr`. With `embedding_lr > 0` the
embedding LR is silently changed for the whole LoRA phase.
**Fix:** three-way branch (`'head'`→`head_lr`, `'frozen'`→`embedding_lr`, else→`encoder_lr`).
**Severity: MED** (HIGH only when `embedding_lr>0`; default 0 unaffected)

### T-2. Embedding group also annealed 1→0 in phase 2
**`src/trainer.py:57-62`** `_phase2_lr_lambdas` decays every non-`'head'` group to zero, including a
trainable embedding group. Related to T-1 — fix both together.
**Severity: MED** (only when `embedding_lr>0`)

### T-3. Logged `lr` / `encoder_lr` name the wrong groups
**`src/train.py:218-219`** `last_lr = last_lrs[0]` and **:244-245** `encoder_lr = last_lrs[-1]`;
consumed at **`trainer.py:485-487, 530`**. Group order is `[emb, head, encoder]` (`trainer.py:229-236`),
so with `embedding_lr>0` the reported "head lr" is the embedding lr, and in phase 1 (no encoder group)
the reported "enc" lr is actually the head lr. Select by group tag, not index.
**Severity: MED** (logging only)

### T-4. Dead `len(preds_tuple) == 8` branch in train-rho unpacking
**`src/trainer.py:418-422`** — `train_epoch` always emits 9 or 10 items (`train.py:247-261`); the 8-item
branch (missing the aux filter `tr_core`) is unreachable but confusing.
**Severity: LOW**

### T-5. Duplicated head-schedule logic
**`src/trainer.py:49-55`** (`_phase2_lr_lambdas`) vs **:267-284** (inline phase-1 lambda builder) — same
cosine/linear/constant formula written twice; they must stay in sync.
**Severity: LOW**

### T-6. `GaussLoss` sigma floor hard-coded, not shared with `GaussHead`
**`src/losses.py:20, 31-33`** `_SIGMA_FLOOR = 0.05` vs `heads.py:8` `SIGMA_FLOOR = 0.04`; the loss clamps
sigma at 0.05 while heads emit ≥ 0.04. Negligible, but should be one constant.
**Severity: LOW**

### T-7. Dead/no-op code
- `trainer.py:20` `get_linear_schedule_with_warmup` imported, unused (only used by `run.py` smoke).
- `train.py:98` `mod_logits = head_logits = pv_logits = None` is a no-op for the only criterion in use.
- `trainer.py:496-498` `diag.get('nn_mod_loss', diag['mod_loss'])` fallback is dead — both keys always
  written (`train.py:238-243`).
- `train.py:22` unused module-level `logger`.
- `run.py:33-34` `_base_config()` returns `{}`.

### Checked and found OK
EMA (shadows registered after `apply_lora`, swap/restore via `finally`, keys match current
`_head_mod/_head_head/_head_pv` names). Optimizer group dedup. `scheduler.step()` decoupled from AMP
skips and `accum_steps` via the step-counter hook. `GaussLoss` mask/weight handling. `proto_rank_loss`
pair selection internally consistent when fed a row-correct cos tensor.

---

## 4. CONFIG code

### C-1. `head_lr_schedule` (and `warmup_ratio`, `head_lr_min_ratio`) never validated
**`src/config.py:132-133`** fields; `validate()` (**225-312**) checks every other enum-looking string but
not these. `Config(head_lr_schedule='stepped')` passes. Worse, an invalid value behaves **differently
per phase**: phase-1 treats anything ≠ cosine/linear as **constant** (`trainer.py:281-282`), phase-2
treats it as **linear** (`trainer.py:50-53`). A negative `head_lr_min_ratio` even yields **negative LR
multipliers** in phase 2.
**Fix:** `head_lr_schedule in ('constant','cosine','linear')`, `0 <= warmup_ratio <= 1`,
`0 <= head_lr_min_ratio <= 1`.
**Severity: HIGH** (silently changes training dynamics)

### C-2. `--set` for List/Tuple fields returns `list[str]`, bypassing normalization
**`src/config.py:328-332`** (`coerce_value`) + **`src/run.py:57`** (`Config(**cfg_dict)`) — the List/Tuple
branch splits on commas into **strings**; `from_dict`'s list→tuple conversion (`config.py:180-184`) is
skipped on the CLI path. `--set gauss_ctx_pv=21,22` → `['21','22']` (vs tuple of ints from `load()`),
and it survives `validate()`. `tests/smoke_src.py:141` **codifies exactly this behavior**.
**Fix:** route CLI dicts through `Config.from_dict`; coerce list elements by field type.
**Severity: MED**

### C-3. `model_backend='combined'` silently rewritten (authoritative type is fictional)
**`src/config.py:228-229`** — `validate()` mutates `'combined'`→`'twostream'`, so the `ModelBackend`
literal (`:18`) and `_MODEL_BACKENDS` (`:21`) partially lie, and `build_model`'s `'combined'` intent is
dead. **Low**: reject or drop from the literal.
**Severity: LOW**

### C-4. Unknown `--set` key crashes with `TypeError`, breaking `coerce_value`'s contract
**`src/run.py:57`** + docstring **`src/config.py:317-324`** — `coerce_value('not_a_field','1')` returns
`'1'` ("validated later"), but next line `Config(**cfg_dict)` raises `TypeError: unexpected keyword
argument`. Smoke test **:143** asserts the false promise.
**Fix:** validate keys against field names in `_build_config`.
**Severity: LOW**

### C-5. `Config.update()` merges raw lists, violating the `Tuple[int,...]` type
**`src/config.py:203-206`** — `from_dict(strict=True)` validates, then builds from `{**asdict(self),
**values}` with the raw list. Same class of bug as C-2 on the API path.
**Severity: LOW**

---

## 5. TOOLING / TESTS

### S-1. `tests/smoke_src.py` cannot pass; section 8 crashes the suite
- **:617-649** — ASCII-source guards search `model_combined.py` (a 15-line alias) for pre-refactor
  fragments (`'self.gauss = GaussHead(...)'`, `'pool_active(hidden, batch, targets)'`,
  `'self.shift_fuse = SemanticShiftFusion(...) if proto_stream else None'`, …). All missing → 6 checks
  always FAIL.
- **:816** — `from src.model_combined import CombinedBackboneModel` → `ImportError` (H-2). If that were
  fixed, **:837-838** `CombinedBackboneModel(..., proto_stream=True)` → `TypeError` — the ctor takes
  `shared_head`/`fusion_type`, not `proto_stream`. Section 8 therefore **crashes**.
- **:654-656** — trainer guard `"shift = getattr(model, 'shift_fuse', None)"` absent (trainer now uses
  `pred_heads()`), FAILs.
- **:840** — `model.shift_fuse is not None` is now vacuously true (property always returns `self.fusion`).
- **:1** — "NO torch needed" docstring false: sections 7/8 unconditionally `import torch`.
- **:285,296,…** — hard-coded exact row counts (3480/570/5862/445…) match current TSVs but break on any
  builder re-run.
**Fix:** rewrite sections 7/8 against the real `model_two_stream.py` API + guard torch imports, or drop.
**Severity: HIGH**

### S-2. `tools/*_aux.py` build brittleness (data-adjacent)
- `build_comp_en_aux.py:69-72` — bare `float(...)` on cells read with `dtype=str,
  keep_default_na=False`; any `''`/`'NA'` raises and kills the build.
- `build_comp_en_aux.py:86-91` — fused-single-token compounds can never re-align (head = whole
  compound).
- `build_ijcnlp_aux.py:38-39` — `_SPAN_RE` hard-codes `rgb(255,0,0)`; formatting drift collapses output.
- `build_nctti_aux.py:81-84,138-143` — compound-splitting regex duplicated and splits differently in two
  places.
- `build_litnlit_aux.py:77-87,95` — `particle_idx` collected, never used; **:158** alignment % includes
  unparseable blocks (understates true rate).
- `build_nctti_aux.py:107` — `pd.to_numeric(mrow.get(f'MeanS{n}'), errors='coerce')` raises when the
  column is absent (guard only checks the next line).
- `build_comp_en_aux.py:104` — dead empty `if m: pass` statement.
**Severity: LOW (correct-but-unmaintained; outputs still match `src/data.py` readers today)**

### S-3. `tools/ceiling.py` — orphan; label-pool mixing
`tools/ceiling.py:73-76` concatenates `ModAvg`+`HeadAvg` into one pool and averages summed row std²,
so `var_noise` under-estimates per-label noise (documented as a band-guess). No callers.
**Severity: LOW**

---

## 6. DEAD CODE (not bugs, removal candidates)

- `src/model.py` `MMBertModel` ('exits' backend) + `'combined'` write-off: legacy paths; `_features`
  computes `mod_exit_emb/head_exit_emb/pv_u/pv_v` that `_forward_gauss` (`:411-412`) discards.
- `src/model.py:84-116` `HybridSpanPool` — self-documented dead (kept for a future ablation).
- `src/model.py:41` re-exports `merge_lora` — no callers.
- `src/targets.py` `pool_span`/`pool_active`/`row_targets`/`target_selector` — used only by smoke tests
  (live pooling is `pool_active_context`, `model_two_stream.py:193`).
- `src/model_two_stream.py` direct two-stream API branch (`:313-321`) + `forward_stream_word`/
  `forward_stream_context` (`:227-265`) — unreachable from the batch path.
- `src/data.py` — `_need_torch()` (:41-43) never called; `row_id` always 0; `context_id` unused;
  `load_trial`/`_find_trial_path` (only notebook consumers).
- `src/run.py:33-34` `_base_config() → {}`.

---

## Quick-fix priority (my recommendation)

1. `lora._layer_idx`: match `layer` too (fixes the current `lora_from_layer: 14` run semantics).
2. `config.validate`: add `head_lr_schedule`/`warmup_ratio`/`head_lr_min_ratio` bounds.
3. `marks._alternate_forms`: always emit the regular `-ed/-ing` (+y-rules), add doubling only for true
   1-syllable CVC.
4. Delete `model_combined.py` + rewrite smoke tests 7/8.
5. `trainer` phase-2 boundary: 3-way LR reset respecting `embedding_lr` (fix T-1/T-2 together).
6. Freeze the no-longer-false `--set` coercion into `from_dict` (C-2).