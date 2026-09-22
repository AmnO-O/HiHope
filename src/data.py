"""Data loading, tokenization and span alignment for the gauss pipeline.

Marker-free design (phase M2): the raw context is tokenized verbatim with
``return_offsets_mapping=True``; the compound's Mod/Head surface forms are
matched inside the original text by ``marks.find_spans`` and mapped onto
token spans. No ``<mod>`` / ``<head>`` markers are inserted, so the pretrained
mmBERT tokenizer/embeddings never see out-of-vocabulary artifacts.

``CompDataset`` -- one row per labeled / aux sentence, for scoring. Yields
span masks plus (possibly NaN) soft labels; aux and unlabeled rows keep the
row but mark ``has_label`` False so losses can mask them. Rows whose span
cannot be aligned still train: the model falls back to whole-sentence pooling
for those positions. Rows pre-tokenize and pre-align in the constructor
(deterministic, single pass), which also surfaces a per-source alignment
report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .marks import Span, SpanResult, find_spans
from .prototype_prompt import format_prototype_text
from .targets import TARGETS, target_code
from .utils import get_logger

_TARGET_CODE = {t: target_code(t) for t in TARGETS}

logger = get_logger('src.data')

try:
    import torch
except ImportError:  # pragma: no cover - datasets need torch; loaders don't
    torch = None


def _need_torch():
    if torch is None:
        raise RuntimeError('torch is required to build torch Datasets')


_DatasetBase = torch.utils.data.Dataset if torch is not None else object


def _is_nn(df: pd.DataFrame) -> bool:
    return 'Mod' in df.columns and 'Head' in df.columns


def _is_pv(df: pd.DataFrame) -> bool:
    return 'Base' in df.columns and 'Particle' in df.columns


def _auto_lang(path: str) -> str:
    name = Path(path).name.lower()
    return 'de' if name.startswith('de-') else 'en'


def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a dataset TSV as raw strings (numeric coercion happens per row)."""
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    return df


def _df_to_rows(df: pd.DataFrame, tag: str, lang: str) -> List[Dict]:
    """Normalize an NN or PV dataframe to row dicts (see module docstring)."""
    nn = _is_nn(df)
    pv = _is_pv(df)
    if not nn and not pv:
        raise ValueError(
            f'{tag}: expected NN columns (Compound/Mod/Head) or PV columns '
            f'(ParticleVerb/Base/Particle), got {list(df.columns)}'
        )

    rows: List[Dict] = []
    for _, r in df.iterrows():
        if nn:
            mod, head, compound = r['Mod'], r['Head'], r.get('Compound', '')
        else:
            mod, head, compound = r['Base'], r['Particle'], r.get('ParticleVerb', '')

        def val(col):
            return _f(r[col]) if col in r else float('nan')

        mod_avg = val('ModAvg' if nn else 'Avg')
        head_avg = val('HeadAvg' if nn else 'Avg')
        mod_std = val('ModStd' if nn else 'Std')
        head_std = val('HeadStd' if nn else 'Std')

        ctx = r['Context']
        ctx_str = str(ctx) if (ctx is not None and pd.notna(ctx)) else ''

        rows.append({
            'context': ctx_str,
            'mod': str(mod) if (mod is not None and pd.notna(mod)) else '',
            'head': str(head) if (head is not None and pd.notna(head)) else '',
            'compound': str(compound) if (compound is not None and pd.notna(compound)) else '',
            'lang': lang,
            'is_pv': pv,
            'has_label': bool(np.isfinite(mod_avg) or np.isfinite(head_avg)),
            'mod_avg': mod_avg,
            'head_avg': head_avg,
            'mod_std': mod_std,
            'head_std': head_std,
            'compound_id': -1,
            'context_id': str(r.get('ContextID', '')),
        })
    return rows


# --------------------------------------------------------------------------- #
# top-level loaders
# --------------------------------------------------------------------------- #
def _load_files(cfg, file_attrs: List[str]) -> List[Dict]:
    """Helper nạp và gộp nhiều file TSV (EN/DE, NN/PV) theo cấu hình Config."""
    data_dir, _ = _resolve(cfg)
    
    files_to_load: List[str] = []
    for attr in file_attrs:
        fname = getattr(cfg, attr, None)
        if fname and fname.strip() and fname not in files_to_load:
            files_to_load.append(fname)

    all_rows: List[Dict] = []
    for fname in files_to_load:
        path = data_dir / fname
        if not path.exists():
            logger.warning('Dataset file missing, skipping: %s', path)
            continue
        df = read_tsv(path)
        rows = _df_to_rows(df, fname, _auto_lang(fname))
        all_rows.extend(rows)
        logger.info('%s: %d rows', fname, len(rows))

    if not all_rows:
        raise FileNotFoundError(f'No valid dataset files found in {data_dir}')

    # Định danh compound_id duy nhất kèm ngôn ngữ (vd: en_blackboard vs de_apfelbaum)
    compound_keys = [f"{r['lang']}_{r['compound']}" for r in all_rows]
    codes, _ = pd.factorize(pd.Series(compound_keys))
    for r, c in zip(all_rows, codes):
        r['compound_id'] = int(c)

    logger.info('Loaded %d total rows across %d unique compounds', len(all_rows), int(codes.max()) + 1)
    return all_rows


def load_labeled(cfg) -> List[Dict]:
    """Load all configured NN and PV training datasets (+ train_aux extras)."""
    train_attrs = ['en_nn_train', 'de_nn_train', 'en_pv_train', 'de_pv_train']
    all_rows = _load_files(cfg, train_attrs)
    aux_names = [a.strip() for a in (getattr(cfg, 'train_aux', None) or [])
                 if a and a.strip()] or []
    if aux_names:
        data_dir, _ = _resolve(cfg)
        n_aux = 0
        for fname in aux_names:
            path = _resolve_aux_path(fname, data_dir)
            if path is None:
                logger.warning('Aux dataset file not found (data_dir + repo dataset/): %s', fname)
                continue
            df = read_tsv(path)
            rows = _df_to_rows(df, fname, _auto_lang(fname))
            for r in rows:
                r['is_aux'] = True
            all_rows.extend(rows)
            n_aux += len(rows)
            logger.info('%s: %d aux rows', fname, len(rows))
        if n_aux > 0:
            # aux rows are new compounds: key them in a dedicated id range ABOVE all
            # core ids so the batch/ranking compound grouping can never mix aux and
            # core compounds together under one compound_id.
            core_ids = [r['compound_id'] for r in all_rows[: len(all_rows) - n_aux]]
            base = (max(core_ids) + 1) if core_ids else 0
            aux_keys = [f"{r['lang']}_{r['compound']}" for r in all_rows[len(all_rows) - n_aux:]]
            aux_codes, _ = pd.factorize(pd.Series(aux_keys))
            for r, c in zip(all_rows[len(all_rows) - n_aux:], aux_codes):
                r['compound_id'] = base + int(c)
            logger.info('Loaded %d rows (incl %d aux) across %d core + %d aux compounds',
                        len(all_rows), n_aux, len(core_ids), int(aux_codes.max()) + 1)
        else:
            logger.warning('train_aux configured but NO aux rows loaded (%s); continuing core-only',
                           ', '.join(aux_names))
    return all_rows


def _resolve_aux_path(fname: str, data_dir: Path) -> Optional[Path]:
    """Locate a train_aux file: data_dir first, then the repo/working dataset/ dir.

    On Kaggle the core TSVs are mounted under the dataset input, while built aux
    files live in the repo clone, so a plain ``data_dir / fname`` misses them.
    """
    candidates = [
        data_dir / fname,
        data_dir / 'dataset' / fname,
        data_dir.parent / 'dataset' / fname,
        Path('dataset') / fname,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_trial_path(fname: str, data_dir: Path) -> Optional[Path]:
    """Find a trial file in trial/ (sibling of dataset/ on Kaggle or root) or data_dir."""
    candidates = [
        data_dir / fname,
        data_dir / 'trial' / fname,
        data_dir.parent / 'trial' / fname,
        data_dir.parent / fname,
        Path('trial') / fname,
        Path('dataset') / fname,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_trial(cfg) -> Dict[str, List[Dict]]:
    """Load the per-lineage TRIAL (or Test) files -> {key: rows}.

    ``key`` is one of ``en-nn`` / ``en-pv`` / ``de-nn`` / ``de-pv``, read from the
    four ``*_trial`` config attrs. Rows keep the file's row order and get a
    per-lineage ``compound_id`` (paths don't participate in train-time splits).
    """
    data_dir, _ = _resolve(cfg)
    lineage_attrs = (
        ('en-nn', 'en_nn_trial'), ('en-pv', 'en_pv_trial'),
        ('de-nn', 'de_nn_trial'), ('de-pv', 'de_pv_trial'),
    )
    out: Dict[str, List[Dict]] = {}
    for key, attr in lineage_attrs:
        fname = (getattr(cfg, attr, '') or '').strip()
        if not fname:
            continue
        path = _find_trial_path(fname, data_dir)
        if path is None:
            logger.warning('%s file missing, skipping: %s', key, fname)
            continue
        rows = _df_to_rows(read_tsv(path), fname, _auto_lang(fname))
        codes, _ = pd.factorize(pd.Series(
            [f"{r['lang']}_{r['compound']}" for r in rows]))
        for r, c in zip(rows, codes):
            r['compound_id'] = int(c)
        out[key] = rows
        logger.info('%s trial: %d rows from %s', key, len(rows), path)

    return out


def _resolve(cfg):
    from .utils import resolve_paths
    return resolve_paths(cfg)


# --------------------------------------------------------------------------- #
# single-target expansion
# --------------------------------------------------------------------------- #
def expand_targets(rows: List[Dict], targets: List[str]) -> List[Dict]:
    """Expand rows into one raw per active target (single-target design).

    A row is a (source sentence + spans + labels) candidate for the targets
    that actually make sense for its type -- NN rows grade the modifier and
    the head noun separately, PV rows grade only the whole phrasal verb:

        NN row  + 'mod'  => kept   (label ModAvg)
        NN row  + 'head' => kept   (label HeadAvg)
        NN row  + 'pv'   => DROPPED (no overall Avg gold for NN compounds)
        PV row  + 'pv'   => kept   (label Avg)
        PV row  + 'mod'/'head' => DROPPED (PV rows have no per-role gold)

    Each kept row carries a ``'target'`` key; ``has_label`` reflects whether
    the gold for that target is present. An empty ``targets`` list returns the
    rows untouched (joint mode: one row scores all targets at once).
    """
    if not targets:
        return rows

    out: List[Dict] = []
    for r in rows:
        nn = not r.get('is_pv', False)
        for t in targets:
            if (nn and t == 'mod') or (nn and t == 'head'):
                has_label = bool(np.isfinite(r['mod_avg' if t == 'mod' else 'head_avg']))
            elif (not nn) and t == 'pv':
                has_label = bool(np.isfinite(r['mod_avg']))   # Avg lives in mod_avg
            else:
                continue                                       # drop entirely
            c = dict(r)
            c['target'] = t
            c['has_label'] = has_label
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _span_mask(span: Optional[Span], length: int) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if span is not None and span.start is not None:
        mask[span.start:span.end] = True
    return mask


class CompDataset(_DatasetBase):
    """One sentence (tokenized verbatim) per row, for scoring."""

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 256,
                 is_test: bool = False, proto_stream: bool = False,
                 target_mask_prob: float = 0.0,
                 proto_mode: str = "hybrid",
                 max_proto_length: int = 32):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.proto_stream = proto_stream
        self.target_mask_prob = float(target_mask_prob)
        self.proto_mode = proto_mode
        self.max_proto_length = int(max_proto_length)
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if self.mask_token_id is None and hasattr(tokenizer, 'convert_tokens_to_ids'):
            try:
                mid = tokenizer.convert_tokens_to_ids('[MASK]')
                if isinstance(mid, int) and mid > 0:
                    self.mask_token_id = mid
            except Exception:
                pass
        self.items = [self._encode(r) for r in rows]
        self._report()

    def _encode(self, r: Dict) -> Dict:
        enc = self.tokenizer(
            r['context'], max_length=self.max_len, truncation=True,
            return_tensors='pt', return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        offsets = enc['offset_mapping'].squeeze(0).tolist()
        length = input_ids.size(0)

        result: SpanResult = find_spans(
            r['context'], offsets, r['mod'], r['head'], r.get('compound', ''),
            is_pv=r.get('is_pv', False)
        ) if (r['mod'] and r['head']) \
            else SpanResult(Span(None, None), Span(None, None), found=False)

        mod_span_mask = _span_mask(result.mod, length)
        head_span_mask = _span_mask(result.head, length)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mod_span_mask': mod_span_mask,
            'head_span_mask': head_span_mask,
            'has_mod': torch.tensor(result.mod.start is not None, dtype=torch.bool),
            'has_head': torch.tensor(result.head.start is not None, dtype=torch.bool),
            'degenerate': torch.tensor(result.degenerate, dtype=torch.bool),
            'compound_id': torch.tensor(int(r['compound_id']), dtype=torch.long),
            'has_label': torch.tensor(bool(r['has_label']), dtype=torch.bool),
            'mod_avg': torch.tensor(float(r['mod_avg']), dtype=torch.float),
            'head_avg': torch.tensor(float(r['head_avg']), dtype=torch.float),
            'mod_std': torch.tensor(float(r['mod_std']), dtype=torch.float),
            'head_std': torch.tensor(float(r['head_std']), dtype=torch.float),
            'row_id': torch.tensor(int(r.get('row_id', 0)), dtype=torch.long),
            'is_pv': torch.tensor(bool(r.get('is_pv', False)), dtype=torch.bool),
            'is_aux': torch.tensor(bool(r.get('is_aux', False)), dtype=torch.bool),
        }
        if self.proto_stream:
            # Stream 1: Tokenize isolated constituent words for prototype representations
            t = r.get('target')
            is_pv = bool(r.get('is_pv', False))
            mod_word = r.get('mod', '') or ''
            head_word = r.get('head', '') or ''
            compound_word = r.get('compound', '') or ''
            pv_word = compound_word or (f"{mod_word} {head_word}".strip())

            if t == 'mod':
                word = mod_word
            elif t == 'head':
                word = head_word
            elif t == 'pv':
                word = pv_word
            elif is_pv:
                # For PV rows, the prototype must represent the full particle verb
                word = pv_word
            else:
                word = mod_word or compound_word

            lang = str(r.get('lang', 'en')).lower()

            def _tok_word(w: str, target_is_pv: bool = False,
                          target_is_verb: bool = False, target_is_particle: bool = False):
                if w and hasattr(self.tokenizer, '__call__'):
                    text = format_prototype_text(
                        w, lang=lang, is_pv=target_is_pv,
                        is_verb=target_is_verb, is_particle=target_is_particle,
                        mode=self.proto_mode
                    )
                    try:
                        p = self.tokenizer(
                            text, max_length=self.max_proto_length, truncation=True,
                            return_tensors='pt', return_offsets_mapping=True,
                        )
                        offsets = p.get('offset_mapping')
                        if offsets is not None:
                            offsets = offsets.squeeze(0).tolist()
                    except Exception:
                        p = self.tokenizer(
                            text, max_length=self.max_proto_length, truncation=True,
                            return_tensors='pt',
                        )
                        offsets = None

                    p_ids = p['input_ids'].squeeze(0)
                    p_mask = p['attention_mask'].squeeze(0)
                    p_span = torch.zeros_like(p_mask, dtype=torch.bool)

                    if offsets is not None and w and text:
                        quoted = f"'{w}'"
                        if quoted in text:
                            sc = text.rfind(quoted) + 1
                            ec = sc + len(w)
                        elif text.startswith(f"{w}:"):
                            sc, ec = 0, len(w)
                        else:
                            sc = text.rfind(w)
                            ec = sc + len(w) if sc != -1 else -1

                        if sc != -1 and ec > sc:
                            for j, (cs, ce) in enumerate(offsets):
                                if cs < ec and ce > sc:
                                    p_span[j] = True

                    if not p_span.any():
                        l_int = int(p_mask.sum().item())
                        p_span = p_mask.bool().clone()
                        if l_int >= 3:
                            p_span[0] = False
                            p_span[l_int - 1] = False

                    return p_ids, p_mask, p_span
                return (
                    torch.zeros(1, dtype=input_ids.dtype),
                    torch.zeros(1, dtype=attention_mask.dtype),
                    torch.zeros(1, dtype=torch.bool),
                )

            p_ids, p_mask, p_span = _tok_word(word, target_is_pv=(t == 'pv' or is_pv))
            item['proto_ids'] = p_ids
            item['proto_mask'] = p_mask
            item['proto_span_mask'] = p_span

            # Provide explicit separate prototypes for joint multi-target mode
            m_ids, m_mask, m_span = _tok_word(mod_word, target_is_pv=False, target_is_verb=is_pv)
            h_ids, h_mask, h_span = _tok_word(head_word, target_is_pv=False, target_is_particle=is_pv)
            item['mod_proto_ids'] = m_ids
            item['mod_proto_mask'] = m_mask
            item['mod_proto_span_mask'] = m_span
            item['head_proto_ids'] = h_ids
            item['head_proto_mask'] = h_mask
            item['head_proto_span_mask'] = h_span
        if r.get('target') is not None:
            item['target'] = torch.tensor(_TARGET_CODE[r['target']], dtype=torch.long)
        return item

    def _report(self) -> None:
        if self.is_test:
            return
        n = len(self.items) or 1
        found = sum(1 for it in self.items if bool(it['has_mod'] and it['has_head']))
        deg = sum(1 for it in self.items if bool(it['degenerate']))
        lbl = sum(1 for it in self.items if bool(it['has_label']))
        logger.info(
            'CompDataset: %d rows, %d%% aligned, %d degenerate, %d labeled',
            len(self.items), int(100 * found / n), deg, lbl,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        if not self.is_test and self.target_mask_prob > 0.0 and self.mask_token_id is not None:
            if torch.rand(1).item() < self.target_mask_prob:
                item = dict(item)
                new_input_ids = item['input_ids'].clone()
                t = item.get('target')
                if t is not None:
                    t_val = int(t.item()) if hasattr(t, 'item') else int(t)
                    if t_val == 0:
                        span = item['mod_span_mask']
                    elif t_val == 1:
                        span = item['head_span_mask']
                    else:
                        span = item['mod_span_mask'] | item['head_span_mask']
                else:
                    # In joint mode, mask modifier OR head alternately (50/50), never both
                    # at the same time so context retainment remains intact.
                    if item['mod_span_mask'].any() and item['head_span_mask'].any():
                        span = item['mod_span_mask'] if torch.rand(1).item() < 0.5 else item['head_span_mask']
                    elif item['mod_span_mask'].any():
                        span = item['mod_span_mask']
                    else:
                        span = item['head_span_mask']

                if span.any():
                    new_input_ids[span] = self.mask_token_id
                    item['input_ids'] = new_input_ids
        return item


# --------------------------------------------------------------------------- #
# collate helpers (pad to max length within the batch)
# --------------------------------------------------------------------------- #
def collate_comp(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    seq_keys = (
        'input_ids', 'attention_mask', 'mod_span_mask', 'head_span_mask',
        'proto_ids', 'proto_mask', 'proto_span_mask',
        'mod_proto_ids', 'mod_proto_mask', 'mod_proto_span_mask',
        'head_proto_ids', 'head_proto_mask', 'head_proto_span_mask',
    )
    for key in batch[0]:
        if key in seq_keys:
            length = max(int(b[key].size(0)) for b in batch)
            fill = pad_token_id if key in ('input_ids', 'proto_ids', 'mod_proto_ids', 'head_proto_ids') else 0
            out[key] = torch.full((len(batch), length), fill_value=fill,
                                  dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                n = int(b[key].size(0))
                out[key][i, :n] = b[key]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
