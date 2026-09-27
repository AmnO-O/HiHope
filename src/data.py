"""Data loading, tokenization and span alignment for Target-Aware Query Attention.

Natural sentence context is tokenized verbatim. Subtoken spans for Mod and Head
are aligned without inserting artificial delimiters. Token roles are mapped to
role IDs:
  - 0: Context
  - 1: Mod
  - 2: Head
  - 3: Compound / Phrasal Verb
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .marks import Span, SpanResult, find_spans
from .targets import TARGETS, target_code
from .utils import get_logger

_TARGET_CODE = {t: target_code(t) for t in TARGETS}

logger = get_logger('src.data')

try:
    import torch
    _DatasetBase = torch.utils.data.Dataset
except ImportError:  # pragma: no cover
    torch = None
    _DatasetBase = object


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
    """Read a dataset TSV as raw strings."""
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def _df_to_rows(df: pd.DataFrame, tag: str, lang: str) -> List[Dict]:
    """Normalize an NN or PV dataframe to standardized row dicts."""
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

        ctx = r.get('Context', '')
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
            'filename': str(tag),
        })
    return rows


def _load_files(cfg, file_attrs: List[str]) -> List[Dict]:
    """Helper to load and merge multiple dataset TSVs."""
    from .utils import resolve_paths
    data_dir, _ = resolve_paths(cfg)
    
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

    compound_keys = [f"{r['lang']}_{r['compound']}" for r in all_rows]
    codes, _ = pd.factorize(pd.Series(compound_keys))
    for r, c in zip(all_rows, codes):
        r['compound_id'] = int(c)

    logger.info('Loaded %d total rows across %d unique compounds', len(all_rows), int(codes.max()) + 1)
    return all_rows


def load_labeled(cfg) -> List[Dict]:
    """Load all configured NN and PV training datasets (+ optional train_aux extras)."""
    train_attrs = ['en_nn_train', 'de_nn_train', 'en_pv_train', 'de_pv_train']
    if getattr(cfg, 'use_aux', False) or getattr(cfg, 'train_aux', None):
        train_attrs.append('train_aux')
    return _load_files(cfg, train_attrs)


def load_trial(cfg) -> Dict[str, List[Dict]]:
    """Load the per-lineage TRIAL files -> {key: rows}."""
    from .utils import resolve_paths
    data_dir, _ = resolve_paths(cfg)
    lineage_attrs = (
        ('en-nn', 'en_nn_trial'), ('en-pv', 'en_pv_trial'),
        ('de-nn', 'de_nn_trial'), ('de-pv', 'de_pv_trial'),
    )
    out: Dict[str, List[Dict]] = {}
    for key, attr in lineage_attrs:
        fname = (getattr(cfg, attr, '') or '').strip()
        if not fname:
            continue
        path = data_dir / fname
        if not path.is_file():
            continue
        rows = _df_to_rows(read_tsv(path), fname, _auto_lang(fname))
        codes, _ = pd.factorize(pd.Series([f"{r['lang']}_{r['compound']}" for r in rows]))
        for r, c in zip(rows, codes):
            r['compound_id'] = int(c)
        out[key] = rows
        logger.info('%s trial: %d rows from %s', key, len(rows), path)

    return out


def expand_targets(rows: List[Dict], targets: List[str]) -> List[Dict]:
    """Expand rows into one row per active target."""
    if not targets:
        return rows

    out: List[Dict] = []
    for r in rows:
        nn = not r.get('is_pv', False)
        for t in targets:
            if (nn and t == 'mod') or (nn and t == 'head'):
                has_label = bool(np.isfinite(r['mod_avg' if t == 'mod' else 'head_avg']))
            elif (not nn) and t == 'pv':
                has_label = bool(np.isfinite(r['mod_avg']))
            else:
                continue
            c = dict(r)
            c['target'] = t
            c['has_label'] = has_label
            out.append(c)
    return out


def _span_mask(span: Optional[Span], length: int) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if span is not None and span.start is not None:
        mask[span.start:span.end] = True
    return mask


class CompDataset(_DatasetBase):
    """One sentence (tokenized verbatim) per row with explicit role tagging."""

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 256, is_test: bool = False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.items = [self._encode(r) for r in rows]

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

        # Build explicit role IDs tensor [S]: 0: Context, 1: Mod, 2: Head, 3: Compound
        role_ids = torch.zeros_like(input_ids, dtype=torch.long)
        role_ids = torch.where(mod_span_mask, torch.full_like(role_ids, 1), role_ids)
        role_ids = torch.where(head_span_mask, torch.full_like(role_ids, 2), role_ids)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'role_ids': role_ids,
            'mod_span_mask': mod_span_mask,
            'head_span_mask': head_span_mask,
            'has_mod': torch.tensor(result.mod.start is not None, dtype=torch.bool),
            'has_head': torch.tensor(result.head.start is not None, dtype=torch.bool),
            'compound_id': torch.tensor(int(r['compound_id']), dtype=torch.long),
            'has_label': torch.tensor(bool(r['has_label']), dtype=torch.bool),
            'mod_avg': torch.tensor(float(r['mod_avg']), dtype=torch.float),
            'head_avg': torch.tensor(float(r['head_avg']), dtype=torch.float),
            'mod_std': torch.tensor(float(r['mod_std']), dtype=torch.float),
            'head_std': torch.tensor(float(r['head_std']), dtype=torch.float),
            'row_id': torch.tensor(int(r.get('row_id', 0)), dtype=torch.long),
            'is_pv': torch.tensor(bool(r.get('is_pv', False)), dtype=torch.bool),
        }
        if r.get('target') is not None:
            item['target'] = torch.tensor(_TARGET_CODE[r['target']], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_comp(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequence tensors within a batch."""
    out: Dict[str, torch.Tensor] = {}
    seq_keys = ('input_ids', 'attention_mask', 'role_ids', 'mod_span_mask', 'head_span_mask')
    for key in batch[0]:
        if key in seq_keys:
            length = max(int(b[key].size(0)) for b in batch)
            fill = pad_token_id if key == 'input_ids' else 0
            out[key] = torch.full((len(batch), length), fill_value=fill, dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                n = int(b[key].size(0))
                out[key][i, :n] = b[key]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
