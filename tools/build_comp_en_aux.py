#!/usr/bin/env python3
"""Build dataset/en-comp-aux.tsv from the Cordeiro et al. (2019) lemma-level
compositionality dataset shipped in comp-datasets-release-v2/.

The resource judges ~190 English nominal compounds OUT of context (one averaged
modifier/head score per lemma). It ships 3 example sentences per compound, which
we reuse as real CONTEXTS, so each compound yields 3 rows in the competition
en-nn schema (ContextID, Compound, Mod, Head, ModAvg, ModStd, HeadAvg, HeadStd,
Context). The label is the compound's averaged human judgment (same for all 3
rows). Rows are aux (is_aux=True at load time via config train_aux), so the
80/20 compound split never selects them for the holdout.

Span alignment: the compound is marked inline with <strong>/<b> in the example
sentences; the tagged surface is split into Mod (all but the last token) and
Head (the last token), which is exactly how marks.find_spans re-finds them.

Run from the repo root (requires the comp-datasets-release-v2/ clone):
    python tools/build_comp_en_aux.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import html as _html
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'comp-datasets-release-v2' / 'comp-datasets-release-v2'
OUT = ROOT / 'dataset' / 'en-comp-aux.tsv'

SETS = (
    ('annotations/filtered/en.filtered.csv', 'compounds-lists/compounds-list-en.tsv'),
    ('annotations/filtered/en-extra.filtered.csv', 'compounds-lists/compounds-list-en-extra.tsv'),
)

_TAG_RE = re.compile(r'<(?:strong|b)>(.*?)</(?:strong|b)>', re.IGNORECASE)
_ANY_TAG_RE = re.compile(r'<[^>]+>')
_SENT_COLS = ('examplesent1', 'examplesent2', 'examplesent3')


def clean_tags(text: str) -> str:
    """Remove HTML tags and unescape entities, keeping the compound intact."""
    return _html.unescape(_ANY_TAG_RE.sub('', text)).strip()


def main() -> int:
    for ann_path, list_path in SETS:
        if not (SRC / ann_path).is_file() or not (SRC / list_path).is_file():
            print(f'ERROR: missing {ann_path} or {list_path} under {SRC}', file=sys.stderr)
            return 1

    rows = []
    no_markup = 0
    multiword = 0
    for ann_path, list_path in SETS:
        ann = pd.read_csv(SRC / ann_path, sep='\t', dtype=str, keep_default_na=False)
        ann.columns = [c.strip() for c in ann.columns]
        sheets = pd.read_csv(SRC / list_path, sep='\t', dtype=str, keep_default_na=False)
        sheets.columns = [c.strip() for c in sheets.columns]
        sheets['key'] = sheets['compound'].str.strip().str.lower()
        ann['key'] = ann['compound_lemma'].str.replace('_', ' ').str.strip().str.lower()
        merged = ann.merge(sheets[['key'] + list(_SENT_COLS)], on='key', how='left')

        for _, r in merged.iterrows():
            surface = str(r['compound_surface']).replace('_', ' ')
            mod_avg = float(r['avgModifier'])
            mod_std = float(r['stdevModifier'])
            head_avg = float(r['avgHead'])
            head_std = float(r['stdevHead'])
            for col in _SENT_COLS:
                raw = str(r.get(col, '')).strip()
                if not raw or raw.lower() == 'nan':
                    continue
                m = _TAG_RE.search(raw)
                comp_tags = m.group(1).strip() if m else ''
                context = clean_tags(raw)
                if not context:
                    continue
                if not comp_tags:
                    no_markup += 1
                    # fall back to the annotated lemma surface
                    comp_tags = surface
                parts = comp_tags.split()
                if len(parts) < 1:
                    continue
                head = parts[-1]
                mod = ' '.join(parts[:-1]) if len(parts) > 1 else str(r['modifierLemma'])
                if len(parts) > 2:
                    multiword += 1
                rows.append({
                    'ContextID': f'COMPEN-{len(rows):04d}',
                    'Compound': comp_tags,
                    'Mod': mod,
                    'Head': head,
                    'ModAvg': f'{mod_avg:.4f}',
                    'ModStd': f'{mod_std:.4f}',
                    'HeadAvg': f'{head_avg:.4f}',
                    'HeadStd': f'{head_std:.4f}',
                    'Context': context,
                })
                if m:  # every row still keeps the marked compound inline
                    pass

    out = pd.DataFrame(rows, columns=[
        'ContextID', 'Compound', 'Mod', 'Head', 'ModAvg', 'ModStd', 'HeadAvg',
        'HeadStd', 'Context'])

    # Sanity: the marked compound survives the tag-stripping verbatim.
    aligned = 0
    for _, r in out.iterrows():
        c = r['Compound']
        if c and c.lower() in clean_tags(r['Context']).lower():
            aligned += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep='\t', index=False)

    print(f'wrote {len(out)} rows -> {OUT}')
    print('aux compounds:', out['Compound'].str.lower().nunique())
    print('rows with compound verbatim in context:', aligned, f'({100.0 * aligned / len(out):.1f}%)' if len(out) else '')
    print(f'(sentences without inline markup: {no_markup}, multi-word compounds: {multiword})')
    return 0


if __name__ == '__main__':
    sys.exit(main())