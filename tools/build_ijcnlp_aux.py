#!/usr/bin/env python3
"""Build dataset/en-ijcnlp-aux.tsv from the Reddy, McCarthy & Manandhar
(IJCNLP 2011) English compound-noun compositionality resource shipped in
ijcnlp_compositionality_data/.

The resource ships:
  * MeanAndDeviations.clean.txt  90 compounds, per-constituent + whole-compound
                                 human means/stds on the 0-5 "how literal" scale
                                 (the same scale the competition NN labels use).
  * mturk_hits/<compound>.html   The actual evidence windows annotators scored;
                                 each <li> carries one real corpus sentence with
                                 the compound highlighted. (crocodile_tear is
                                 missing from mturk_hits -> 89 compounds used.)

Output is the NN schema (same as en-comp-aux.tsv, which is is_pv=False and
trains the mod/head exits): ContextID, Compound, Mod, Head, ModAvg, ModStd,
HeadAvg, HeadStd, Context. Labels are the lemma-level annotator means; Context
is each real evidence window in which the compound occurs (verbatim or plural).

Run from the repo root (requires the ijcnlp_compositionality_data/ clone):
    python tools/build_ijcnlp_aux.py
"""

from __future__ import annotations

import html as _html
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'ijcnlp_compositionality_data' / 'ijcnlp_compositionality_data'
OUT = ROOT / 'dataset' / 'en-ijcnlp-aux.tsv'

# highlighted compound inside the evidence windows
_SPAN_RE = re.compile(
    r'color:\s*rgb\(255,\s*0,\s*0\);[^>]*><b>\s*(.*?)\s*</b>', re.S)
# one evidence sentence per <li>
_LI_RE = re.compile(r'<li>(.*?)</li>', re.S)


def _clean(li: str) -> str:
    """Strip tags from one evidence window and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', li)
    text = _html.unescape(text)
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    # baked-in double-encoding artifacts (â€TM -> apostrophe, etc.)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main() -> int:
    mean_lines = (SRC / 'MeanAndDeviations.clean.txt').read_text(
        encoding='utf-8', errors='replace').splitlines()

    rows = []
    skipped_compounds = []
    for line in mean_lines[1:]:
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        word, score_block = parts
        scores = score_block.split()
        if len(scores) != 7:  # W1m W1s W2m W2s Cm Cs m1*mean2
            continue
        comp_words = [w for w in word.split()]
        if len(comp_words) != 2:
            continue
        mod = comp_words[0].rsplit('-', 1)[0].strip()
        head = comp_words[1].rsplit('-', 1)[0].strip()
        compound = f'{mod} {head}'
        if not mod or not head:
            continue

        w1m, w1s, w2m, w2s = map(float, scores[:4])

        pfx = '_'.join(w.rsplit('-', 1)[0] for w in comp_words)
        hit = SRC / 'mturk_hits' / f'{pfx}.html'
        if not hit.is_file():
            skipped_compounds.append(compound)
            continue

        text = hit.read_text(encoding='utf-8', errors='replace')
        n_contexts = 0
        for li in _LI_RE.findall(text):
            ctx = _clean(li)
            if not ctx or _SPAN_RE.search(li) is None:
                # skip the dictionary definition <li> (no highlight)
                continue
            rows.append({
                'ContextID': f'IJCNLP-{len(rows):05d}',
                'Compound': compound,
                'Mod': mod,
                'Head': head,
                'ModAvg': f'{w1m:.4f}',
                'ModStd': f'{w1s:.4f}',
                'HeadAvg': f'{w2m:.4f}',
                'HeadStd': f'{w2s:.4f}',
                'Context': ctx,
            })
            n_contexts += 1
        if n_contexts == 0:
            skipped_compounds.append(compound)

    out = pd.DataFrame(rows, columns=[
        'ContextID', 'Compound', 'Mod', 'Head',
        'ModAvg', 'ModStd', 'HeadAvg', 'HeadStd', 'Context'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep='\t', index=False)

    print(f'wrote {len(out)} rows -> {OUT}')
    print('compounds in file:', out['Compound'].nunique())
    print('skipped compounds (no mturk html / no evidence):', len(skipped_compounds))
    print('   ', skipped_compounds)
    print('rows/compound mean: %.1f' % (len(out) / max(out['Compound'].nunique(), 1)))
    print('ModAvg %.2f..%.2f | HeadAvg %.2f..%.2f' % (
        float(out['ModAvg'].min()), float(out['ModAvg'].max()),
        float(out['HeadAvg'].min()), float(out['HeadAvg'].max())))
    return 0


if __name__ == '__main__':
    sys.exit(main())