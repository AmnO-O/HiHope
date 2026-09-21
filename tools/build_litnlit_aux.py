#!/usr/bin/env python3
"""Build dataset/de-litnlit-aux.tsv from the Koeper & Schulte im Walde
(NAACL-HLT 2016) German particle-verb literalness resource shipped in
pv_litnlit/.

The resource provides 6436 human-annotated sentences (3+ native speakers) for
159 separable German particle verbs. Each sentence block is:

  LITERAL|NON-LITERAL\\t<4 scores 0..5>     (0 = literal, 5 = non-literal)
   <idx>\\t<FORM>\\t[X|_]\\t<LEMMA>\\t...     (CoNLL-style, 14 fields)

``X`` marks the base-verb token; when the verb is syntactically separated
(e.g. "biegen ... ab") the particle is a separate PTKVZ token, otherwise the
particle is fused onto the base form (e.g. "abgesaeh... abgesägt").

The competition's de-pv Avg runs literal=high / idiomatic=low, which is the
OPPOSITE polarity of this resource's literalness scores, so every label is
inverted:  aux_avg = 5 - mean(4 scores), and Std is the real spread of the
4 per-sentence scores (no borrowed proxy).

Output schema matches de-pv-train.tsv (PV rows -> is_pv=True -> trains the
overall gauss head): ContextID, ParticleVerb, Base, Particle, Avg, Std,
Context. Only BASE verbs that _match_german_pv can find (fused or separated)
are kept, so every row exercises the PV head the same way competition rows do.

Run from the repo root (requires the pv_litnlit/ clone):
    python tools/build_litnlit_aux.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / 'pv_litnlit' / 'dataset'
OUT = ROOT / 'dataset' / 'de-litnlit-aux.tsv'

# Separable particles present in the corpus (10). Longest-first so "zurück"
# beats "zu" etc.; only these prefixes appear in the file names.
PARTICLES = (
    "auf", "aus", "durch", "zurück", "ab", "an", "ein", "mit", "nach",
    "vor", "zu",
)


def _split_particle_verb(verb: str) -> tuple[str, str]:
    """Return (particle, base) for a separable PV infinitive, e.g. "abbiegen" -> ("ab", "biegen")."""
    for p in PARTICLES:
        if verb.startswith(p) and len(verb) > len(p):
            return p, verb[len(p):]
    raise ValueError(f'no known particle prefix for {verb!r}')


def _parse_block(lines: list[str]) -> dict | None:
    """Parse one sentence block into a row dict, or None if unusable."""
    header = lines[0].split('\t')
    if len(header) < 2 or header[0] not in ('LITERAL', 'NON-LITERAL'):
        return None
    scores = []
    for s in header[1:]:
        try:
            scores.append(float(s))
        except ValueError:
            continue
    if len(scores) < 2:
        return None

    toks: list[str] = []
    base_marked = 0
    particle_idx: list[int] = []
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) < 3:
            continue
        toks.append(fields[1])
        if fields[2] == 'X':
            base_marked += 1
        # separated particles hang off the verb as PTKVZ tokens
        if len(fields) > 5 and fields[5] == 'PTKVZ':
            particle_idx.append(len(toks) - 1)
    if not toks or base_marked < 1:
        return None

    return {
        'scores': scores,
        'context': ' '.join(toks).strip(),
        'base_marked': base_marked,
        'particle_idx': particle_idx,
    }


def main() -> int:
    files = sorted(SRC.glob('*.txt'))
    if not files:
        print(f'ERROR: no pv_litnlit .txt files under {SRC}', file=sys.stderr)
        return 1

    from src.marks import _match_german_pv, normalize

    rows = []
    n_blocks = 0
    n_skip = 0
    n_align = 0
    for path in files:
        verb = path.stem.strip()
        try:
            particle, base = _split_particle_verb(verb)
        except ValueError as e:
            print(f'WARN: {path.name}: {e}; skipping verb', file=sys.stderr)
            continue

        text = path.read_text(encoding='utf-8')
        blocks = re.split(r'\n\s*\n', text)
        for blk in blocks:
            lines = [l.rstrip() for l in blk.splitlines() if l.strip()]
            if not lines:
                continue
            n_blocks += 1
            parsed = _parse_block(lines)
            if parsed is None:
                n_skip += 1
                continue

            ctx = parsed['context']
            # Alignment sanity: marks._match_german_pv must find the PV span.
            pair = _match_german_pv(normalize(ctx), base, particle)
            if pair is None:
                n_skip += 1
                continue
            n_align += 1

            scores = parsed['scores']
            avg = 5.0 - float(np.mean(scores))
            std = float(np.std(scores)) if len(scores) > 1 else 0.0
            rows.append({
                'ContextID': f'LITNLT-{len(rows):05d}',
                'ParticleVerb': verb,
                'Base': base,
                'Particle': particle,
                'Avg': f'{avg:.4f}',
                'Std': f'{std:.4f}',
                'Context': ctx,
            })

    out = pd.DataFrame(rows, columns=[
        'ContextID', 'ParticleVerb', 'Base', 'Particle', 'Avg', 'Std', 'Context'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep='\t', index=False)

    print(f'blocks parsed: {n_blocks}, kept: {len(rows)}, skipped: {n_skip}')
    print(f'PV span alignment (fused+separated): {n_align} ({100.0 * n_align / n_blocks:.1f}%)')
    print(f'wrote {len(rows)} rows -> {OUT}')
    print('verbs:', out['ParticleVerb'].nunique())
    print('particles:', dict(out['Particle'].value_counts()))
    print('avg range: %.2f..%.2f' % (float(out['Avg'].min()), float(out['Avg'].max())))
    return 0


if __name__ == '__main__':
    sys.exit(main())