#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compositionality Data Augmentation via LLM Paraphrase (noun compounds / phrasal verbs).

Given low-compositionality gold rows (ModAvg/HeadAvg/Avg <= --max-score), asks the
LLM for ``--n-variants`` natural paraphrases that KEEP the target construction
(Compound = Mod+Head, or ParticleVerb = Base+Particle) verbatim and preserve the
same (non-compositional / idiomatic) sense, so the gold score + std transfer.

Every generated variant is validated against the SAME char-matching logic used by
``src.marks.find_spans`` (via ``src.marks._match_*``), so every row emitted into the
TSV is guaranteed to yield a valid mod/head alignment in the training loader.

Outputs:
  - {out_dir}/comp_outputs.json                      : Full metadata & audit trace (per row)
  - {tsv_dir}/SynthNN_{en|de}_training.tsv           : Gold-schema NN rows
    (ContextID | Compound | Mod | Head | ModAvg | ModStd | HeadAvg | HeadStd | Context)
  - {tsv_dir}/SynthPV_{en|de}_training.tsv           : Gold-schema PV rows
    (ContextID | ParticleVerb | Base | Particle | Avg | Std | Context)

Scores/stds are inherited verbatim from the gold source row.

Usage:
  # 1. Batch Mode (5 gold rows per API call, 3 paraphrases each -> 15 synthetic rows):
  python gen/llm_synth_comp_augment.py --api gemini --limit 100 --batch-size 5 --n-variants 3

  # 2. Free Tier Mode (rate-limit):
  python gen/llm_synth_comp_augment.py --api gemini --limit 100 --batch-size 5 --rpm 14 --max-concurrency 1

  # 3. German phrasal verbs only, stricter score cutoff:
  python gen/llm_synth_comp_augment.py --lang de --input dataset/de-pv-train.tsv --max-score 2.5

  # 4. Resume interrupted generation:
  python gen/llm_synth_comp_augment.py --resume

  # 5. Inspect candidates without calling the API:
  python gen/llm_synth_comp_augment.py --dry-run
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow ``from src.marks import ...`` regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def auto_load_dotenv():
    """Automatically loads .env from current or parent directories without external dependencies."""
    search_dirs = [os.getcwd(), _HERE, _REPO]
    for d in search_dirs:
        env_path = os.path.join(d, '.env')
        if os.path.isfile(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, val = line.split('=', 1)
                        key = key.strip()
                        val = val.strip().strip("'").strip('"')
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass
            break


auto_load_dotenv()

# Same char-matching primitives the training loader uses for span alignment.
try:
    from src.marks import (
        _match_from_compound, _match_fused, _match_german_pv, _match_independent,
        _match_spaced, _GERMAN_PARTICLES, normalize,
    )
except Exception:  # pragma: no cover - CLI will explain if the env is broken
    raise SystemExit(
        'Cannot import src.marks — run this script from the repo root '
        '(C:\\CODE_SOMETHING\\HiHope) so the src package is importable.'
    )

# -----------------------------------------------------------------------------
# CONSTANTS & SCHEMAS
# -----------------------------------------------------------------------------

HEADER_NN = ['ContextID', 'Compound', 'Mod', 'Head', 'ModAvg', 'ModStd', 'HeadAvg', 'HeadStd', 'Context']
HEADER_PV = ['ContextID', 'ParticleVerb', 'Base', 'Particle', 'Avg', 'Std', 'Context']

DEFAULT_INPUT_GLOBS = ['dataset/*-train.tsv']

REFUSAL_PATTERNS = [
    r"^i cannot (?:fulfill|generate|create|comply|assist)",
    r"^as an ai\b",
    r"^i am an ai\b",
    r"\bagainst my (?:safety|content|usage) (?:guidelines|policies)\b",
    r"^i am unable to (?:generate|create|comply|assist)",
    r"cannot generate",
    r"^i apologize, but i cannot",
]

LANGUAGE_INSTRUCTIONS = {
    'EN': ("English. Natural, neutral written English — formal or encyclopedic prose "
           "typical of dictionary example sentences. Keep roughly the same length, "
           "tone and register as the source sentence. For phrasal verbs, natural tense inflections "
           "(e.g. 'gave up', 'gives up', 'giving up') are encouraged where grammatically fitting."),
    'DE': ("German. Natural, neutral written German — formal or encyclopedic prose "
           "typical of dictionary example sentences. For German separable particle verbs (trennbare Verben, "
           "e.g. 'abhauen', 'aufgeben'), you are fully permitted and encouraged to conjugate and separate "
           "the verb in main clauses (e.g. 'haute ... ab', 'gibt ... auf') or use natural participial forms."),
}

SINGLE_SYSTEM_PROMPT = """You are an expert computational linguist assisting in corpus annotation for a compositionality research dataset.
Your task is to paraphrase a source sentence that contains a TARGET CONSTRUCTION used in a NON-COMPOSITIONAL / IDIOMATIC sense, into {n_variants} DISTINCT natural paraphrases.

The dataset scores how compositional each construction is on a 0-5 scale. The source sentence has a LOW score: the construction is non-compositional (its meaning is not the sum of its parts). Each paraphrase must therefore keep the construction in the SAME idiomatic sense, so the low score stays valid.

STRICT REQUIREMENTS:
1. CONSTRUCTION PRESERVATION: Include the target construction VERBATIM (or naturally inflected/separated for verbs):
   {construction_desc}
2. SENSE PRESERVATION: Keep the EXACT same reading / meaning as the source sentence — nothing is re-scored, nothing becomes more literal or more idiomatic.
3. LINGUISTIC DIVERSITY ACROSS VARIANTS: Vary word order, clause structure, synonymy of NON-construction words, and phrasing so each variant is genuinely distinct.
4. REGISTER: Match the source sentence's length and prose style ({lang_instruction}).
5. Do NOT explain, define, gloss, or comment on the construction.

OUTPUT FORMAT:
You MUST reply with ONLY a raw JSON object (no markdown, no backticks, no explanations):
{{
  "variants": [
    "Paraphrase 1",
    "Paraphrase 2"
  ]
}}"""

BATCH_SYSTEM_PROMPT = """You are an expert computational linguist assisting in corpus annotation for a compositionality research dataset.
Your task is to paraphrase a BATCH of {batch_size} source sentences, each containing a TARGET CONSTRUCTION used in a NON-COMPOSITIONAL / IDIOMATIC sense. For EACH source sentence, generate {n_variants} DISTINCT natural paraphrases.

The dataset scores how compositional each construction is on a 0-5 scale. Each source sentence has a LOW score: the construction is non-compositional. Each paraphrase must keep the construction in the SAME idiomatic sense.

STRICT REQUIREMENTS PER ITEM:
1. CONSTRUCTION PRESERVATION: Include every item's "construction" verbatim (or naturally inflected/separated for verbs) in ALL of its paraphrases.
2. SENSE PRESERVATION: Keep the exact same reading / meaning as the source sentence.
3. LINGUISTIC DIVERSITY: Vary word order, clause structure, subject/topics, and phrasing across the variants.
4. REGISTER: Match the source sentence's length and prose style ({lang_instruction}).
5. Do NOT explain, define, gloss, or comment on any construction.

OUTPUT FORMAT:
You MUST reply with ONLY a raw JSON object (no markdown, no backticks, no explanations):
{{
  "results": [
    {{
      "id": "source_id_here",
      "variants": ["paraphrase 1", "paraphrase 2", "paraphrase 3"]
    }}
  ]
}}"""


class RetryableAPIError(Exception):
    def __init__(self, wait_hint: Optional[float] = None):
        super().__init__('Retryable API Error')
        self.wait_hint = wait_hint


class FatalAPIError(Exception):
    pass


# -----------------------------------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api', choices=['gemini', 'groq'], default='gemini',
                        help='LLM API provider: gemini (Google GenAI) or groq (OpenAI-compatible SDK). Default: gemini')
    parser.add_argument('--model', default=None,
                        help='Model name. Default: gemini-3.5-flash-lite (gemini) or openai/gpt-oss-120b (groq)')
    parser.add_argument('--input', action='append', default=None,
                        help='File or glob pattern of gold dataset files (e.g. dataset/en-nn-train.tsv)')
    parser.add_argument('--lang', choices=['en', 'de', 'both'], default=None,
                        help="Restrict to 'en', 'de' (default: auto by filename)")
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='Only source rows whose score is >= this. Default: 0.0')
    parser.add_argument('--max-score', type=float, default=3.0,
                        help='Only source rows whose score (min ModAvg/HeadAvg for NN, Avg for PV) is <= this. Default: 3.0')
    parser.add_argument('--max-sim', type=float, default=0.85,
                        help='Maximum allowed lexical Jaccard similarity between generated variants (default: 0.85 to filter near-duplicates)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of low-score source rows to process')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of source sentences to bundle per single API call (default: 1; recommended 5 for free tier)')
    parser.add_argument('--n-variants', type=int, default=3,
                        help='Number of distinct paraphrases to generate per source sentence (default: 3)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Pause/cooldown in seconds between requests (e.g. --delay 4.0 for Gemini Free Tier 15 RPM)')
    parser.add_argument('--rpm', type=int, default=None,
                        help='Target maximum Requests Per Minute rate limit (e.g. --rpm 14 for Free Tier)')
    parser.add_argument('--max-concurrency', type=int, default=None,
                        help='Max parallel worker threads (default: 8 for gemini, 4 for groq; use 1-2 if on Free Tier)')
    parser.add_argument('--max-retries', type=int, default=10,
                        help='Max retries per sample upon rate limit')
    parser.add_argument('--max-chars', type=int, default=1200,
                        help='Truncate over-length source contexts')
    parser.add_argument('--temp', type=float, default=0.75,
                        help='Sampling temperature')
    parser.add_argument('--out', default='gen/comp_outputs.json',
                        help='Output JSON metadata path')
    parser.add_argument('--tsv-dir', default='gen/synth',
                        help='Output directory for SynthNN_*/SynthPV_* TSV files')
    parser.add_argument('--resume', action='store_true',
                        help='Skip rows already present in output JSON')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only inspect and print source statistics without making API calls')
    return parser.parse_args()


ARGS = parse_args()


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_existing(out_path: str) -> Dict[str, Dict[str, Any]]:
    if not ARGS.resume or not os.path.exists(out_path):
        return {}
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('items', [])
        loaded: Dict[str, Dict[str, Any]] = {}
        for it in items:
            key = it.get('source_id') or it.get('id')
            if key and (it.get('variants') or it.get('rewritten')):
                loaded[key] = it
        return loaded
    except Exception as e:
        print(f"  [Warning] Could not load resume cache from {out_path}: {e}")
        return {}


def find_input_paths() -> List[str]:
    patterns = ARGS.input or DEFAULT_INPUT_GLOBS
    found: List[str] = []
    for p in patterns:
        if os.path.isdir(p):
            found += glob.glob(os.path.join(p, '*-train.tsv'))
        else:
            found += glob.glob(p, recursive=True)
    return sorted(set(found))


def _lan(header: List[str]) -> str:
    return 'pv' if 'ParticleVerb' in header and 'Base' in header and 'Particle' in header else 'nn'


def _f(s: str) -> Optional[float]:
    try:
        x = float(s)
        return x if x == x else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def _f6(x: Optional[float]) -> str:
    return f'{x:.6f}' if x is not None else ''


def read_source_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        m = re.search(r'[\\/](en|de)-(.*?)\.tsv$', p, re.IGNORECASE)
        lang = 'de' if m and m.group(1).lower() == 'de' else 'en'
        if ARGS.lang and ARGS.lang != 'both' and lang != ARGS.lang:
            continue

        try:
            with open(p, 'r', encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader, None)
                if not header:
                    continue
                idx = {name: i for i, name in enumerate(header)}
                kind = _lan(header)

                if kind == 'nn':
                    need = ['ContextID', 'Compound', 'Mod', 'Head', 'ModAvg', 'HeadAvg', 'Context']
                    if not all(k in idx for k in need):
                        continue
                    a = 'ModAvg'
                    b = 'HeadAvg'
                else:
                    need = ['ContextID', 'ParticleVerb', 'Base', 'Particle', 'Avg', 'Context']
                    if not all(k in idx for k in need):
                        continue
                    a = 'Avg'
                    b = 'Avg'

                for line in reader:
                    if len(line) <= max(idx.values()):
                        continue
                    cid = line[idx['ContextID']].strip() or f'{lang}-{len(rows)}'
                    ctx = line[idx['Context']].strip()
                    if not ctx:
                        continue

                    score_a = _f(line[idx[a]])
                    score_b = _f(line[idx[b]])
                    finite = [s for s in (score_a, score_b) if s is not None]
                    if not finite:
                        continue
                    base_score = min(finite)

                    if base_score > ARGS.max_score or base_score < ARGS.min_score:
                        continue

                    def col(name: str) -> str:
                        return line[idx[name]].strip() if name in idx else ''

                    rows.append({
                        'id': f'{lang}-{cid}',
                        'lang': lang,
                        'kind': kind,
                        'context': ctx[:ARGS.max_chars],
                        'compound': col('Compound') if kind == 'nn' else col('ParticleVerb'),
                        'mod': col('Mod') if kind == 'nn' else col('Base'),
                        'head': col('Head') if kind == 'nn' else col('Particle'),
                        'mod_avg': score_a,
                        'head_avg': score_b,
                        'mod_std': _f(col('ModStd')) if kind == 'nn' and 'ModStd' in idx else _f(col('Std')),
                        'head_std': _f(col('HeadStd')) if kind == 'nn' and 'HeadStd' in idx else _f(col('Std')),
                        'score': base_score,
                    })
                    # The source must itself be alignable: otherwise the gold row
                    # yields no spans in the loader, and generated paraphrases of it
                    # would inherit unusable supervision. (_aligns is defined below;
                    # globals are resolved at call time.)
                    if not _aligns(rows[-1]['context'], rows[-1]):
                        rows.pop()
        except Exception as e:
            print(f"  [Error reading {p}]: {e}", file=sys.stderr)
    return rows


# -----------------------------------------------------------------------------
# VALIDATION: only emit variants that the training aligner can align
# -----------------------------------------------------------------------------

def _construction_desc(row: Dict[str, Any]) -> str:
    if row['kind'] == 'nn':
        return (f'modifier + head compound: "{{{row["compound"]}}}" '
                f'(modifier "{row["mod"]}", head "{row["head"]}")')
    return (f'phrasal verb: "{{{row["compound"]}}}" '
            f'(base verb "{row["mod"]}", particle "{row["head"]}")')


def _aligns(text: str, row: Dict[str, Any]) -> bool:
    """True iff the trained aligner (src.marks.find_spans char logic) can align."""
    text_n = normalize(text)
    compound, mod, head = row['compound'], row['mod'], row['head']

    if row['kind'] == 'pv' and (
            normalize(head) in _GERMAN_PARTICLES
            or bool(compound and normalize(compound).startswith(normalize(head)))):
        pair = _match_german_pv(text_n, mod, head)
    else:
        pair = (
            _match_from_compound(text_n, compound, mod, head)
            or _match_fused(text_n, mod, head)
            or _match_spaced(text_n, mod, head)
            or _match_independent(text_n, mod, head)
        )
    return pair is not None


def _clean_variant(raw: str, row: Dict[str, Any]) -> Optional[str]:
    v = re.sub(r'[\t\r\n]+', ' ', raw).strip().strip('"').strip("'")
    if len(v) < 5:
        return None
    if normalize(v) == normalize(row['context']):
        return None
    if any(re.search(p, v.lower()) for p in REFUSAL_PATTERNS):
        return None
    if not _aligns(v, row):
        return None
    return v


def _word_jaccard(s1: str, s2: str) -> float:
    """Computes lexical token Jaccard similarity between two sentences."""
    w1 = set(re.findall(r'\w+', s1.lower()))
    w2 = set(re.findall(r'\w+', s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def clean_and_validate_single_variants(raw_text: str, row: Dict[str, Any]) -> List[str]:
    if not raw_text:
        return []

    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    candidate_list: List[str] = []
    if cleaned.startswith('{'):
        try:
            parsed = json.loads(cleaned)
            v = (parsed.get('variants') or parsed.get('paraphrases')
                 or parsed.get('rewritten'))
            if isinstance(v, list):
                candidate_list = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str) and v.strip():
                candidate_list = [v.strip()]
        except json.JSONDecodeError:
            pass
    elif cleaned.startswith('['):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                candidate_list = [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if not candidate_list:
        for l in cleaned.split('\n'):
            stripped = re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', l).strip()
            if len(stripped) > 10 and not stripped.startswith('{') and not stripped.endswith('}'):
                candidate_list.append(stripped)

    valid: List[str] = []
    for item in candidate_list:
        v = _clean_variant(item, row)
        if not v:
            continue
        # Lexical diversity check: reject near-identical variants
        if any(_word_jaccard(v, existing) > ARGS.max_sim for existing in valid):
            continue
        valid.append(v)
    return valid


def clean_and_validate_batch_results(raw_text: str, batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not raw_text:
        return {}
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    results_map: Dict[str, List[str]] = {}
    row_map = {r['id']: r for r in batch}

    try:
        parsed = json.loads(cleaned)
        items_list: List[Any] = []
        if isinstance(parsed, dict):
            items_list = parsed.get('results') or parsed.get('items') or parsed.get('data') or []
            if not items_list and any(k in row_map for k in parsed):
                for k, v in parsed.items():
                    if isinstance(v, list):
                        items_list.append({'id': k, 'variants': v})
        elif isinstance(parsed, list):
            items_list = parsed

        for entry in items_list:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get('id', '')).strip()
            v_list = entry.get('variants') or entry.get('paraphrases') or []
            if item_id in row_map and isinstance(v_list, list):
                row = row_map[item_id]
                clean_v: List[str] = []
                for v in v_list:
                    vv = _clean_variant(re.sub(r'[\t\r\n]+', ' ', str(v)), row)
                    if not vv:
                        continue
                    if any(_word_jaccard(vv, existing) > ARGS.max_sim for existing in clean_v):
                        continue
                    clean_v.append(vv)
                if clean_v:
                    results_map[item_id] = clean_v
    except Exception:
        pass

    return results_map


# -----------------------------------------------------------------------------
# CLIENT & MODEL INITIALIZATION
# -----------------------------------------------------------------------------

def init_client() -> Tuple[str, Any]:
    if ARGS.api == 'gemini':
        try:
            from google import genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise SystemExit('Missing GEMINI_API_KEY environment variable. Add it to .env or run: export GEMINI_API_KEY="..."')
            return ('gemini', genai.Client(api_key=key))
        except ImportError:
            raise SystemExit("Please install google-genai: pip install google-genai")

    elif ARGS.api == 'groq':
        try:
            from openai import OpenAI
            key = os.environ.get('GROQ_API_KEY')
            if not key:
                raise SystemExit('Missing GROQ_API_KEY environment variable. Add it to .env or run: export GROQ_API_KEY="..."')
            return ('groq', OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1'))
        except ImportError:
            raise SystemExit("Please install openai: pip install openai")

    raise ValueError(f"Unknown API provider: {ARGS.api}")


_CTX: Dict[str, Any] = {'client': None, 'model': None}


def get_default_model(api_kind: str) -> str:
    if ARGS.model:
        return ARGS.model
    if api_kind == 'gemini':
        return 'gemini-3.5-flash-lite'
    return 'openai/gpt-oss-120b'


def extract_retry_after(err: Any) -> Optional[float]:
    headers = (getattr(getattr(err, 'response', None), 'headers', None)
               or getattr(getattr(err, 'http_response', None), 'headers', None))
    if headers:
        ra = headers.get('retry-after') or headers.get('Retry-After')
        if ra:
            try:
                return max(0.5, float(ra))
            except (ValueError, TypeError):
                pass
    return None


# -----------------------------------------------------------------------------
# API CALL & PARSING
# -----------------------------------------------------------------------------

def build_prompt_payload(batch: List[Dict[str, Any]]) -> Tuple[str, str]:
    lang = batch[0]['lang']
    lang_inst = LANGUAGE_INSTRUCTIONS.get(lang.upper(), 'Match the source sentence exactly.')

    if len(batch) == 1:
        row = batch[0]
        system = SINGLE_SYSTEM_PROMPT.format(
            n_variants=ARGS.n_variants,
            construction_desc=_construction_desc(row),
            lang_instruction=lang_inst,
        )
        user = (
            f"Source sentence (low compositionality, score {row['score']:.2f}):\n"
            f"\"{row['context']}\"\n\n"
            f"Generate {ARGS.n_variants} distinct paraphrases that keep "
            f"\"{row['compound']}\" verbatim and preserve the exact same meaning:"
        )
        return system, user

    system = BATCH_SYSTEM_PROMPT.format(
        lang_instruction=lang_inst,
        batch_size=len(batch),
        n_variants=ARGS.n_variants,
    )
    items_json = [
        {
            "id": r['id'],
            "sentence": r['context'],
            "construction": r['compound'],
            "modifier": r['mod'],
            "head": r['head'] if r['kind'] == 'nn' else 'particle: ' + r['head'],
            "score": round(r['score'], 2),
        }
        for r in batch
    ]
    user = (
        f"Input Batch ({len(batch)} source sentences):\n"
        f"{json.dumps(items_json, ensure_ascii=False, indent=2)}\n\n"
        f"Generate {ARGS.n_variants} distinct paraphrases for each item "
        f"(keep the 'construction' verbatim, preserve meaning):"
    )
    return system, user


def call_llm(system_prompt: str, user_prompt: str, token_budget: int = 2000) -> str:
    kind, cli = _CTX['client']
    model_name = _CTX['model']

    if kind == 'gemini':
        from google.genai import types
        from google.genai import errors as genai_errors

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=ARGS.temp,
                max_output_tokens=token_budget,
                response_mime_type="application/json",
            )
            response = cli.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            return response.text or ""
        except genai_errors.APIError as e:
            code = getattr(e, 'code', None)
            wait = extract_retry_after(e) or 2.0
            if code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(wait_hint=wait) from e
            raise FatalAPIError(f"Gemini API Error (code {code}): {e}") from e
        except Exception as e:
            wait = extract_retry_after(e) or 3.0
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str:
                raise RetryableAPIError(wait_hint=wait) from e
            raise

    elif kind == 'groq':
        from openai import APIConnectionError, APIStatusError, RateLimitError
        try:
            response = cli.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=ARGS.temp,
                max_tokens=token_budget,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError) as e:
            wait = extract_retry_after(e) or 2.0
            raise RetryableAPIError(wait_hint=wait) from e
        except APIStatusError as e:
            wait = extract_retry_after(e) or 2.0
            if e.status_code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(wait_hint=wait) from e
            raise FatalAPIError(f"Groq HTTP Error {e.status_code}: {e}") from e


def process_batch(batch: List[Dict[str, Any]]) -> Dict[str, Tuple[List[str], str]]:
    """Processes a batch of 1 or more source sentences with retries."""
    out_dict: Dict[str, Tuple[List[str], str]] = {}
    try:
        system_prompt, user_prompt = build_prompt_payload(batch)
    except Exception as e:
        print(f"  [Error building batch prompt]: {e}", file=sys.stderr)
        for r in batch:
            out_dict[r['id']] = ([], 'fatal')
        return out_dict

    token_budget = min(4096, max(2000, len(batch) * ARGS.n_variants * 220))

    for attempt in range(1, ARGS.max_retries + 1):
        try:
            raw_out = call_llm(system_prompt, user_prompt, token_budget=token_budget)
            if len(batch) == 1:
                r_id = batch[0]['id']
                v_list = clean_and_validate_single_variants(raw_out, batch[0])
                out_dict[r_id] = (v_list, 'ok' if v_list else 'empty_or_invalid')
                return out_dict
            else:
                batch_res = clean_and_validate_batch_results(raw_out, batch)
                all_ok = True
                for r in batch:
                    r_id = r['id']
                    if r_id in batch_res and batch_res[r_id]:
                        out_dict[r_id] = (batch_res[r_id], 'ok')
                    else:
                        out_dict[r_id] = ([], 'empty_or_invalid')
                        all_ok = False
                if all_ok or attempt >= ARGS.max_retries:
                    return out_dict
        except RetryableAPIError as e:
            if attempt >= ARGS.max_retries:
                break
            wait_time = e.wait_hint or min(45.0, (1.8 ** attempt)) + random.uniform(0.1, 0.5)
            time.sleep(wait_time)
        except FatalAPIError as e:
            print(f"  [FATAL API Error]: {e}", file=sys.stderr)
            for r in batch:
                out_dict[r['id']] = ([], 'fatal')
            return out_dict
        except Exception as e:
            if attempt >= ARGS.max_retries:
                print(f"  [Error] Batch failed after {attempt} retries ({type(e).__name__}: {e})", file=sys.stderr)
                break
            time.sleep(1.5 * attempt)

    for r in batch:
        if r['id'] not in out_dict:
            out_dict[r['id']] = ([], 'gave_up')
    return out_dict


# -----------------------------------------------------------------------------
# IO & TSV EXPORT (gold schema, per lang/type)
# -----------------------------------------------------------------------------

def atomic_save_json(path: str, data: Any):
    dir_name = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_canonical_tsv(lang: str, kind: str, items: List[Dict[str, Any]], out_tsv_path: str):
    """Writes gold-schema TSV rows (scores/std inherited from the source row)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_tsv_path)) or '.', exist_ok=True)
    header = HEADER_NN if kind == 'nn' else HEADER_PV
    with open(out_tsv_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(header) + '\n')
        for item in sorted(items, key=lambda x: x.get('source_id') or x.get('id', '')):
            variants = item.get('variants', [])
            if not variants and item.get('rewritten'):
                variants = [item['rewritten']]
            src = item.get('source', {})
            if kind == 'nn':
                for v_idx, text in enumerate(variants, start=1):
                    row_fields = [
                        f"XSYNTH{src.get('id', item.get('source_id', ''))}V{v_idx}",
                        src.get('compound', ''),
                        src.get('mod', ''),
                        src.get('head', ''),
                        _f6(src.get('mod_avg')),
                        _f6(src.get('mod_std')),
                        _f6(src.get('head_avg')),
                        _f6(src.get('head_std')),
                        re.sub(r'[\t\r\n]+', ' ', text).strip(),
                    ]
                    f.write('\t'.join(row_fields) + '\n')
            else:
                for v_idx, text in enumerate(variants, start=1):
                    row_fields = [
                        f"XSYNTH{src.get('id', item.get('source_id', ''))}V{v_idx}",
                        src.get('compound', ''),
                        src.get('mod', ''),
                        src.get('head', ''),
                        _f6(src.get('mod_avg')),
                        _f6(src.get('mod_std')),
                        re.sub(r'[\t\r\n]+', ' ', text).strip(),
                    ]
                    f.write('\t'.join(row_fields) + '\n')


# -----------------------------------------------------------------------------
# MAIN LIFECYCLE
# -----------------------------------------------------------------------------

def main():
    print("=" * 64)
    print(" Compositionality Augment: LLM Paraphrase of Low-Score Rows")
    print("=" * 64)

    paths = find_input_paths()
    if not paths:
        sys.exit(f"No gold dataset found matching patterns: {ARGS.input or DEFAULT_INPUT_GLOBS}")

    print(f"Found input files: {paths}")
    rows = read_source_rows(paths)
    if not rows:
        sys.exit("No low-score labeled rows found under the current filters.")

    from collections import Counter
    counter = Counter((r['lang'], r['kind']) for r in rows)

    if ARGS.dry_run:
        print("\n[DRY RUN] Low-score rows available as paraphrase sources:")
        for (lang, kind), count in sorted(counter.items()):
            est_samples = count * ARGS.n_variants
            est_calls = (count + ARGS.batch_size - 1) // ARGS.batch_size
            print(f"  - {lang} {kind}: {count} rows -> ~{est_samples} synthetic rows "
                  f"({ARGS.n_variants}/row, ~{est_calls} API calls at batch-size={ARGS.batch_size})")
        total = len(rows)
        print(f"Total low-score rows: {total} -> ~{total * ARGS.n_variants} expected synthetic rows "
              f"(after alignment validation)")
        return

    existing_items = load_existing(ARGS.out) if ARGS.resume else {}
    done_ids: Set[str] = set(existing_items.keys())
    todo_rows = [r for r in rows if r['id'] not in done_ids]
    # Keep batches homogeneous per (lang, kind).
    todo_rows.sort(key=lambda r: (r['lang'], r['kind']))

    if ARGS.limit:
        todo_rows = todo_rows[:ARGS.limit]

    print(f"Low-score sources: {len(rows)} | Already Done: {len(done_ids)} | To Process: {len(todo_rows)}")
    print(f"Batch Size: {ARGS.batch_size} | Variants per row: {ARGS.n_variants} | Max Score: {ARGS.max_score}")

    if not todo_rows:
        print("All matching rows are already processed. Exiting.")
        return

    kind, cli = init_client()
    _CTX['client'] = (kind, cli)
    _CTX['model'] = get_default_model(kind)
    concurrency = ARGS.max_concurrency or (8 if kind == 'gemini' else 4)

    effective_delay = ARGS.delay
    if ARGS.rpm and ARGS.rpm > 0:
        effective_delay = max(effective_delay, 60.0 / ARGS.rpm)
    if effective_delay > 0:
        print(f"Rate Limiter Active: {effective_delay:.2f}s cooldown (~{60.0/effective_delay:.1f} RPM)")

    print(f"API Provider: {kind} | Model: {_CTX['model']} | Parallel Workers: {concurrency}")

    results: Dict[str, Dict[str, Any]] = dict(existing_items)
    stats = {'ok': 0, 'empty_or_invalid': 0, 'fatal': 0, 'gave_up': 0}
    row_lookup = {r['id']: r for r in rows}

    prompt_batches = [todo_rows[i:i + ARGS.batch_size] for i in range(0, len(todo_rows), ARGS.batch_size)]
    print(f"API Requests to make: {len(prompt_batches)}")

    def persist_snapshot():
        total_variants = sum(len(it.get('variants', [])) for it in results.values())
        meta = {
            'generator': 'llm_synth_comp_augment.py',
            'api': ARGS.api,
            'model': _CTX['model'],
            'max_score': ARGS.max_score,
            'batch_size': ARGS.batch_size,
            'n_variants_per_row': ARGS.n_variants,
            'updated_at': now_iso(),
            'source_rows_count': len(results),
            'total_synthetic_rows_count': total_variants,
            'items': [results[k] for k in sorted(results.keys())],
        }
        atomic_save_json(ARGS.out, meta)

        by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for it in results.values():
            by_key.setdefault((it['lang'], it['kind']), []).append(it)
        for (lang, rkind), items in by_key.items():
            prefix = 'SynthNN' if rkind == 'nn' else 'SynthPV'
            tsv_path = os.path.join(ARGS.tsv_dir, f"{prefix}_{lang}_training.tsv")
            write_canonical_tsv(lang, rkind, items, tsv_path)

        return total_variants

    try:
        t0 = time.time()
        processed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(0, len(prompt_batches), concurrency):
                chunk = prompt_batches[i:i + concurrency]
                future_map = {executor.submit(process_batch, b): b for b in chunk}

                for future in as_completed(future_map):
                    batch_res = future.result()
                    processed += 1

                    for r_id, (variants_list, status) in batch_res.items():
                        stats[status] = stats.get(status, 0) + 1
                        if status == 'ok' and variants_list:
                            src = row_lookup[r_id]
                            results[r_id] = {
                                'source_id': r_id,
                                'lang': src['lang'],
                                'kind': src['kind'],
                                'compound': src['compound'],
                                'score': src['score'],
                                'source': {k: src[k] for k in (
                                    'id', 'compound', 'mod', 'head',
                                    'mod_avg', 'head_avg', 'mod_std', 'head_std', 'context',
                                )},
                                'original': src['context'],
                                'variants': variants_list,
                                'variants_count': len(variants_list),
                                'generated_at': now_iso(),
                            }

                    if processed % 10 == 0 or processed == len(prompt_batches):
                        total_saved = persist_snapshot()
                        elapsed = time.time() - t0
                        speed = (processed * ARGS.batch_size) / max(elapsed, 0.001)
                        print(f"  Progress: {processed}/{len(prompt_batches)} requests "
                              f"({speed:.1f} rows/s) | Synthetic Rows: {total_saved} | "
                              f"Ok: {stats['ok']} | Errors: {stats['empty_or_invalid'] + stats['gave_up']}",
                              flush=True)

                if effective_delay > 0 and (i + concurrency) < len(prompt_batches):
                    time.sleep(effective_delay * len(chunk))

    except KeyboardInterrupt:
        print("\n[Ctrl+C detected] Gracefully saving generated items before exit...", flush=True)
    finally:
        total_saved = persist_snapshot()
        print("\n" + "=" * 64)
        print(" Generation Complete!")
        print(f" Source Rows Processed: {len(results)}")
        print(f" Synthetic Rows Generated: {total_saved}")
        print(f" JSON Registry: {ARGS.out}")
        print(f" TSV Output: {ARGS.tsv_dir}/SynthNN_{{en|de}}_training.tsv, SynthPV_{{en|de}}_training.tsv")
        print(f" Stats: {stats}")
        print(" Add outputs to training via config: train_aux='gen/synth/SynthNN_en_training.tsv' (or a glob).")
        print("=" * 64)


if __name__ == '__main__':
    main()