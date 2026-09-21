"""Canonical prototype prompting and WordNet gloss enrichment for Stream 1.

Transforms isolated target words into canonical prototype texts:
1. WordNet Gloss (English, auto-fetched): e.g. "flea: a small wingless jumping insect"
2. Canonical Template Prompting (Multilingual):
   - English Nouns: "The literal definition of the noun '{word}'."
   - German Nouns: "Die wörtliche Bedeutung des Nomens '{word}'."
   - English Particle Verbs: "The physical action to {word}."
   - German Particle Verbs: "Die wörtliche Handlung, {word}."
3. Raw Fallback: "{word}"
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_WN = None
_WN_TRIED = False


def _get_wordnet():
    global _WN, _WN_TRIED
    if not _WN_TRIED:
        _WN_TRIED = True
        try:
            from nltk.corpus import wordnet as wn
            # Test if synsets work without downloading errors
            _ = wn.synsets('dog')
            _WN = wn
            logger.info("WordNet initialized successfully for prototype glosses")
        except Exception:
            _WN = None
            logger.debug("WordNet not available; using canonical prompt templates")
    return _WN


def get_canonical_template(word: str, lang: str = "en", is_pv: bool = False,
                           is_verb: bool = False, is_particle: bool = False) -> str:
    """Return a canonical prompt template for the target word."""
    w = (word or "").strip()
    if not w:
        return ""
    if is_pv:
        if lang == "de":
            return f"Die wörtliche Handlung, {w}."
        return f"The physical action to {w}."
    elif is_verb:
        if lang == "de":
            return f"Das Verb '{w}'."
        return f"The verb '{w}'."
    elif is_particle:
        if lang == "de":
            return f"Die Partikel '{w}'."
        return f"The particle '{w}'."
    else:
        if lang == "de":
            return f"Die wörtliche Bedeutung des Nomens '{w}'."
        return f"The literal definition of the noun '{w}'."


def get_wordnet_gloss(word: str) -> Optional[str]:
    """Auto-fetch primary literal definition from WordNet for English single-token nouns."""
    w = (word or "").strip()
    # Skip empty or multiword phrases (e.g. 'pull up' or 'break up') where synset[0] is often figurative
    if not w or ' ' in w:
        return None
    wn = _get_wordnet()
    if wn is None:
        return None
    try:
        query = w.replace(' ', '_').lower()
        synsets = wn.synsets(query, pos='n')  # Restrict to noun synsets
        if not synsets:
            synsets = wn.synsets(query)
        if synsets:
            primary_def = synsets[0].definition()
            return f"{w}: {primary_def}"
    except Exception:
        pass
    return None


def format_prototype_text(
    word: str,
    lang: str = "en",
    is_pv: bool = False,
    is_verb: bool = False,
    is_particle: bool = False,
    mode: str = "template",
) -> str:
    """Format a target word into an enriched prototype string.

    Args:
        word: Target word or compound (e.g. 'flea', 'market', 'pull up', 'Handschuh').
        lang: Language code ('en' or 'de').
        is_pv: True if particle verb, False for noun compound.
        is_verb: True if single base verb constituent.
        is_particle: True if single particle constituent.
        mode: Formatting mode:
              - 'template': Strict canonical template prompting (deterministic, fast, zero dependency).
              - 'hybrid' / 'auto': WordNet gloss for single English nouns; canonical template otherwise.
              - 'wordnet': WordNet only; falls back to bare word if unavailable.
              - 'raw': Returns the bare word.

    Returns:
        Formatted prototype string ready for tokenization.
    """
    w = (word or "").strip()
    if not w or mode == "raw":
        return w

    # Particle verbs and PV constituents always bypass glosses and use canonical action templates
    if not is_pv and not is_verb and not is_particle and mode in ("hybrid", "auto", "wordnet") and lang == "en":
        gloss = get_wordnet_gloss(w)
        if gloss is not None:
            return gloss
        if mode == "wordnet":
            return w

    # Canonical template prompting (Technique 1)
    return get_canonical_template(w, lang=lang, is_pv=is_pv, is_verb=is_verb, is_particle=is_particle)
