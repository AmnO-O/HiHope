"""Semantic prompt routing and template formulation for Cloze-based Compositionality Detection.

This module formats samples into prompt sequences designed for Masked Language Modeling:
    - Real Noun Compounds (en-nn, de-nn, nctti_en, en-comp, en-ijcnlp)
    - Particle Verbs (en-pv, de-pv)
    - Bare Lemmas (de-litnlit)

It ensures grammatically natural carrier sentences, accurate span tracking,
and provides verbalizer vocabulary tokens for English and German.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union


# Standard verbalizer tokens representing high (literal) vs low (figurative) compositionality
VERBALIZER_TOKENS = {
    "en": {
        "literal": ["literal", "real", "physical"],
        "figurative": ["metaphorical", "figurative", "symbolic"],
    },
    "de": {
        "literal": ["wörtlich", "wörtlichen", "real"],
        "figurative": ["übertragen", "übertragenen", "metaphorisch"],
    },
}


def classify_semantic_type(
    word: str,
    compound: str,
    is_pv: bool = False,
    filename: Optional[str] = None,
) -> str:
    """Deterministically categorize a sample into one of three linguistic types:
    
    1. 'bare_lemma': A single standalone lemma without a distinct parent compound (e.g. de-litnlit "abbiegen").
    2. 'compound': A real multi-word compound where a constituent belongs to a parent compound.
    3. 'particle_verb': A multi-word phrasal verb construction (e.g. "take off", "aufgeben").
    """
    fname = (filename or "").lower()
    w_clean = (word or "").strip()
    c_clean = (compound or "").strip()

    # Explicit file overrides
    if "litnlit" in fname:
        return "bare_lemma"
    if "nctti" in fname or "ijcnlp" in fname or "comp" in fname:
        return "compound"
    if "pv" in fname:
        return "particle_verb"

    # Lexical structure checks
    if not c_clean or c_clean == w_clean:
        return "bare_lemma"

    if is_pv:
        return "particle_verb"

    return "compound"


def build_cloze_prompt(
    sentence: str,
    word: str,
    compound: Optional[str] = None,
    lang: str = "en",
    is_pv: bool = False,
    filename: Optional[str] = None,
    mask_token: str = "[MASK]",
) -> Tuple[str, str]:
    """Construct a grammatically natural cloze prompt for the target.
    
    Returns:
        (prompt_text, semantic_type)
    """
    w = (word or "").strip()
    c = (compound or "").strip()
    s = (sentence or "").strip()
    l = "de" if str(lang).lower().startswith("de") else "en"
    
    sem_type = classify_semantic_type(w, c, is_pv=is_pv, filename=filename)

    if l == "de":
        if sem_type == "bare_lemma":
            # Avoid vacuous "abbiegen in abbiegen"
            prompt = f'Ziel: "{w}". Satz: "{s}". In diesem Satz wird das Wort "{w}" im {mask_token} Sinne verwendet.'
        elif sem_type == "particle_verb":
            target = c if c else w
            prompt = f'Ziel: "{target}". Satz: "{s}". In diesem Satz ist der Ausdruck "{target}" {mask_token}.'
        else:  # compound
            prompt = f'Ziel: "{w}" in "{c}". Satz: "{s}". In diesem Satz ist das Wort "{w}" {mask_token}.'
    else:  # en
        if sem_type == "bare_lemma":
            prompt = f'Target: "{w}". Sentence: "{s}". In this sentence, the word "{w}" is used in a {mask_token} sense.'
        elif sem_type == "particle_verb":
            target = c if c else w
            prompt = f'Target: "{target}". Sentence: "{s}". In this sentence, the expression "{target}" is {mask_token}.'
        else:  # compound
            prompt = f'Target: "{w}" in "{c}". Sentence: "{s}". In this sentence, the word "{w}" is {mask_token}.'

    return prompt, sem_type
