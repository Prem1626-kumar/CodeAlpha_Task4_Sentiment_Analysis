"""
src/preprocessor.py
Text cleaning and normalisation pipeline.

Design decisions
----------------
* Apostrophes are kept so negations like "don't" / "isn't" survive.
* Negation stopwords (not, no, never, nor, neither) are kept even when
  stop-word removal is enabled, because they invert sentiment polarity.
* Lemmatisation is optional (requires NLTK wordnet / averaged_perceptron_tagger).
* An empty or whitespace-only input returns "" without raising an exception.
"""

from __future__ import annotations

import logging
import re
import string
from typing import List

logger = logging.getLogger(__name__)

# ── NLTK lazy-load helpers ─────────────────────────────────────────────────────

_stopwords_set: set[str] | None = None
_lemmatizer = None
_nltk_available: bool | None = None


def _try_import_nltk():
    """Return True if NLTK resources are available, False otherwise."""
    global _nltk_available
    if _nltk_available is not None:
        return _nltk_available
    try:
        import nltk  # noqa: F401
        from nltk.corpus import stopwords  # noqa: F401
        from nltk.stem import WordNetLemmatizer  # noqa: F401
        _nltk_available = True
    except Exception:
        _nltk_available = False
    return _nltk_available


def _get_stopwords() -> set[str]:
    global _stopwords_set
    if _stopwords_set is None:
        if _try_import_nltk():
            try:
                from nltk.corpus import stopwords as sw
                raw = set(sw.words("english"))
            except Exception:
                raw = _FALLBACK_STOPWORDS.copy()
        else:
            raw = _FALLBACK_STOPWORDS.copy()
        # Always keep negation words
        _NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "hardly",
                           "barely", "scarcely", "without"}
        _stopwords_set = raw - _NEGATION_WORDS
    return _stopwords_set


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None and _try_import_nltk():
        try:
            from nltk.stem import WordNetLemmatizer
            _lemmatizer = WordNetLemmatizer()
        except Exception:
            _lemmatizer = None
    return _lemmatizer


# Minimal fallback stopwords (does NOT include negation words)
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "as",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "same", "so", "than", "too", "very", "just", "because",
    "while", "although", "though", "since", "until", "unless",
    "of", "s", "t", "can", "will", "would", "should", "could",
    "may", "might", "shall", "also", "even", "still",
}

# Characters to strip (everything except letters, digits, whitespace, apostrophe)
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE
)
_HTML_RE = re.compile(r"<[^>]+>")
_DIGIT_RE = re.compile(r"\b\d+\b")


def clean_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = False,
    remove_digits: bool = False,
) -> str:
    """
    Clean a single review string.

    Parameters
    ----------
    text : str
        Raw input text.
    remove_stopwords : bool
        Remove common stopwords (negation words are always preserved).
    lemmatize : bool
        Apply NLTK WordNet lemmatisation (requires NLTK data).
    remove_digits : bool
        Strip standalone digits.

    Returns
    -------
    str
        Cleaned text, or "" if input is empty/whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    # Remove URLs
    text = _URL_RE.sub(" ", text)
    # Remove HTML tags
    text = _HTML_RE.sub(" ", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep apostrophes)
    text = _PUNCT_RE.sub(" ", text)
    # Optionally remove digits
    if remove_digits:
        text = _DIGIT_RE.sub(" ", text)
    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = text.split()

    # Lemmatise
    if lemmatize:
        lemmatizer = _get_lemmatizer()
        if lemmatizer is not None:
            tokens = _lemmatize_tokens(tokens, lemmatizer)

    # Remove stopwords
    if remove_stopwords:
        sw = _get_stopwords()
        tokens = [t for t in tokens if t not in sw]

    return " ".join(tokens)


def _lemmatize_tokens(tokens: List[str], lemmatizer) -> List[str]:
    """Lemmatise tokens using POS-aware NLTK lemmatisation when available."""
    try:
        import nltk
        tagged = nltk.pos_tag(tokens)
        from nltk.corpus import wordnet

        def _wordnet_pos(tag: str) -> str:
            if tag.startswith("J"):
                return wordnet.ADJ
            if tag.startswith("V"):
                return wordnet.VERB
            if tag.startswith("R"):
                return wordnet.ADV
            return wordnet.NOUN

        return [lemmatizer.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged]
    except Exception:
        # Fall back to plain lemmatisation
        return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_series(
    series: "pd.Series",  # type: ignore[name-defined]
    **kwargs,
) -> "pd.Series":  # type: ignore[name-defined]
    """
    Apply clean_text to a pandas Series.

    Parameters
    ----------
    series : pd.Series of str
    **kwargs
        Forwarded to clean_text.

    Returns
    -------
    pd.Series of str
    """
    return series.fillna("").apply(lambda x: clean_text(x, **kwargs))
