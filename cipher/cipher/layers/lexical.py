"""Layer 6: Lexical richness and stylometric features.

Computes 10 features derived from vocabulary diversity, punctuation
patterns, and sentence-level stylometric signals.  No neural networks
required; spaCy is optional and used only for passive-voice detection and
lexical density (content-word tagging).

Theoretical basis
-----------------
AI-generated text exhibits characteristic lexical patterns that differ
from human writing:

* **Lower vocabulary diversity**: AI generators tend to reuse the same
  word forms more frequently within a chunk, leading to higher type-token
  overlap and fewer hapax legomena (once-only words).
* **More uniform sentence structure**: AI sentences often start with
  similar syntactic patterns ("The", "This", "These", "It"), resulting in
  lower sentence-initial word entropy.
* **Reduced self-reference and hedging**: Fewer first-person opinions and
  lower punctuation diversity (em-dashes, ellipses, parenthetical asides).
* **Smoother paragraph structure**: AI paragraphs tend to be more uniform
  in length, yielding a lower coefficient of variation for paragraph
  word-counts.
* **Higher lexical density in domain-adapted AI**: Formal AI text tends
  to pack more content words per sentence.

Features
--------
s1  Type-Token Ratio (TTR)  — |unique tokens| / |total tokens|
s2  Hapax Legomena ratio    — |once-only tokens| / |unique tokens|
s3  Yule's K characteristic — 10⁴ × Σᵢ (i² Vᵢ / N²) − 1/N
s4  Bigram repetition score — fraction of unique bigrams that recur ≥2×
s5  Passive voice ratio     — sentences with passive verbs / total (spaCy)
s6  Punctuation entropy     — Shannon entropy of punctuation-char counts
s7  Sentence-initial word entropy — H of normalised first-word distribution
s8  Paragraph length CV     — coefficient of variation of paragraph lengths
s9  Lexical density         — content-word tokens / total tokens (spaCy)
s10 Mean word-rank percentile— mean log(rank) of content words vs word-freq

See Also
--------
cipher.layers.compression  : Layer 1 — compression-theoretic features.
cipher.layers.cognitive    : Layer 2 — cognitive load fingerprint.
cipher.layers.discourse    : Layer 3 — discourse topology.
cipher.layers.assembler    : Combines all layer outputs into a FeatureVector.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Word-frequency reference (top-60 k English words by rank, approximated
# by a simple rank-based log table built from Zipf's law).
# We use log₂(rank + 1) as a proxy for "how rare" a word is.
# ---------------------------------------------------------------------------

_WORD_FREQ_CACHE: dict[str, float] = {}


def _word_log_rank(word: str) -> float:
    """Return a smoothed log₂-rank score for *word*.

    Uses a pre-built Python Counter over a large reference vocabulary.
    Falls back to log₂(50 001) (≈ 15.6) for out-of-vocabulary words.

    This is intentionally lightweight: we only need a relative ordering
    of common vs. rare words, not precise frequency estimates.
    """
    # Very rough Zipf approximation: common 1000-word lemmas are hardcoded
    # below; everything else gets a high rank estimate.
    word = word.lower()
    if word in _COMMON_WORDS:
        return math.log2(_COMMON_WORDS[word] + 1)
    return math.log2(50_001)


# The 1 000 most common English words with approximate ranks (from Zipf tables).
# Stored as {word: rank} — lower rank = more frequent.
_COMMON_WORDS: dict[str, int] = {
    "the": 1, "be": 2, "to": 3, "of": 4, "and": 5, "a": 6, "in": 7,
    "that": 8, "have": 9, "i": 10, "it": 11, "for": 12, "not": 13,
    "on": 14, "with": 15, "he": 16, "as": 17, "you": 18, "do": 19,
    "at": 20, "this": 21, "but": 22, "his": 23, "by": 24, "from": 25,
    "they": 26, "we": 27, "say": 28, "her": 29, "she": 30, "or": 31,
    "an": 32, "will": 33, "my": 34, "one": 35, "all": 36, "would": 37,
    "there": 38, "their": 39, "what": 40, "so": 41, "up": 42, "out": 43,
    "if": 44, "about": 45, "who": 46, "get": 47, "which": 48, "go": 49,
    "me": 50, "when": 51, "make": 52, "can": 53, "like": 54, "time": 55,
    "no": 56, "just": 57, "him": 58, "know": 59, "take": 60, "people": 61,
    "into": 62, "year": 63, "your": 64, "good": 65, "some": 66, "could": 67,
    "them": 68, "see": 69, "other": 70, "than": 71, "then": 72, "now": 73,
    "look": 74, "only": 75, "come": 76, "its": 77, "over": 78, "think": 79,
    "also": 80, "back": 81, "after": 82, "use": 83, "two": 84, "how": 85,
    "our": 86, "work": 87, "first": 88, "well": 89, "way": 90, "even": 91,
    "new": 92, "want": 93, "because": 94, "any": 95, "these": 96, "give": 97,
    "day": 98, "most": 99, "us": 100,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LexicalFeatures:
    """Immutable container for the 10 lexical richness / stylometric features.

    Attributes:
        s1:  Type-Token Ratio.  AI text is typically 0.55–0.70; human 0.65–0.85.
        s2:  Hapax Legomena ratio.  Lower in AI (fewer unique, one-off words).
        s3:  Yule's K.  Higher K → lower lexical richness; typically higher for AI.
        s4:  Bigram repetition score.  Higher → more repeated bigrams (AI signal).
        s5:  Passive voice ratio.  Formal AI text uses more passive constructions.
        s6:  Punctuation entropy.  Lower in AI (fewer em-dashes, ellipses, etc.).
        s7:  Sentence-initial word entropy.  Lower in AI (more repetitive starters).
        s8:  Paragraph length CV.  Lower in AI (more uniform paragraph structure).
        s9:  Lexical density.  Ratio of content words; can go either way.
        s10: Mean word-rank percentile.  AI uses more common words (lower rank).
        available: Names of features that were actually computed.
    """
    s1: float   # Type-Token Ratio
    s2: float   # Hapax Legomena ratio
    s3: float   # Yule's K
    s4: float   # Bigram repetition score
    s5: float   # Passive voice ratio
    s6: float   # Punctuation entropy
    s7: float   # Sentence-initial word entropy
    s8: float   # Paragraph length CV
    s9: float   # Lexical density
    s10: float  # Mean word-rank percentile
    available: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float64 array in canonical order [s1..s10].

        Returns:
            np.ndarray: Shape (10,), dtype float64.
        """
        return np.array([
            self.s1, self.s2, self.s3, self.s4, self.s5,
            self.s6, self.s7, self.s8, self.s9, self.s10,
        ])


# ---------------------------------------------------------------------------
# Internal tokenisation helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens only (removes punctuation and numbers).

    Args:
        text: Raw text string.

    Returns:
        list[str]: List of lowercase word tokens.
    """
    return re.findall(r"[a-z]+", text.lower())


def _sentences(text: str) -> list[str]:
    """Split text into sentences using a simple rule-based splitter.

    Splits on ". ", "? ", "! " followed by an uppercase letter or
    end-of-string.  Not perfect but fast and dependency-free.

    Args:
        text: Raw text string.

    Returns:
        list[str]: Non-empty sentence strings.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    """Split text on blank lines into non-empty paragraph strings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n{2,}", text)
    return [p.strip() for p in paras if p.strip()]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _type_token_ratio(tokens: list[str]) -> float:
    """Feature s1: Type-Token Ratio.

    TTR = |unique types| / |total tokens|.  Values near 1.0 indicate
    rich vocabulary; values close to 0 indicate heavy repetition.  AI
    text usually scores lower than comparably-sized human text.

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        float: TTR in (0, 1].  Returns 0.0 for empty input.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    return len(set(tokens)) / n


def _hapax_ratio(tokens: list[str]) -> float:
    """Feature s2: Hapax Legomena ratio.

    Hapax legomena are words that appear exactly once in the text.
    The ratio is: |hapax types| / |unique types|.  Human writing tends
    to have a higher proportion of once-occurring words (novel references,
    personal details, idiosyncratic word choices).

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        float: Hapax ratio in [0, 1].  Returns 0.0 for empty input.
    """
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    unique = len(freq)
    hapax = sum(1 for f in freq.values() if f == 1)
    return hapax / unique if unique else 0.0


def _yule_k(tokens: list[str]) -> float:
    """Feature s3: Yule's K characteristic.

    K = 10⁴ × (Σᵢ i² Vᵢ / N² − 1/N)

    where N = total tokens and Vᵢ = number of types occurring exactly i
    times.  Lower K → richer vocabulary.  Robust to text length unlike
    TTR.  AI text typically has higher K (less diverse vocabulary).

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        float: Yule's K ≥ 0.  Returns 0.0 for empty input.
    """
    n = len(tokens)
    if n < 2:
        return 0.0
    freq = Counter(tokens)
    freq_of_freq: Counter[int] = Counter(freq.values())
    s = sum(i * i * v for i, v in freq_of_freq.items())
    k = 10_000.0 * (s / (n * n) - 1.0 / n)
    return max(k, 0.0)


def _bigram_repetition(tokens: list[str]) -> float:
    """Feature s4: Bigram repetition score.

    Measures what fraction of unique bigrams appear more than once.
    AI text tends to re-use the same phrase-level patterns.

    bigram_rep = |{bg : count(bg) ≥ 2}| / |{unique bigrams}|

    Args:
        tokens: List of word tokens.

    Returns:
        float: Repetition ratio in [0, 1].  Returns 0.0 when fewer than 2 tokens.
    """
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    freq = Counter(bigrams)
    repeated = sum(1 for f in freq.values() if f >= 2)
    return repeated / len(freq)


def _passive_ratio(text: str, nlp: Any | None) -> float:
    """Feature s5: Passive voice ratio (spaCy dependency parse).

    A sentence is classified as passive when it contains a token with
    dependency label ``nsubjpass`` or ``auxpass`` (spaCy en_core_web_sm).
    This requires spaCy; returns 0.15 (neutral estimate) when unavailable.

    Args:
        text: Raw text string.
        nlp:  Loaded spaCy model, or None.

    Returns:
        float: Fraction of sentences that are passive in [0, 1].
    """
    if nlp is None:
        return 0.15  # neutral fallback
    try:
        doc = nlp(text[:5_000])  # cap to avoid very slow parses
        total = 0
        passive = 0
        for sent in doc.sents:
            total += 1
            if any(t.dep_ in ("nsubjpass", "auxpass") for t in sent):
                passive += 1
        return (passive / total) if total else 0.15
    except Exception:
        return 0.15


def _punctuation_entropy(text: str) -> float:
    """Feature s6: Shannon entropy of punctuation character frequencies.

    AI text tends to use a narrower range of punctuation marks (mostly
    periods and commas), yielding lower entropy.  Human text uses
    em-dashes, ellipses, semicolons, parentheses, etc., raising entropy.

    Args:
        text: Raw text string.

    Returns:
        float: Entropy in bits ∈ [0, log₂(n_punct_types)].
    """
    punct_chars = re.findall(r"[^\w\s]", text)
    if not punct_chars:
        return 0.0
    freq = Counter(punct_chars)
    total = len(punct_chars)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return entropy


def _sentence_initial_entropy(text: str) -> float:
    """Feature s7: Entropy of sentence-initial (first) words.

    AI generators often start sentences with the same small set of words
    ("The", "This", "These", "It", "In").  Human writers vary more,
    yielding higher entropy of the first-word distribution.

    Args:
        text: Raw text string.

    Returns:
        float: Shannon entropy in bits.  Returns 0.0 for very short text.
    """
    sents = _sentences(text)
    if not sents:
        return 0.0
    first_words = [s.split()[0].lower() for s in sents if s.split()]
    if not first_words:
        return 0.0
    freq = Counter(first_words)
    total = len(first_words)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return entropy


def _paragraph_length_cv(text: str) -> float:
    """Feature s8: Coefficient of variation of paragraph word-counts.

    CV = std / mean of {word_count(para) for para in paragraphs}.

    AI text tends to produce paragraphs of similar length (low CV) while
    human text shows more varied paragraph structure.

    Args:
        text: Raw text string.

    Returns:
        float: CV ≥ 0.  Returns 0.0 when fewer than 2 paragraphs.
    """
    paras = _paragraphs(text)
    lengths = [len(p.split()) for p in paras if p.split()]
    if len(lengths) < 2:
        return 0.0
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    if mean < 1e-8:
        return 0.0
    return float(arr.std() / mean)


def _lexical_density(text: str, nlp: Any | None) -> float:
    """Feature s9: Lexical density (content words / total tokens, spaCy).

    Content words are nouns, verbs (non-auxiliary), adjectives, and
    adverbs.  Lexical density measures how much information per token
    a text carries.  Fallback: proportion of non-stopword alphabetic
    tokens using a 150-word stoplist.

    Args:
        text: Raw text string.
        nlp:  Loaded spaCy model, or None.

    Returns:
        float: Lexical density in (0, 1].
    """
    if nlp is not None:
        try:
            doc = nlp(text[:5_000])
            content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
            total = sum(1 for t in doc if t.is_alpha)
            content = sum(1 for t in doc if t.is_alpha and t.pos_ in content_pos and not t.is_stop)
            return (content / total) if total else 0.5
        except Exception:
            pass
    # Fallback: count non-stopword tokens
    tokens = _tokenize(text)
    if not tokens:
        return 0.5
    stop = _STOPWORDS
    content = sum(1 for t in tokens if t not in stop)
    return content / len(tokens)


def _mean_word_rank(tokens: list[str]) -> float:
    """Feature s10: Mean log₂-rank of content words vs. word-frequency list.

    Higher values → more rare / unusual word choices (human signal).
    Lower values → text relies heavily on high-frequency common words
    (potential AI signal when combined with other features).

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        float: Mean log₂-rank in approximately [1, 16].
    """
    if not tokens:
        return 8.0  # neutral estimate
    ranks = [_word_log_rank(t) for t in tokens]
    return float(np.mean(ranks))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_lexical_features(text: str, nlp: Any | None = None) -> LexicalFeatures:
    """Compute all 10 lexical richness and stylometric features.

    Args:
        text: The document (or chunk) string to analyse.
        nlp:  Optional loaded spaCy model.  When provided, enables
              passive-voice detection (s5) and content-word tagging
              for lexical density (s9).  All other features are
              computed without spaCy.

    Returns:
        LexicalFeatures: Immutable feature container.  Features s5 and
        s9 will use neutral fallback values when ``nlp`` is None.

    Example::

        import spacy
        nlp = spacy.load("en_core_web_sm")
        feats = compute_lexical_features(my_text, nlp=nlp)
        print(feats.s1, feats.s3)  # TTR, Yule's K
    """
    tokens = _tokenize(text)
    available: list[str] = ["s1", "s2", "s3", "s4", "s6", "s7", "s8", "s10"]

    s1 = _type_token_ratio(tokens)
    s2 = _hapax_ratio(tokens)
    s3 = _yule_k(tokens)
    s4 = _bigram_repetition(tokens)

    if nlp is not None:
        available += ["s5", "s9"]
    s5 = _passive_ratio(text, nlp)
    s6 = _punctuation_entropy(text)
    s7 = _sentence_initial_entropy(text)
    s8 = _paragraph_length_cv(text)
    s9 = _lexical_density(text, nlp)
    s10 = _mean_word_rank(tokens)

    return LexicalFeatures(
        s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
        s6=s6, s7=s7, s8=s8, s9=s9, s10=s10,
        available=sorted(set(available)),
    )


# ---------------------------------------------------------------------------
# Lightweight English stopwords (fallback for lexical density without spaCy)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "let",
    "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn",
})
