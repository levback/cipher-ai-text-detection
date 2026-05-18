"""Layer 3: Discourse topology features.

Computes 7 features from co-reference graphs, topic boundaries, and
syntactic parse trees.  Requires spaCy for full functionality; coreferee
and BERTopic are optional.  All features fall back gracefully when the
dependencies are absent.

The key intuition is that human discourse is *topologically richer* than
AI discourse: entities recur in unpredictable ways (low graph density
but high local clustering), topics shift abruptly at paragraph boundaries,
and syntactic trees reach greater depths due to embedded clauses.

Features
--------
d1  co-reference graph density
d2  mean clustering coefficient of co-reference graph
d3  standard deviation of node degree in co-reference graph
d4  topic-boundary JS-divergence variance
d5  maximum topic-boundary JS divergence
d6  syntactic depth per-sentence variance
d7  syntactic depth per-sentence skewness

See Also
--------
cipher.layers.cognitive    : Cognitive load features (Layer 2).
cipher.layers.compression  : Compression-theoretic features (Layer 1).
cipher.layers.assembler    : Combines all layer outputs into a FeatureVector.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


@dataclass(frozen=True)
class DiscourseFeatures:
    """Immutable container for the 7 discourse topology feature values.

    Attributes:
        d1: Co-reference graph density = |E| / (|V|*(|V|-1)/2).  Higher
            values indicate more densely interconnected entity mentions.
        d2: Mean clustering coefficient of the co-reference graph.
        d3: Standard deviation of node degree.  Low std implies uniform
            entity connectivity (AI signal).
        d4: Variance of Jensen-Shannon divergence between consecutive
            paragraph topic distributions.  Low variance implies uniform
            topic progression (AI signal).
        d5: Maximum JS divergence across consecutive paragraph pairs.
        d6: Variance of per-sentence syntactic parse tree depth.
        d7: Skewness of per-sentence syntactic depth distribution.
        available: Names of features computed from real data.
    """
    d1: float  # co-reference graph density
    d2: float  # mean clustering coefficient
    d3: float  # degree std dev
    d4: float  # topic JS-divergence variance
    d5: float  # max topic JS divergence
    d6: float  # syntactic depth variance
    d7: float  # syntactic depth skewness
    available: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float64 array in canonical order [d1..d7].

        Returns:
            np.ndarray: Shape (7,), dtype float64.
        """
        return np.array([self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7])


# ---------------------------------------------------------------------------
# Co-reference graph features (d1, d2, d3)
# ---------------------------------------------------------------------------

def _coref_graph_from_coreferee(doc: Any) -> Any | None:
    """Build a NetworkX graph from a coreferee-annotated spaCy Doc.

    Each co-reference chain contributes one node (the first mention's
    surface form).  Nodes that co-occur in the same sentence are
    connected by an undirected edge.

    Args:
        doc: A spaCy ``Doc`` with the coreferee extension loaded.

    Returns:
        nx.Graph | None: Co-reference graph, or None when coreferee is
        unavailable or no chains were detected.
    """
    if not _NX_AVAILABLE:
        return None
    try:
        chains = doc._.coref_chains
    except AttributeError:
        return None
    if not chains:
        return None

    g = nx.Graph()
    for chain in chains:
        cluster_mentions = [doc[mention[0]] for mention in chain]
        texts = [t.text.lower() for t in cluster_mentions]
        node = texts[0]
        g.add_node(node)

    # Add edges based on sentence co-occurrence
    for sent in doc.sents:
        sent_text = sent.text.lower()
        nodes_in_sent = [n for n in g.nodes if n in sent_text]
        for i in range(len(nodes_in_sent)):
            for j in range(i + 1, len(nodes_in_sent)):
                g.add_edge(nodes_in_sent[i], nodes_in_sent[j])
    return g


def _coref_graph_from_ner(doc: Any) -> Any | None:
    """Fallback: build an entity co-occurrence graph from spaCy NER output.

    Constructs a co-occurrence graph where two named entities are
    connected if they appear in the same sentence.  Less precise than
    full co-reference resolution but always available when spaCy is loaded.

    Args:
        doc: A spaCy ``Doc`` object.

    Returns:
        nx.Graph | None: Co-occurrence graph with at least 2 nodes, or
        None if fewer than 2 unique entities are found.
    """
    if not _NX_AVAILABLE:
        return None

    g: Any = nx.Graph()
    # Extract unique entity surface forms
    entities = {ent.text.lower() for ent in doc.ents}
    for ent in entities:
        g.add_node(ent)

    # Add edges for entities co-occurring in the same sentence
    for sent in doc.sents:
        sent_ents = [ent.text.lower() for ent in sent.ents]
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                if sent_ents[i] != sent_ents[j]:
                    g.add_edge(sent_ents[i], sent_ents[j])
    return g if len(g.nodes) >= 2 else None  # type: ignore[no-any-return]


def coref_graph_features(doc: Any) -> tuple[float, float, float]:
    """Compute d1 (density), d2 (mean clustering), d3 (degree std).

    Tries coreferee chains first; falls back to NER co-occurrence when
    coreferee is unavailable.  Returns fallback values when no graph
    with at least 2 nodes can be constructed.

    Args:
        doc: A spaCy ``Doc`` object (with or without coreferee).

    Returns:
        tuple[float, float, float]: (d1_density, d2_mean_clustering, d3_deg_std).
        Fallback: (0.3, 0.3, 1.0) when networkx or spaCy is unavailable.
    """
    if not _NX_AVAILABLE:
        return 0.3, 0.3, 1.0

    # Try coreferee first, fall back to NER
    g = _coref_graph_from_coreferee(doc)
    if g is None or len(g.nodes) < 2:
        g = _coref_graph_from_ner(doc)
    if g is None or len(g.nodes) < 2:
        return 0.3, 0.3, 1.0

    n = len(g.nodes)
    max_edges = n * (n - 1) / 2
    density = len(g.edges) / max_edges if max_edges > 0 else 0.0

    try:
        mean_clust = nx.average_clustering(g)
    except Exception:
        mean_clust = 0.0

    degrees = [d for _, d in g.degree()]
    deg_std = float(np.std(degrees)) if degrees else 0.0

    return float(density), float(mean_clust), deg_std


# ---------------------------------------------------------------------------
# Topic boundary features (d4, d5)
# ---------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete probability distributions.

    Symmetrised, smoothed variant of KL divergence:

        JSD(P || Q) = 0.5*KL(P || M) + 0.5*KL(Q || M),   M = (P + Q) / 2

    where KL(P || Q) = sum P(x)*log(P(x)/Q(x)).

    JSD is bounded in [0, ln(2)] using natural log (~[0, 0.693]).
    High JSD between consecutive paragraph distributions signals a topic
    boundary.  Small epsilon clamps are applied to avoid log(0).

    Args:
        p: First probability distribution array.
        q: Second probability distribution array.

    Returns:
        float: JS divergence in [0, ln(2)].
    """
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-10, None)
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _tfidf_paragraph_vectors(paragraphs: list[str]) -> np.ndarray:
    """Compute L2-normalised TF-IDF vectors for a list of paragraph strings.

    Uses smooth IDF: idf(t) = log((1 + N) / (1 + df(t))) + 1
    to prevent division-by-zero for terms absent from the corpus.
    Dependency-free implementation (no sklearn required).

    Args:
        paragraphs: List of paragraph strings.

    Returns:
        np.ndarray: Shape (n_paragraphs, n_vocab), L2-normalised TF-IDF.
        Returns (n_paragraphs, 1) zero matrix if vocabulary is empty.
    """
    import math

    # Build vocabulary
    vocab: dict[str, int] = {}
    tokenized = []
    for para in paragraphs:
        tokens = para.lower().split()
        tokenized.append(tokens)
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    n_docs = len(paragraphs)
    n_terms = len(vocab)
    if n_terms == 0 or n_docs == 0:
        return np.zeros((n_docs, 1))

    tf = np.zeros((n_docs, n_terms))
    df = np.zeros(n_terms)

    for i, tokens in enumerate(tokenized):
        for tok in tokens:
            tf[i, vocab[tok]] += 1
        for term_idx in {vocab[tok] for tok in tokens if tok in vocab}:
            df[term_idx] += 1

    # Normalize TF
    row_sums = tf.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    tf = tf / row_sums

    # IDF
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    tfidf = tf * idf

    # L2 normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return tfidf / norms


def _topic_distributions(paragraphs: list[str], n_topics: int = 10) -> np.ndarray:
    """Return soft topic distributions via TF-IDF + truncated SVD (LSA).

    Projects TF-IDF vectors into a low-dimensional semantic space via
    Singular Value Decomposition (Latent Semantic Analysis), then
    converts each row into a valid probability distribution.  Falls
    back to softmax-normalised TF-IDF when SVD is inapplicable.

    Args:
        paragraphs: List of paragraph strings.
        n_topics: Number of latent semantic dimensions (default 10).

    Returns:
        np.ndarray: Shape (n_paragraphs, k), each row sums to 1.0.
    """
    vecs = _tfidf_paragraph_vectors(paragraphs)
    if vecs.shape[1] < n_topics or len(paragraphs) < 3:
        # Fallback: use raw TF-IDF vectors normalized to sum to 1
        softmax = np.exp(vecs - vecs.max(axis=1, keepdims=True))
        softmax /= softmax.sum(axis=1, keepdims=True)
        return softmax

    try:
        from numpy.linalg import svd

        U, s, Vt = svd(vecs, full_matrices=False)
        k = min(n_topics, len(s))
        reduced = U[:, :k] * s[:k]
        # Shift to non-negative and normalize to distribution
        reduced -= reduced.min(axis=1, keepdims=True)
        row_sums = reduced.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        return reduced / row_sums
    except Exception:
        softmax = np.exp(vecs - vecs.max(axis=1, keepdims=True))
        softmax /= softmax.sum(axis=1, keepdims=True)
        return softmax


def topic_boundary_features(paragraphs: list[str]) -> tuple[float, float]:
    """Features d4 (JS-divergence variance) and d5 (max JS divergence).

    Computes a soft topic distribution per paragraph, then measures the
    Jensen-Shannon divergence between each consecutive pair.  Derives:

        d4 = Var[JSD(para_i, para_{i+1})]
        d5 = max[JSD(para_i, para_{i+1})]

    Low d4 and d5 imply smooth, uniform topic flow (AI pattern).  A high
    d5 signals at least one sharp topic boundary (human pattern).

    Args:
        paragraphs: List of paragraph strings.  Needs >= 3 non-empty
            paragraphs for meaningful results.

    Returns:
        tuple[float, float]: (d4_js_variance, d5_js_max).
        Returns (0.0, 0.0) for fewer than 3 paragraphs.
    """
    clean = [p.strip() for p in paragraphs if p.strip()]
    if len(clean) < 3:
        # Return near-zero defaults — the training distribution is tightly clustered
        # near 0.0, so 0.1 would produce an outlier z-score and bias the prediction.
        return 0.0, 0.0

    distributions = _topic_distributions(clean)
    js_divs = [
        _js_divergence(distributions[i], distributions[i + 1])
        for i in range(len(distributions) - 1)
    ]
    if not js_divs:
        return 0.1, 0.3

    arr = np.array(js_divs)
    return float(np.var(arr)), float(np.max(arr))


# ---------------------------------------------------------------------------
# Syntactic depth features (d6, d7)
# ---------------------------------------------------------------------------

def _max_dep_depth(token: Any, memo: dict[int, int]) -> int:
    """Recursively compute maximum dependency subtree depth for *token*.

    Depth is the length of the longest root-to-leaf path:
    leaf nodes have depth 0, their parents have depth 1, and so on.
    Memoisation via *memo* prevents redundant recursion on shared
    subtree nodes within the same sentence.

    Args:
        token: A spaCy ``Token`` object.
        memo: Memoisation dict mapping token index to computed depth.

    Returns:
        int: Maximum depth of the subtree rooted at *token*.
    """
    if token.i in memo:
        return memo[token.i]
    children = list(token.children)
    if not children:
        memo[token.i] = 0
        return 0
    depth = 1 + max(_max_dep_depth(child, memo) for child in children)
    memo[token.i] = depth
    return depth


def syntactic_depth_features(doc: Any) -> tuple[float, float]:
    """Features d6 (per-sentence depth variance) and d7 (skewness).

    Computes the maximum dependency-tree depth for each sentence's ROOT
    token, then derives variance and skewness across all sentences.

    Deep, varied syntax (high d6, positive d7) characterises complex
    human prose with embedded clauses.  AI text tends toward shallower,
    more uniform sentence structures.

    Args:
        doc: A spaCy ``Doc`` with a dependency parse.

    Returns:
        tuple[float, float]: (d6_variance, d7_skewness).
        Returns (1.0, 0.0) for fewer than 3 sentences.
    """
    depths: list[int] = []
    for sent in doc.sents:
        memo: dict[int, int] = {}
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if not roots:
            continue
        depths.append(_max_dep_depth(roots[0], memo))

    if len(depths) < 3:
        return 1.0, 0.0

    arr = np.array(depths, dtype=float)
    variance = float(np.var(arr))
    skewness = float(stats_skewness(arr))
    return variance, skewness


def stats_skewness(arr: np.ndarray) -> float:
    """Compute sample skewness (third standardised central moment) of *arr*.

    Skewness gamma_1 = E[(X - mu)^3] / sigma^3

    Positive skew: tail extends right (most values < mean).
    Negative skew: tail extends left.
    Returns 0.0 for constant arrays or n < 3.

    Args:
        arr: 1-D numeric array.

    Returns:
        float: Skewness; approximately in (-3, 3) for natural data.
    """
    n = len(arr)
    if n < 3:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float(((arr - mean) ** 3).mean() / std**3)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_discourse_features(
    text: str,
    nlp: Any | None = None,
) -> DiscourseFeatures:
    """Compute all 7 discourse topology features for *text*.

    Public entry point for Layer 3 of the CIPHER pipeline.  Features
    d1-d3 and d6-d7 require a loaded spaCy model; d4 and d5 are always
    computed using the built-in TF-IDF/SVD implementation.

    When ``nlp`` is None, d1/d2/d3 default to (0.3, 0.3, 1.0) and
    d6/d7 default to (2.0, 0.5) -- values matching the MetaLearner
    normalisation means so their z-scores are 0 (neutral contribution).

    Paragraph segmentation (tried in order):
    1. Double-newline boundaries.
    2. Single-newline lines with >10 words.
    3. 200-word sliding windows.

    Args:
        text: The document to analyse.  Truncated to 50 000 chars before
            spaCy processing to respect memory limits.
        nlp: Pre-loaded spaCy language model.  Pass None to skip
            spaCy-dependent features.

    Returns:
        DiscourseFeatures: Dataclass with all 7 feature values and an
        ``available`` list of features computed from real data.
    """
    available: list[str] = []

    if nlp is not None:
        # Limit to 50k chars to avoid spaCy's default 1M limit
        doc = nlp(text[:50_000])

        d1, d2, d3 = coref_graph_features(doc)
        available.extend(["d1", "d2", "d3"])

        d6, d7 = syntactic_depth_features(doc)
        available.extend(["d6", "d7"])
    else:
        # Defaults that represent "unknown" state
        d1, d2, d3 = 0.3, 0.3, 1.0
        d6, d7 = 2.0, 0.5

    # Topic features use simple TF-IDF; no spaCy needed
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        # Try single newline splits
        paragraphs = [p for p in text.split("\n") if len(p.split()) > 10]
    if len(paragraphs) < 3:
        # Fall back to 200-word windows
        words = text.split()
        paragraphs = [
            " ".join(words[i : i + 200])
            for i in range(0, len(words), 200)
        ]

    d4, d5 = topic_boundary_features(paragraphs)
    available.extend(["d4", "d5"])

    return DiscourseFeatures(
        d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6, d7=d7,
        available=available,
    )
