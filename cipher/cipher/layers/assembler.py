"""Feature assembler: combine all layer outputs into a named feature vector.

This module is the integration point for the CIPHER feature pipeline.  It
accepts the output from each feature layer (compression, cognitive, discourse,
lexical) and an optional manifold score (Layer 4) and packs them into a
single, ordered ``FeatureVector``.

The canonical feature ordering is:
    l1..l7   (7)  compression-theoretic (Layer 1)
    c1..c8   (8)  cognitive load (Layer 2)
    d1..d7   (7)  discourse topology (Layer 3)
    m1       (1)  contrastive manifold score (Layer 4, optional)
    s1..s10 (10)  lexical richness & stylometric (Layer 6)
                  Total: 32 without manifold / 33 with manifold

See Also
--------
cipher.layers.compression  : Layer 1 feature computation.
cipher.layers.cognitive    : Layer 2 feature computation.
cipher.layers.discourse    : Layer 3 feature computation.
cipher.layers.lexical      : Layer 6 feature computation.
cipher.models.meta_learner : Consumes the FeatureVector from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cipher.layers.cognitive import CognitiveFeatures
from cipher.layers.compression import CompressionFeatures
from cipher.layers.discourse import DiscourseFeatures
from cipher.layers.lexical import LexicalFeatures


@dataclass(frozen=True)
class FeatureVector:
    """Immutable, named container for the full CIPHER feature representation.

    Combines the outputs of all feature layers into a single object.  The
    ``to_array()`` method serialises them into the canonical 1-D vector
    expected by ``MetaLearner.predict_proba``.

    Attributes:
        compression: Layer 1 output (7 compression-theoretic features).
        cognitive: Layer 2 output (8 cognitive load features).
        discourse: Layer 3 output (7 discourse topology features).
        lexical: Layer 6 output (10 lexical richness & stylometric features).
        manifold_score: Layer 4 contrastive manifold score P(AI | z),
            or None when the neural encoder is not loaded.
        FEATURE_NAMES: Class-level tuple of 33 canonical feature names
            in the same order as ``to_array()``.

    Example::

        from cipher.layers.assembler import FeatureVector
        fv = FeatureVector(comp, cog, disc, lex, manifold_score=0.72)
        arr = fv.to_array()          # shape (33,)
        d   = fv.as_dict()           # {name: value}
    """
    compression: CompressionFeatures
    cognitive: CognitiveFeatures
    discourse: DiscourseFeatures
    lexical: LexicalFeatures
    # Layer 4 manifold score (optional — None when no trained model)
    manifold_score: float | None = None

    # Ordered feature names matching to_array()
    FEATURE_NAMES: tuple[str, ...] = (
        # Layer 1 — compression-theoretic (7)
        "l1_lzc", "l2_ncd_ai", "l3_ncd_human",
        "l4_redundancy", "l5_asym", "l6_ent_var", "l7_ent_autocorr",
        # Layer 2 — cognitive load (8)
        "c1_sl_kl", "c2_hurst", "c3_coh_var", "c4_coh_autocorr", "c5_coh_mean",
        "c6_epistemic", "c7_opinion", "c8_correction",
        # Layer 3 — discourse topology (7)
        "d1_coref_density", "d2_clustering", "d3_deg_std",
        "d4_topic_var", "d5_topic_max",
        "d6_syn_var", "d7_syn_skew",
        # Layer 4 — contrastive manifold (1)
        "m1_manifold",
        # Layer 6 — lexical richness & stylometric (10)
        "s1_ttr", "s2_hapax", "s3_yule_k", "s4_bigram_rep",
        "s5_passive", "s6_punct_ent", "s7_sent_init_ent",
        "s8_para_cv", "s9_lex_density", "s10_word_rank",
    )

    def to_array(self, include_manifold: bool = True) -> np.ndarray:
        """Serialise feature vector to a 1-D float64 numpy array.

        The canonical ordering is:
            [l1..l7] [c1..c8] [d1..d7] [m1] [s1..s10]

        Args:
            include_manifold: When True, includes the manifold score as
                the 23rd element (m1) before the lexical features.
                When False, the vector has 32 elements (no m1).
                The MetaLearner automatically adapts to both sizes.

        Returns:
            np.ndarray: Shape (32,) or (33,), dtype float64.  When
            ``manifold_score`` is None and ``include_manifold=True``,
            a value of 0.5 is used as the neutral placeholder.
        """
        parts = [
            self.compression.to_array(),
            self.cognitive.to_array(),
            self.discourse.to_array(),
        ]
        vec = np.concatenate(parts)
        if include_manifold:
            m1 = self.manifold_score if self.manifold_score is not None else 0.5
            vec = np.append(vec, m1)
        vec = np.concatenate([vec, self.lexical.to_array()])
        return vec

    def available_features(self, include_manifold: bool = True) -> list[str]:
        """Return a list of feature names that were computed from real data.

        Features backed by fallback default values are excluded.  This is
        useful for interpretability and for identifying which layers were
        active during a detection run.

        Args:
            include_manifold: Whether to include ``m1_manifold`` in the
                result when the manifold score is not None.

        Returns:
            list[str]: Subset of ``FEATURE_NAMES`` for which real values
            are available.
        """
        # Features 0-21: l + c + d layers
        core_names = list(self.FEATURE_NAMES[:22])
        avail = (
            self.compression.available
            + self.cognitive.available
            + self.discourse.available
        )
        result = [n for n in core_names if any(a in n for a in avail)]
        if include_manifold and self.manifold_score is not None:
            result.append("m1_manifold")
        # Lexical features (always available)
        result += list(self.FEATURE_NAMES[23:])
        return result

    def as_dict(self) -> dict[str, float]:
        """Return a name-to-value dictionary for all 23 features.

        Useful for logging, serialisation, and human-readable inspection.

        Returns:
            dict[str, float]: Mapping from each of the 23 ``FEATURE_NAMES``
            to its corresponding float value.
        """
        vec = self.to_array(include_manifold=True)
        return dict(zip(self.FEATURE_NAMES, vec.tolist()))
