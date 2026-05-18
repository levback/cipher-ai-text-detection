# CIPHER Documentation

**CIPHER** — *Cognitive Information Pattern Hierarchy for Entropy-based Recognition*

CIPHER is a Python library for detecting AI-generated text. It operates entirely offline at inference time: no LLM queries, no cloud calls, no API keys. Detection is based on measurable statistical properties of text drawn from information theory, cognitive linguistics, and discourse topology.

---

## Documentation Map

| Document | Description |
|---|---|
| **[User Guide](user_guide.md)** | Installation, CLI usage, Python API, worked examples |
| **[Architecture](architecture.md)** | System design, six-layer pipeline, feature engineering decisions |
| **[API Reference](api_reference.md)** | All public classes, methods, return types, and exceptions |
| **[Dataset Documentation](datasets.md)** | Training data sources, statistics, preprocessing, citations |

---

## Quick Start

```bash
# Install
pip install cipher-detect

# Detect text
cipher detect "The model generated highly coherent text across all measured parameters."

# Detect a file
cipher detect-file report.pdf

# Train on your own data
cipher train meta --data data/train.csv --output-dir models/
```

Python API:

```python
from cipher import CIPHERDetector

detector = CIPHERDetector()
result = detector.detect("Your text here.")
print(f"{result.label}: {result.document_score:.1%} AI probability")
```

---

## Design Philosophy

CIPHER was designed around three constraints:

1. **No inference-time LLM** — detection must work on air-gapped systems.
2. **Calibrated uncertainty** — every score comes with a conformal prediction interval at a user-specified coverage level (default 95%).
3. **Explainability** — every document score decomposes into per-layer contributions and per-sentence attribution.

---

## Cross-References

- Feature engineering details → [Architecture § Layer Descriptions](architecture.md#layer-descriptions)
- Training your own model → [User Guide § Training](user_guide.md#training)
- Dataset sources used in the pre-trained model → [Dataset Documentation](datasets.md)
- `CIPHERDetector` class reference → [API Reference § CIPHERDetector](api_reference.md#cipherdetector)
- `DetectionResult` fields → [API Reference § DetectionResult](api_reference.md#detectionresult)
- Security considerations → [User Guide § Security](user_guide.md#security)

---

## Version

Current: **1.0.0** (heuristic meta-learner, AUROC 0.915 on held-out public datasets).

See [Architecture § Model Performance](architecture.md#model-performance) for benchmarks.
