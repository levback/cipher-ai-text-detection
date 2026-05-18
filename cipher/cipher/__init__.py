"""CIPHER: Cognitive Information Pattern Hierarchy for Entropy-based Recognition.

AI-generated text detection library that works without calling an LLM at inference time.

Quick start
-----------
>>> from cipher import CIPHERDetector
>>> detector = CIPHERDetector()
>>> result = detector.detect("The quick brown fox jumps over the lazy dog. " * 20)
>>> print(f"Score: {result.document_score:.2%}  Label: {result.label}")
"""

from cipher.detect import CIPHERDetector, DetectionResult, SignalBreakdown

__version__ = "1.0.0"
__all__ = ["CIPHERDetector", "DetectionResult", "SignalBreakdown"]
