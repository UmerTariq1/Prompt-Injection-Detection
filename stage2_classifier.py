"""
stage2_classifier.py
--------------------
Stage 2 — Lightweight ML Classifier

Uses ProtectAI's DeBERTa-v3-small-prompt-injection-v2 (44M parameters).
Loaded once at startup, cached as a module-level singleton.
Runs on CPU via standard PyTorch / HuggingFace pipeline.
"""

import logging
import time
from typing import Optional

import torch
from transformers import pipeline

from utils.config import CFG
from utils.result import DetectionResult, Verdict

logger = logging.getLogger(__name__)


class _ClassifierSingleton:
    """
    Holds the loaded model + tokenizer.
    Initialized lazily on first use to keep import time fast.
    """

    def __init__(self):
        self._pipe = None
        self._model_name: str = CFG.classifier_model

    def _load(self) -> None:
        logger.info("Loading Stage 2 classifier: %s ...", self._model_name)
        t0 = time.perf_counter()
        self._pipe = self._load_pytorch()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Stage 2 classifier ready in %.0fms", elapsed)

    def _load_pytorch(self):
        """Standard HuggingFace pipeline running on CPU."""
        return pipeline(
            "text-classification",
            model=self._model_name,
            device=-1,
            max_length=CFG.classifier_max_length,
            truncation=True,
        )

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (label, confidence_score).
        Label is one of the model's output labels (e.g. 'INJECTION', 'SAFE').
        """
        if self._pipe is None:
            self._load()

        output = self._pipe(text)[0]           # [{"label": "...", "score": 0.xx}]
        return output["label"], output["score"]


# Module-level singleton
_classifier = _ClassifierSingleton()


def preload() -> None:
    """
    Explicitly load the model at startup.
    Call this from main.py to avoid latency on the first real request.
    """
    _classifier._load()


def check(text: str) -> DetectionResult:
    """
    Run the DeBERTa classifier on preprocessed text.

    Returns
    -------
    DetectionResult with:
      INJECTION  — if injection probability >= classifier_inject_threshold
      SAFE       — if injection probability <= classifier_safe_threshold
      UNCERTAIN  — if score falls between the two thresholds (triggers Stage 3)
    """
    t0 = time.perf_counter()

    try:
        label, confidence = _classifier.predict(text)
    except Exception as exc:
        logger.error("Stage 2 classifier error: %s", exc)
        # Fail open — return UNCERTAIN so Stage 3 can still try
        return DetectionResult(
            verdict=Verdict.UNCERTAIN,
            score=0.5,
            stage=2,
            reason=f"Classifier error: {exc} — escalating to deep scan",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    latency = (time.perf_counter() - t0) * 1000

    # Normalise label → injection probability
    label_upper = label.upper()
    if "INJECTION" in label_upper or "UNSAFE" in label_upper or label_upper == "1":
        injection_prob = confidence
    else:
        injection_prob = 1.0 - confidence

    raw_scores = {"label": label, "confidence": confidence, "injection_prob": injection_prob}

    if injection_prob >= CFG.classifier_inject_threshold:
        return DetectionResult(
            verdict=Verdict.INJECTION,
            score=injection_prob,
            stage=2,
            reason=f"ML classifier: injection probability {injection_prob:.3f}",
            latency_ms=latency,
            raw_scores=raw_scores,
        )

    if injection_prob <= CFG.classifier_safe_threshold:
        return DetectionResult(
            verdict=Verdict.SAFE,
            score=injection_prob,
            stage=2,
            reason=f"ML classifier: safe (injection probability {injection_prob:.3f})",
            latency_ms=latency,
            raw_scores=raw_scores,
        )

    # Uncertain band — escalate
    return DetectionResult(
        verdict=Verdict.UNCERTAIN,
        score=injection_prob,
        stage=2,
        reason=(
            f"ML classifier uncertain (score={injection_prob:.3f}, "
            f"thresholds [{CFG.classifier_safe_threshold}, {CFG.classifier_inject_threshold}])"
            " — escalating to deep scan"
        ),
        latency_ms=latency,
        raw_scores=raw_scores,
    )
