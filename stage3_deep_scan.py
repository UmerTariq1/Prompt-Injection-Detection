"""
stage3_deep_scan.py
-------------------
Stage 3 — Deep Scan (last line of defence)

Triggered only when Stage 2 returns UNCERTAIN (~5% of traffic).
Uses ProtectAI's DeBERTa-v3-base-prompt-injection-v2 (184M parameters) —
the full-size model that trades latency for accuracy.

Also applies a secondary OR-ensemble heuristic: if Stage 1 matched ANY
patterns and the base model scores > 0.4, we vote INJECTION.
This catches adversarially-crafted prompts designed to sit in the
classifier's uncertain band.

Runs on CPU via standard PyTorch / HuggingFace pipeline.
"""

import logging
import time

from utils.config import CFG
from utils.result import DetectionResult, Verdict

logger = logging.getLogger(__name__)


class _DeepScanSingleton:
    """Lazily loads the heavy DeBERTa-v3-base model."""

    def __init__(self):
        self._pipe = None
        self._model_name: str = CFG.deep_model

    def _load(self) -> None:
        logger.info("Loading Stage 3 deep-scan model: %s ...", self._model_name)
        t0 = time.perf_counter()
        self._pipe = self._load_pytorch()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Stage 3 deep model ready in %.0fms", elapsed)

    def _load_pytorch(self):
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model=self._model_name,
            device=-1,
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> tuple[str, float]:
        if self._pipe is None:
            self._load()
        output = self._pipe(text)[0]
        return output["label"], output["score"]


_deep_scanner = _DeepScanSingleton()


def preload() -> None:
    """Load the deep model at startup (optional — saves first-call latency)."""
    _deep_scanner._load()


def check(
    text: str,
    stage1_patterns: list[str] | None = None,
    stage2_score: float | None = None,
) -> DetectionResult:
    """
    Deep scan with the full DeBERTa-v3-base model.

    Parameters
    ----------
    text            : preprocessed input text
    stage1_patterns : patterns already matched in Stage 1 (used in OR-ensemble)
    stage2_score    : injection probability from Stage 2 (used in OR-ensemble)

    Returns
    -------
    DetectionResult with INJECTION or SAFE (never UNCERTAIN — final stage).
    """
    t0 = time.perf_counter()

    try:
        label, confidence = _deep_scanner.predict(text)
    except Exception as exc:
        logger.error("Stage 3 deep scan error: %s", exc)
        # Fail safe — if model crashes, flag as UNCERTAIN but escalate cannot happen;
        # return a low-confidence SAFE to avoid blocking all traffic on model errors.
        return DetectionResult(
            verdict=Verdict.SAFE,
            score=0.5,
            stage=3,
            reason=f"Deep scan error: {exc} — defaulting to SAFE (fail-open)",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    latency = (time.perf_counter() - t0) * 1000

    # Normalise label → injection probability
    label_upper = label.upper()
    if "INJECTION" in label_upper or "UNSAFE" in label_upper or label_upper == "1":
        injection_prob = confidence
    else:
        injection_prob = 1.0 - confidence

    raw_scores = {
        "label": label,
        "confidence": confidence,
        "injection_prob": injection_prob,
        "stage2_score": stage2_score,
    }

    # ── OR-ensemble: boost score if Stage 1 already found patterns ────────────
    # If Stage 1 matched ANY patterns AND this model is also uncertain (>0.4),
    # we push over the threshold. This catches adversarial "uncertain band" attacks.
    has_stage1_signal  = bool(stage1_patterns)
    ensemble_triggered = False

    if has_stage1_signal and injection_prob > 0.40:
        injection_prob     = min(injection_prob + 0.20, 1.0)
        ensemble_triggered = True
        logger.debug(
            "OR-ensemble boost applied (Stage 1 patterns: %s)",
            stage1_patterns,
        )

    if injection_prob >= CFG.deep_inject_threshold:
        reason = f"Deep scan: injection probability {injection_prob:.3f}"
        if ensemble_triggered:
            reason += f" (OR-ensemble with Stage 1 patterns: {stage1_patterns})"
        return DetectionResult(
            verdict=Verdict.INJECTION,
            score=injection_prob,
            stage=3,
            reason=reason,
            latency_ms=latency,
            matched_patterns=stage1_patterns or [],
            raw_scores=raw_scores,
        )

    return DetectionResult(
        verdict=Verdict.SAFE,
        score=injection_prob,
        stage=3,
        reason=f"Deep scan: safe (injection probability {injection_prob:.3f})",
        latency_ms=latency,
        matched_patterns=stage1_patterns or [],
        raw_scores=raw_scores,
    )
