"""
pipeline.py
-----------
The Cascade Orchestrator

Wires Stages 0 → 3 together with early-exit logic.
Each stage can short-circuit the pipeline by returning a definitive verdict.
Only UNCERTAIN results proceed to the next stage.

Flow:
  preprocessor → Stage 0 (cache) → Stage 1 (heuristics)
               → Stage 2 (ML small) → Stage 3 (ML large)

The pipeline also updates the Stage 0 cache after every final verdict
so future identical inputs are served instantly.
"""

import logging
import time
from dataclasses import dataclass

import stage0_cache      as s0
import stage1_heuristics as s1
import stage2_classifier as s2
import stage3_deep_scan  as s3
from preprocessor import preprocess
from utils.result import DetectionResult, Verdict

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full audit trail from a single pipeline run."""
    final:          DetectionResult
    preprocessed:   str
    transformations: list[str]
    stage_results:  list[DetectionResult]
    total_latency_ms: float
    preprocess_latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  VERDICT : {self.final.verdict.value}",
            f"  Score   : {self.final.score:.3f}",
            f"  Stage   : {self.final.stage}",
            f"  Latency : {self.total_latency_ms:.1f}ms total",
            f"  Reason  : {self.final.reason}",
        ]
        if self.final.matched_patterns:
            lines.append(f"  Patterns: {', '.join(self.final.matched_patterns[:5])}")
        if self.transformations:
            lines.append(f"  PreProc : {', '.join(self.transformations)}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run(raw_text: str) -> PipelineResult:
    """
    Run a single prompt through the full detection cascade.

    Parameters
    ----------
    raw_text : str
        The raw, unprocessed user input.

    Returns
    -------
    PipelineResult containing the final verdict and full audit trail.
    """
    pipeline_start = time.perf_counter()
    stage_results: list[DetectionResult] = []

    # ── Preprocessing ─────────────────────────────────────────────────────────
    t_pre = time.perf_counter()
    cleaned, transformations = preprocess(raw_text)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    if transformations:
        logger.debug("Preprocessing applied: %s", ", ".join(transformations))

    # ── Stage 0: Hash Cache ───────────────────────────────────────────────────
    s0_result = s0.check(cleaned)
    if s0_result is not None:
        stage_results.append(s0_result)
        logger.debug("Stage 0 cache hit → %s", s0_result.verdict.value)
        return _build_result(
            final=s0_result,
            preprocessed=cleaned,
            transformations=transformations,
            stage_results=stage_results,
            pipeline_start=pipeline_start,
            preprocess_latency_ms=preprocess_ms,
        )

    # ── Stage 1: Heuristics ───────────────────────────────────────────────────
    s1_result = s1.check(cleaned)
    stage_results.append(s1_result)
    logger.debug(
        "Stage 1 → %s (score=%.3f, patterns=%s)",
        s1_result.verdict.value, s1_result.score, s1_result.matched_patterns
    )

    if s1_result.verdict != Verdict.UNCERTAIN:
        s0.update(cleaned, s1_result)
        return _build_result(
            final=s1_result,
            preprocessed=cleaned,
            transformations=transformations,
            stage_results=stage_results,
            pipeline_start=pipeline_start,
            preprocess_latency_ms=preprocess_ms,
        )

    # ── Stage 2: ML Classifier (DeBERTa-v3-small) ────────────────────────────
    s2_result = s2.check(cleaned)
    stage_results.append(s2_result)
    logger.debug(
        "Stage 2 → %s (score=%.3f)", s2_result.verdict.value, s2_result.score
    )

    if s2_result.verdict != Verdict.UNCERTAIN:
        s0.update(cleaned, s2_result)
        return _build_result(
            final=s2_result,
            preprocessed=cleaned,
            transformations=transformations,
            stage_results=stage_results,
            pipeline_start=pipeline_start,
            preprocess_latency_ms=preprocess_ms,
        )

    # ── Stage 3: Deep Scan (DeBERTa-v3-base) ─────────────────────────────────
    logger.debug("Stage 2 uncertain — escalating to Stage 3 deep scan.")
    s3_result = s3.check(
        text=cleaned,
        stage1_patterns=s1_result.matched_patterns,
        stage2_score=s2_result.score,
    )
    stage_results.append(s3_result)
    logger.debug(
        "Stage 3 → %s (score=%.3f)", s3_result.verdict.value, s3_result.score
    )

    s0.update(cleaned, s3_result)
    return _build_result(
        final=s3_result,
        preprocessed=cleaned,
        transformations=transformations,
        stage_results=stage_results,
        pipeline_start=pipeline_start,
        preprocess_latency_ms=preprocess_ms,
    )


def _build_result(
    final: DetectionResult,
    preprocessed: str,
    transformations: list[str],
    stage_results: list[DetectionResult],
    pipeline_start: float,
    preprocess_latency_ms: float = 0.0,
) -> PipelineResult:
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    return PipelineResult(
        final=final,
        preprocessed=preprocessed,
        transformations=transformations,
        stage_results=stage_results,
        total_latency_ms=total_ms,
        preprocess_latency_ms=preprocess_latency_ms,
    )
