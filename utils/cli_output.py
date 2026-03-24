"""
cli_output.py
-------------
Stdout formatting for the CLI: per-prompt trace and batch summary.
"""

from __future__ import annotations

import sys

import pipeline
from .result import Verdict

# ── ANSI colours (safe on non-TTY) ────────────────────────────────────────────

_COLOUR = {
    Verdict.INJECTION: "\033[91m",
    Verdict.SAFE: "\033[92m",
    Verdict.UNCERTAIN: "\033[93m",
}
_RESET = "\033[0m"


def _colour(verdict: Verdict, text: str) -> str:
    if sys.stdout.isatty():
        return f"{_COLOUR[verdict]}{text}{_RESET}"
    return text


# ── Stage labels (pipeline order) ─────────────────────────────────────────────

_STEP_PREPROCESS = "Preprocess"
_STAGE_NAMES: dict[int, str] = {
    0: "Stage 0 - Cache lookup",
    1: "Stage 1 - Heuristics (regex)",
    2: "Stage 2 - ML classifier (small)",
    3: "Stage 3 - Deep scan (large)",
}


def _print_pipeline_trace(pr: pipeline.PipelineResult, verbose: bool) -> None:
    print("  Steps:")
    pre_ms = pr.preprocess_latency_ms
    pre_note = ", ".join(pr.transformations) if pr.transformations else "(no changes)"
    print(f"    * {_STEP_PREPROCESS:28s}  {pre_ms:6.1f} ms  - {pre_note}")

    rs = pr.stage_results
    if not rs:
        return

    if rs[0].stage == 0:
        sr = rs[0]
        v = _colour(sr.verdict, sr.verdict.value)
        print(
            f"    * {_STAGE_NAMES[0]:28s}  {sr.latency_ms:6.1f} ms  |  "
            f"verdict={v:12s}  score={sr.score:.3f}  (replay from earlier run)"
        )
        for s in (1, 2, 3):
            print(
                f"    * {_STAGE_NAMES[s]:28s}  {'-':>6}      - "
                "SKIPPED (cache hit)"
            )
        return

    print(
        f"    * {_STAGE_NAMES[0]:28s}  {'-':>6}      - "
        "miss -> continue"
    )

    i = 0
    sr1 = rs[i]
    i += 1
    v1 = _colour(sr1.verdict, sr1.verdict.value)
    print(
        f"    * {_STAGE_NAMES[1]:28s}  {sr1.latency_ms:6.1f} ms  |  "
        f"verdict={v1:12s}  score={sr1.score:.3f}"
    )
    if verbose and sr1.matched_patterns:
        print(f"        patterns: {', '.join(sr1.matched_patterns[:8])}")
    if verbose and sr1.verdict == Verdict.UNCERTAIN:
        print(f"        reason: {sr1.reason}")
    if sr1.verdict != Verdict.UNCERTAIN:
        print(
            f"    * {_STAGE_NAMES[2]:28s}  {'-':>6}      - "
            "SKIPPED (Stage 1 decided)"
        )
        print(
            f"    * {_STAGE_NAMES[3]:28s}  {'-':>6}      - "
            "SKIPPED (Stage 1 decided)"
        )
        if verbose:
            print(f"        reason: {sr1.reason}")
        return

    if i >= len(rs):
        return
    sr2 = rs[i]
    i += 1
    v2 = _colour(sr2.verdict, sr2.verdict.value)
    print(
        f"    * {_STAGE_NAMES[2]:28s}  {sr2.latency_ms:6.1f} ms  |  "
        f"verdict={v2:12s}  score={sr2.score:.3f}"
    )
    if verbose:
        print(f"        reason: {sr2.reason}")
        if sr2.raw_scores:
            print(f"        raw: {sr2.raw_scores}")

    if sr2.verdict != Verdict.UNCERTAIN:
        print(
            f"    * {_STAGE_NAMES[3]:28s}  {'-':>6}      - "
            "SKIPPED (Stage 2 decided)"
        )
        return

    if i >= len(rs):
        return
    sr3 = rs[i]
    v3 = _colour(sr3.verdict, sr3.verdict.value)
    print(
        f"    * {_STAGE_NAMES[3]:28s}  {sr3.latency_ms:6.1f} ms  |  "
        f"verdict={v3:12s}  score={sr3.score:.3f}"
    )
    if verbose:
        print(f"        reason: {sr3.reason}")
        if sr3.raw_scores:
            print(f"        raw: {sr3.raw_scores}")


def print_result(prompt: str, pr: pipeline.PipelineResult, verbose: bool = False) -> None:
    """Print one pipeline run to stdout."""
    verdict_str = _colour(pr.final.verdict, pr.final.verdict.value.center(10))

    print(f"\n{'-' * 60}")
    print(f"  Prompt  : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Verdict : {verdict_str}   (final stage {pr.final.stage})")
    print(
        f"  Score   : {pr.final.score:.3f}   |   Total time: {pr.total_latency_ms:.1f} ms "
        f"(preprocess {pr.preprocess_latency_ms:.1f} ms)"
    )
    _print_pipeline_trace(pr, verbose=verbose)
    print(f"  Reason  : {pr.final.reason}")

    if pr.final.matched_patterns:
        print(f"  Patterns: {', '.join(pr.final.matched_patterns[:5])}")


def print_batch_summary(
    total: int,
    injection_count: int,
    safe_count: int,
    stage_counts: dict[int, int],
    avg_lat_ms: float,
    p95_lat_ms: float,
    batch_ms: float,
) -> None:
    """Print batch run statistics."""
    print(f"\n{'=' * 60}")
    print(f"  BATCH SUMMARY  ({total} prompts)")
    print(f"{'=' * 60}")
    inj_pct = injection_count / total * 100 if total else 0
    safe_pct = safe_count / total * 100 if total else 0
    print(
        f"  Injections : {_colour(Verdict.INJECTION, str(injection_count))}  ({inj_pct:.1f}%)"
    )
    print(
        f"  Safe       : {_colour(Verdict.SAFE, str(safe_count))}  ({safe_pct:.1f}%)"
    )
    print(
        f"  Stage hits : Stage0={stage_counts.get(0, 0)}  Stage1={stage_counts.get(1, 0)}  "
        f"Stage2={stage_counts.get(2, 0)}  Stage3={stage_counts.get(3, 0)}"
    )
    print(
        f"  Latency    : avg={avg_lat_ms:.1f}ms  p95={p95_lat_ms:.1f}ms  "
        f"total={batch_ms / 1000:.1f}s"
    )
    print(f"{'=' * 60}")


def print_validation_report(
    *,
    prompt_column: str,
    label_column: str | None,
    n_rows_file: int,
    n_prompts_run: int,
    n_empty_skipped: int,
    n_bad_labels: int,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    injection_pred: int,
    safe_pred: int,
    stage_counts: dict[int, int],
    avg_lat_ms: float,
    p95_lat_ms: float,
    batch_ms: float,
) -> None:
    """
    Rich summary after a labeled CSV run: confusion matrix, rates, latency.
    """
    evaluated = tp + fp + tn + fn
    w = 64
    line = lambda ch="-": ch * w

    def _fmt_pct(num: float) -> str:
        return f"{num * 100:6.2f}%"

    print(f"\n{line('=')}")
    print(f"  {'VALIDATION REPORT':^{w-4}}")
    print(f"{line('=')}")
    print(f"  Text column : {prompt_column}")
    print(f"  Label column: {label_column or '(none - metrics below are prediction counts only)'}")
    print(line())

    print(f"  Rows in file     : {n_rows_file:,}")
    print(f"  Prompts evaluated: {n_prompts_run:,}")
    if n_empty_skipped:
        print(f"  Empty skipped    : {n_empty_skipped:,}")
    if label_column and n_bad_labels:
        print(f"  Bad/missing label: {n_bad_labels:,} (excluded from accuracy)")

    print(line())
    print(f"  {'Pipeline predictions':^{w-4}}")
    print(f"  Injections : {_colour(Verdict.INJECTION, str(injection_pred))}")
    print(f"  Safe       : {_colour(Verdict.SAFE, str(safe_pred))}")
    print(line())

    if evaluated == 0:
        print("  No rows with valid labels - confusion matrix and accuracy omitted.")
        print(
            f"  Latency    : avg={avg_lat_ms:.1f}ms  p95={p95_lat_ms:.1f}ms  "
            f"total={batch_ms / 1000:.1f}s"
        )
        print(f"  Stage hits : Stage0={stage_counts.get(0, 0)}  Stage1={stage_counts.get(1, 0)}  "
              f"Stage2={stage_counts.get(2, 0)}  Stage3={stage_counts.get(3, 0)}")
        print(f"{line('=')}")
        return

    acc = (tp + tn) / evaluated
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    acc_disp = _colour(Verdict.SAFE, _fmt_pct(acc)) if acc >= 0.95 else _colour(
        Verdict.UNCERTAIN, _fmt_pct(acc)
    )
    if not sys.stdout.isatty():
        acc_disp = _fmt_pct(acc)

    print(f"  Accuracy ({evaluated:,} labeled rows) : {acc_disp}")
    print()
    print("  Confusion matrix  (label 0 = benign -> expect SAFE, 1 = injection -> expect INJECTION)")
    print()
    print(f"  {'':12}  {'Predicted SAFE':^18}  {'Predicted INJECTION':^20}")
    print(f"  {'Actual benign (0)':12}  {tn:^18}  {fp:^20}")
    print(f"  {'Actual attack (1)':12}  {fn:^18}  {tp:^20}")
    print()
    print(f"  True negatives (TN) : {tn:,}   |   False positives (FP): {fp:,}  (benign flagged as attack)")
    print(f"  False negatives (FN): {fn:,}   |   True positives (TP) : {tp:,}  (attacks caught)")
    print(line())
    print(f"  Precision (injection) : {prec:.4f}")
    print(f"  Recall    (injection) : {rec:.4f}")
    print(f"  F1 score  (injection) : {f1:.4f}")
    print(f"  Specificity (benign)  : {spec:.4f}")
    print(line())
    print(
        f"  Stage hits : Stage0={stage_counts.get(0, 0)}  Stage1={stage_counts.get(1, 0)}  "
        f"Stage2={stage_counts.get(2, 0)}  Stage3={stage_counts.get(3, 0)}"
    )
    print(
        f"  Latency    : avg={avg_lat_ms:.1f}ms  p95={p95_lat_ms:.1f}ms  "
        f"total={batch_ms / 1000:.1f}s"
    )
    print(f"{line('=')}")
