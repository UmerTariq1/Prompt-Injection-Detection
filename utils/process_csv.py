"""Run the detection pipeline over a CSV file with optional ground-truth labels."""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

import pipeline
from .batch_report import write_batch_report_markdown
from .config import CFG, csv_batch_output_dir
from .csv_helpers import parse_ground_truth, resolve_prompt_column
from .result import Verdict

logger = logging.getLogger(__name__)


def _prediction_column_name(fieldnames: list[str]) -> str:
    name = CFG.csv_prediction_column
    if name in fieldnames:
        return "guard_prediction"
    return name


def _verdict_to_csv_label(verdict: Verdict) -> str:
    return "1" if verdict == Verdict.INJECTION else "0"


def _resolve_predictions_path(csv_in: Path, user_path: Path | None) -> Path:
    base = csv_batch_output_dir(csv_in)
    if user_path is None:
        return base / f"{csv_in.stem}_predictions.csv"
    if user_path.is_absolute():
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return user_path
    return base / user_path


def run_csv_batch(csv_path: Path, predictions_path_user: Path | None) -> None:
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    predictions_path = _resolve_predictions_path(csv_path, predictions_path_user)
    report_path = csv_batch_output_dir(csv_path) / f"{csv_path.stem}_report.md"

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        prompt_col = resolve_prompt_column(fieldnames)
        if prompt_col is None:
            logger.error(
                "No text column found. Tried %r, 'text', 'prompts'. Available: %s",
                CFG.csv_prompt_column,
                fieldnames,
            )
            sys.exit(1)
        label_col = CFG.csv_label_column if CFG.csv_label_column in fieldnames else None
        rows = list(reader)

    pred_col = _prediction_column_name(fieldnames)
    n_rows_file = len(rows)
    n_to_run = sum(1 for r in rows if (r.get(prompt_col) or "").strip())
    n_empty_skipped = n_rows_file - n_to_run

    logger.info(
        "Processing %d rows (%d non-empty prompts) from %s; text column=%r%s",
        n_rows_file,
        n_to_run,
        csv_path,
        prompt_col,
        f"; labels={label_col!r}" if label_col else "",
    )
    logger.info("Predictions CSV → %s", predictions_path)
    logger.info("Report          → %s", report_path)

    results_data: list[dict[str, str]] = []
    injection_count = 0
    safe_count = 0
    stage_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    latencies: list[float] = []

    tp = fp = tn = fn = 0
    n_bad_labels = 0
    ok = err = 0

    t_batch_start = time.perf_counter()

    bar = tqdm(
        total=n_to_run,
        desc="Pipeline",
        unit="prompt",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.2,
    )
    if label_col:
        bar.set_postfix(ok=0, err=0)

    out_fieldnames = list(fieldnames) + [pred_col]

    try:
        for row in rows:
            row_out: dict[str, str] = {
                k: "" if row.get(k) is None else str(row.get(k, "")) for k in fieldnames
            }
            prompt = (row.get(prompt_col) or "").strip()
            if not prompt:
                row_out[pred_col] = ""
                results_data.append(row_out)
                continue

            pr = pipeline.run(prompt)
            row_out[pred_col] = _verdict_to_csv_label(pr.final.verdict)

            if pr.final.verdict == Verdict.INJECTION:
                injection_count += 1
            else:
                safe_count += 1

            stage_counts[pr.final.stage] = stage_counts.get(pr.final.stage, 0) + 1
            latencies.append(pr.total_latency_ms)

            if label_col:
                truth = parse_ground_truth(row.get(label_col))
                if truth is None:
                    n_bad_labels += 1
                else:
                    pred_inj = pr.final.verdict == Verdict.INJECTION
                    truth_inj = truth == 1
                    if pred_inj and truth_inj:
                        tp += 1
                    elif pred_inj and not truth_inj:
                        fp += 1
                    elif not pred_inj and truth_inj:
                        fn += 1
                    else:
                        tn += 1
                    if pred_inj == truth_inj:
                        ok += 1
                    else:
                        err += 1
                    bar.set_postfix(ok=ok, err=err)

            results_data.append(row_out)
            bar.update(1)
    finally:
        bar.close()

    batch_ms = (time.perf_counter() - t_batch_start) * 1000
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    write_batch_report_markdown(
        report_path,
        input_csv=csv_path.resolve(),
        predictions_csv=predictions_path.resolve(),
        prompt_column=prompt_col,
        label_column=label_col,
        pred_column=pred_col,
        n_rows_file=n_rows_file,
        n_prompts_run=len(latencies),
        n_empty_skipped=n_empty_skipped,
        n_bad_labels=n_bad_labels,
        injection_count=injection_count,
        safe_count=safe_count,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        stage_counts=stage_counts,
        avg_lat_ms=avg_lat,
        p95_lat_ms=p95_lat,
        batch_ms=batch_ms,
    )

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_data)

    logger.info(
        "Finished: %d rows, column %r uses 0=benign, 1=injection (empty=no run).",
        len(results_data),
        pred_col,
    )
    print(f"\nPredictions CSV: {predictions_path}")
    print(f"Report (Markdown): {report_path}")
