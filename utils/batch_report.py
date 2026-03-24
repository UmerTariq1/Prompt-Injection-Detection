"""Write a Markdown batch / validation report (tables, metrics) to disk."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def write_batch_report_markdown(
    path: Path,
    *,
    input_csv: Path,
    predictions_csv: Path,
    prompt_column: str,
    label_column: str | None,
    pred_column: str,
    n_rows_file: int,
    n_prompts_run: int,
    n_empty_skipped: int,
    n_bad_labels: int,
    injection_count: int,
    safe_count: int,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    stage_counts: dict[int, int],
    avg_lat_ms: float,
    p95_lat_ms: float,
    batch_ms: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluated = tp + fp + tn + fn
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# Prompt injection guard — batch report",
        "",
        f"**Generated:** {now}  ",
        "",
        "## Files",
        "",
        "| Role | Path |",
        "|------|------|",
        f"| Input CSV | `{input_csv.as_posix()}` |",
        f"| Predictions CSV | `{predictions_csv.as_posix()}` |",
        "",
        "## Column mapping",
        "",
        f"- **Text column:** `{prompt_column}`",
        f"- **Label column (optional):** `{label_column or '—'}`",
        f"- **Prediction column added:** `{pred_column}` — values **`0`** = benign (SAFE), **`1`** = injection",
        "",
        "## Dataset summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Rows in file | {n_rows_file:,} |",
        f"| Prompts run through pipeline | {n_prompts_run:,} |",
        f"| Empty text rows (no prediction) | {n_empty_skipped:,} |",
    ]
    if label_column:
        lines.append(f"| Invalid / missing labels (excluded from accuracy) | {n_bad_labels:,} |")
    lines.extend(["", "## Model outputs (counts)", "", "| Verdict | Count |", "|---------|------:|"])
    lines.append(f"| `0` — SAFE (benign) | {safe_count:,} |")
    lines.append(f"| `1` — INJECTION | {injection_count:,} |")
    lines.append("")

    if label_column and evaluated > 0:
        acc = (tp + tn) / evaluated
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        lines.extend(
            [
                "## Validation vs labels",
                "",
                f"_Compared **{evaluated:,}** rows with valid labels (ground truth `0` / `1`)._",
                "",
                "| Metric | Value |",
                "|--------|------:|",
                f"| **Accuracy** | {acc * 100:.2f}% |",
                f"| Precision (class 1 — injection) | {prec:.4f} |",
                f"| Recall (class 1 — injection) | {rec:.4f} |",
                f"| F1 (class 1 — injection) | {f1:.4f} |",
                f"| Specificity (class 0 — benign) | {spec:.4f} |",
                "",
                "### Confusion matrix",
                "",
                "Rows = actual label, columns = predicted value in output CSV.",
                "",
                "|  | Predicted `0` (safe) | Predicted `1` (injection) |",
                "|--|---------------------:|--------------------------:|",
                f"| **Actual `0` (benign)** | {tn:,} (TN) | {fp:,} (FP) |",
                f"| **Actual `1` (injection)** | {fn:,} (FN) | {tp:,} (TP) |",
                "",
                "| Cell | Meaning |",
                "|------|---------|",
                "| TN | True negative — benign correctly marked `0` |",
                "| TP | True positive — attack correctly marked `1` |",
                "| FP | False positive — benign wrongly marked `1` |",
                "| FN | False negative — attack wrongly marked `0` |",
                "",
            ]
        )
    elif label_column:
        lines.extend(
            [
                "## Validation vs labels",
                "",
                "_No rows had both a valid label and a pipeline prediction; confusion matrix skipped._",
                "",
            ]
        )

    lines.extend(
        [
            "## Where the pipeline stopped (stage counts)",
            "",
            "| Stage | Role | Count |",
            "|-------|------|------:|",
            f"| 0 | Cache hit | {stage_counts.get(0, 0):,} |",
            f"| 1 | Heuristics | {stage_counts.get(1, 0):,} |",
            f"| 2 | Small ML model | {stage_counts.get(2, 0):,} |",
            f"| 3 | Deep scan | {stage_counts.get(3, 0):,} |",
            "",
            "## Latency",
            "",
            "| Statistic | ms |",
            "|-----------|---:|",
            f"| Average per prompt | {avg_lat_ms:.1f} |",
            f"| p95 per prompt | {p95_lat_ms:.1f} |",
            f"| Total batch wall time | {batch_ms / 1000:.1f} s |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
