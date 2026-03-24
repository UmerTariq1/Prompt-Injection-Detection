"""
config.py
---------
Single source of truth for all tunable parameters.
Change thresholds here — nowhere else.
"""

from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:

    # ── Stage 0: Hash Cache ──────────────────────────────────────────────────
    cache_max_size:        int   = 10_000   # max LRU entries (in-memory)
    cache_ttl_seconds:     int   = 3_600    # 1 hour

    # ── Stage 1: Heuristics ──────────────────────────────────────────────────
    # If heuristic score >= this, immediately flag as INJECTION (skip ML)
    heuristic_inject_threshold:  float = 0.80
    # If heuristic score <= this, immediately flag as SAFE (skip ML).
    # Use <= 0 to disable this fast-path (low scores go to Stage 2 instead).
    heuristic_safe_threshold:    float = 0.00

    # ── Stage 2: ML Classifier (DeBERTa-v3-small) ───────────────────────────
    classifier_model:      str   = "protectai/deberta-v3-small-prompt-injection-v2"
    classifier_max_length: int   = 256
    # Score >= this → INJECTION verdict from Stage 2
    classifier_inject_threshold:  float = 0.80
    # Score <= this → SAFE verdict from Stage 2
    classifier_safe_threshold:    float = 0.00
    # Between the two thresholds → UNCERTAIN → escalate to Stage 3

    # ── Stage 3: Deep Scan (DeBERTa-v3-base, heavier) ───────────────────────
    deep_model:            str   = "protectai/deberta-v3-base-prompt-injection-v2"
    deep_inject_threshold: float = 0.55   # lower bar — last line of defence

    # ── CSV Input ────────────────────────────────────────────────────────────
    # Primary text column; if missing, process_csv tries "text" then "prompts".
    csv_prompt_column:     str   = "text"
    # Optional: 0 = benign (SAFE), 1 = injection — used for accuracy / confusion matrix.
    csv_label_column:      str   = "label"
    # Appended to batch CSV: "0" = benign (SAFE), "1" = injection; renamed if column name collides.
    csv_prediction_column: str   = "prediction"
    # If None: write *_predictions.csv and *_report.md in the **same directory as the input CSV**.
    # If set (e.g. "data/output"): write under that path relative to repo root instead.
    csv_output_dir:        str | None = None


CFG = Config()


def csv_batch_output_dir(input_csv: Path) -> Path:
    """Directory for batch predictions CSV and Markdown report."""
    if CFG.csv_output_dir is not None:
        d = _REPO_ROOT / CFG.csv_output_dir
        d.mkdir(parents=True, exist_ok=True)
        return d
    return input_csv.resolve().parent
