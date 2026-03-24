# Prompt injection guard — batch report

**Generated:** 2026-03-24 16:53:02 UTC  

## Files

| Role | Path |
|------|------|
| Input CSV | `C:/side projects/prompt_injection/assets/data/deepset_prompt-injections_test.csv` |
| Predictions CSV | `C:/side projects/prompt_injection/assets/data/deepset_prompt-injections_test_predictions.csv` |

## Column mapping

- **Text column:** `prompts`
- **Label column (optional):** `label`
- **Prediction column added:** `prediction` — values **`0`** = benign (SAFE), **`1`** = injection

## Dataset summary

| Metric | Value |
|--------|------:|
| Rows in file | 116 |
| Prompts run through pipeline | 116 |
| Empty text rows (no prediction) | 0 |
| Invalid / missing labels (excluded from accuracy) | 0 |

## Model outputs (counts)

| Verdict | Count |
|---------|------:|
| `0` — SAFE (benign) | 82 |
| `1` — INJECTION | 34 |

## Validation vs labels

_Compared **116** rows with valid labels (ground truth `0` / `1`)._

| Metric | Value |
|--------|------:|
| **Accuracy** | 74.14% |
| Precision (class 1 — injection) | 0.9412 |
| Recall (class 1 — injection) | 0.5333 |
| F1 (class 1 — injection) | 0.6809 |
| Specificity (class 0 — benign) | 0.9643 |

### Confusion matrix

Rows = actual label, columns = predicted value in output CSV.

|  | Predicted `0` (safe) | Predicted `1` (injection) |
|--|---------------------:|--------------------------:|
| **Actual `0` (benign)** | 54 (TN) | 2 (FP) |
| **Actual `1` (injection)** | 28 (FN) | 32 (TP) |

| Cell | Meaning |
|------|---------|
| TN | True negative — benign correctly marked `0` |
| TP | True positive — attack correctly marked `1` |
| FP | False positive — benign wrongly marked `1` |
| FN | False negative — attack wrongly marked `0` |

## Where the pipeline stopped (stage counts)

| Stage | Role | Count |
|-------|------|------:|
| 0 | Cache hit | 0 |
| 1 | Heuristics | 8 |
| 2 | Small ML model | 25 |
| 3 | Deep scan | 83 |

## Latency

| Statistic | ms |
|-----------|---:|
| Average per prompt | 147.0 |
| p95 per prompt | 243.8 |
| Total batch wall time | 17.1 s |
