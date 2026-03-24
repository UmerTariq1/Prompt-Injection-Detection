# Prompt injection guard — batch report

**Generated:** 2026-03-24 15:44:27 UTC  

## Files


| Role            | Path                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------ |
| Input CSV       | `C:/side projects/prompt_injection/assets/data/safe-guard-prompt-injection-test.csv`             |
| Predictions CSV | `C:/side projects/prompt_injection/assets/data/safe-guard-prompt-injection-test_predictions.csv` |


## Column mapping

- **Text column:** `text`
- **Label column (optional):** `label`
- **Prediction column added:** `prediction` — values `**0`** = benign (SAFE), `**1**` = injection

## Dataset summary


| Metric                                            | Value |
| ------------------------------------------------- | ----- |
| Rows in file                                      | 2,060 |
| Prompts run through pipeline                      | 2,060 |
| Empty text rows (no prediction)                   | 0     |
| Invalid / missing labels (excluded from accuracy) | 0     |


## Model outputs (counts)


| Verdict             | Count |
| ------------------- | ----- |
| `0` — SAFE (benign) | 1,530 |
| `1` — INJECTION     | 530   |


## Validation vs labels

*Compared **2,060** rows with valid labels (ground truth `0` / `1`).*


| Metric                          | Value  |
| ------------------------------- | ------ |
| **Accuracy**                    | 91.46% |
| Precision (class 1 — injection) | 0.9472 |
| Recall (class 1 — injection)    | 0.7723 |
| F1 (class 1 — injection)        | 0.8508 |
| Specificity (class 0 — benign)  | 0.9801 |


### Confusion matrix

Rows = actual label, columns = predicted value in output CSV.


|                            | Predicted `0` (safe) | Predicted `1` (injection) |
| -------------------------- | -------------------- | ------------------------- |
| **Actual `0` (benign)**    | 1,382 (TN)           | 28 (FP)                   |
| **Actual `1` (injection)** | 148 (FN)             | 502 (TP)                  |



| Cell | Meaning                                     |
| ---- | ------------------------------------------- |
| TN   | True negative — benign correctly marked `0` |
| TP   | True positive — attack correctly marked `1` |
| FP   | False positive — benign wrongly marked `1`  |
| FN   | False negative — attack wrongly marked `0`  |


## Where the pipeline stopped (stage counts)


| Stage | Role           | Count |
| ----- | -------------- | ----- |
| 0     | Cache hit      | 11    |
| 1     | Heuristics     | 164   |
| 2     | Small ML model | 1,808 |
| 3     | Deep scan      | 77    |


## Latency


| Statistic             | ms      |
| --------------------- | ------- |
| Average per prompt    | 59.7    |
| p95 per prompt        | 124.3   |
| Total batch wall time | 124.0 s |


