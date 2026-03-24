# Technical wiki

The pipeline runs **preprocessing → cache → regex rules → small ML model → big ML model** (later steps only run when the previous step is unsure). Every dial lives in `[utils/config.py](../utils/config.py)`. Scores are **0–1** where higher means “looks more like injection.”

---

## Detection flow


| Step                     | What it does                                                  | Score? Uses `utils/config.py`                                               |
| ------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Preprocessing**        | Normalises input (`preprocessor.py`).                         | **No** — only `transformations`.                                      |
| **Stage 0 — Cache**      | Hash lookup (`stage0_cache.py`); replays stored `score`.      | **Yes** — `cache_max_size`, `cache_ttl_seconds`.                      |
| **Stage 1 — Heuristics** | Weighted regex (`stage1_heuristics.py`), score capped at 1.0. | **Yes** — `heuristic_inject_threshold`, `heuristic_safe_threshold`.   |
| **Stage 2 — ML (small)** | HF `classifier_model`, `classifier_max_length`.               | **Yes** — `classifier_inject_threshold`, `classifier_safe_threshold`. |
| **Stage 3 — Deep scan**  | HF `deep_model`; OR-ensemble logic in `stage3_deep_scan.py`.  | **Yes** — `deep_inject_threshold`.                                    |

## Reading CLI output

Every run prints a **Steps:** block in order:

1. **Preprocess** — Time in ms and what changed (or “no changes”). No score.
2. **Stage 0** — Either **miss -> continue** or a **cache hit** with verdict, score, and Stages 1–3 marked **SKIPPED**.
3. **Stage 1** — Regex: verdict and score (`UNCERTAIN` means Stage 2 will run).
4. **Stage 2** — Small model: same. If it decides, Stage 3 is **SKIPPED**.
5. **Stage 3** — Large model: only if Stage 2 was uncertain.

**Final** verdict, score, and **Reason** are the decision you care about; `final stage N` matches the last step that actually decided. Use `--verbose` for regex pattern names, per-stage reasons, and ML `raw` dicts.

---

## Catch more real attacks (raise detection rate)

Do one or more of these; each makes the system **more sensitive**, so expect **more false alarms** and often **slower** runs (more work for the ML stages).


| Do this                                                                                      | In `utils/config.py`                          |
| -------------------------------------------------------------------------------------------- | --------------------------------------- |
| Call **INJECTION** on weaker regex matches                                                   | **Lower** `heuristic_inject_threshold`  |
| Stop sending tricky prompts out as **SAFE** too early                                        | **Raise** `heuristic_safe_threshold`    |
| Call **INJECTION** on weaker signals from the small model                                    | **Lower** `classifier_inject_threshold` |
| Send more borderline cases to the heavy model instead of **SAFE**                            | **Raise** `classifier_safe_threshold`   |
| Call **INJECTION** after the heavy model on weaker signals                                   | **Lower** `deep_inject_threshold`       |


If attacks use **new wording** the regex never sees, add or tighten patterns in `stage1_heuristics.py`. If text is **hidden** by encoding or weird characters, extend `preprocessor.py`.

Run `python main.py --prompt "..." --verbose` to see **which stage** decided and the **scores** so you know which row in the table to touch first.

---

## Fewer false “injection” flags

Reverse the table above: **raise** inject thresholds, **lower** `heuristic_safe_threshold` and `classifier_safe_threshold` cautiously. You trade away some detection rate.

---

## Reference: `utils/config.py` fields


| Field                                                       | Role                                                            |
| ----------------------------------------------------------- | --------------------------------------------------------------- |
| `heuristic_inject_threshold` / `heuristic_safe_threshold`   | Stage 1 (regex) — above / below → injection / safe              |
| `classifier_inject_threshold` / `classifier_safe_threshold` | Stage 2 (small model) — between them → uncertain, go to Stage 3 |
| `deep_inject_threshold`                                     | Stage 3 (big model) — final injection cutoff                    |
| `classifier_model`, `deep_model`, `classifier_max_length`   | Which HF models and token limit                                 |
| `cache_max_size`, `cache_ttl_seconds`                       | Cache size / TTL (if used in `stage0_cache.py`)                 |
| `csv_prompt_column` / `csv_label_column` / `csv_prediction_column` | Batch input text / optional label / output prediction column name |
| `csv_output_dir` | `None` (default): write `*_predictions.csv` and `*_report.md` next to the input CSV; or e.g. `"data/output"` under repo root |


