# Evaluation explainer

## Evaluation questions

ShoeLens is evaluated against four questions:

1. Can it retrieve the correct catalog product after controlled image transformations?
2. Can its decision gate avoid claiming identity when evidence is weak or the item is absent?
3. Does it help users complete a product-and-attributes task faster and more successfully than manual browsing?
4. Are optional AI explanations source-supported, usable, and affordable?

The frozen calibration set is used to select decision thresholds. Reported product-search performance comes from the separate 60-query test set.

## Target metrics

| Area | Target |
|---|---|
| Accepted product matches | At least 85% precision during calibration |
| User task | At least 50% median paired time reduction, with incorrect answers penalized at 180 seconds |
| Grounded catalog summaries | Claims must match supported catalog fields |
| Uncertain-photo descriptions | Human review of visible fields and unsupported claims |

The calibrated identity gate requires both a combined score of at least **0.234958** and a top-one versus top-two gap of at least **0.040742**. The combined score is 70% geometric evidence and 30% CLIP cosine similarity. It is not a probability or confidence percentage.

## 1. Local retrieval and decision evaluation

Run:

```bash
.venv/bin/python benchmark.py --study artifacts/user-study.csv
```

The script evaluates CLIP plus SIFT and a deterministic non-AI baseline on the same frozen queries. The baseline uses a 64-bit difference hash and foreground colour histogram. Main outputs are:

- `artifacts/evaluation.json` — aggregate metrics and protocols
- `artifacts/predictions.json` — per-query retrieval and decision records
- `artifacts/failure-cases.json` — accepted errors and rejected known items
- `artifacts/calibration.json` — frozen score and margin thresholds
- `artifacts/calibration-curve.json` — threshold trade-offs

### Held-out results

| Metric | CLIP + SIFT | Handcrafted baseline |
|---|---:|---:|
| Known-item Top-1 accuracy | 41/42, 97.6% | 15/42, 35.7% |
| Answer coverage across all test queries | 42/60, 70.0% | 0/60, 0.0% |
| Accepted-answer accuracy | 41/42, 97.6% | Undefined: no answers accepted |
| Abstention rate | 30.0% | 100.0% |
| Unknown-item false acceptance | 0/18, 0.0% | 0/18, 0.0% |
| Mean local computation after startup | 0.0572 s | 0.0044 s |

The 95% Wilson interval for ShoeLens accepted-answer accuracy is approximately 87.7%–99.6%. The single wrong accepted identity is more important than the headline accuracy because it is a confident-looking product error. Zero false accepts among only 18 unknown/non-footwear test images is not enough to establish real-world safety.

The baseline's undefined accepted-answer accuracy must not be presented as 100%: it accepted nothing. Its behavior shows why coverage and accuracy must be reported together.

## 2. User-study evaluation

Recompute only the study analysis with:

```bash
.venv/bin/python user_study.py --analyze artifacts/user-study.csv
```

The study contains five participants and 20 complete paired tasks. A complete-task success requires the correct product, season, and usage.

| Metric | Manual browsing | ShoeLens |
|---|---:|---:|
| Complete-task success | 13/20, 65% | 17/20, 85% |
| Raw median completion time | 31.5 s | 15.5 s |

The median paired reduction with wrong answers counted as 180 seconds was **53.6%**. Among the 11 pairs where both methods were correct, the median reduction was **50.0%**. The prespecified 50% target was therefore reached in this study.

The result is bounded by the small sample, catalog-origin task images, and an asymmetric interface: manual participants browsed 20 preselected alternatives, while ShoeLens searched 2,000 products. It should be interpreted as evidence for these tasks, not a universal shopping-time claim.

## 3. Grounded catalog-summary evaluation

Refresh the report from existing saved results without paid calls:

```bash
.venv/bin/python evaluate_summary.py
```

To fill missing summaries through OpenRouter, configure `.env` and explicitly cap paid calls:

```bash
.venv/bin/python evaluate_summary.py --max-summaries 42
```

The complete accepted set contains 42 searches. All 42 model responses passed schema and source-field validation; 210/210 emitted catalog claims matched the retrieved records. End-to-end claim accuracy was **97.6%**, because claims based on the one wrongly retrieved identity count as wrong against query ground truth.

The recorded run made 37 billed calls and reused five cached summaries. The 37 live calls cost **US$0.0022863**. Four cached summaries retain US$0.0002481 in original recorded costs, while one older cached summary has unknown original cost. Mean retrieval plus summary time was 1.70 seconds.

These checks establish agreement with catalog fields, not writing quality or user comprehension. A perfectly grounded summary can still describe the wrong item if retrieval is wrong.

## 4. Uncertain-photo description evaluation

Refresh saved metrics and the review page without new API calls:

```bash
.venv/bin/python evaluate_visual.py
```

Generate at most nine missing descriptions with:

```bash
.venv/bin/python evaluate_visual.py --max-images 9
```

Outputs include `artifacts/visual-predictions.json`, `artifacts/visual-evaluation.json`, `artifacts/visual-human-review.html`, and `artifacts/visual-human-review.csv`.

All nine held-out uncertain-shoe images received a description. Original generation cost was **US$0.0119124**, mean remote API latency was 2.41 seconds, and mean retrieval-plus-description time was 2.47 seconds. The project owner reviewed 36 visible-field judgments and confirmed all 36, with no unsupported claims reported.

This is owner-attested review rather than independent blinded evaluation. The images are transformed catalog photos, so the result cannot be generalized to independently captured phone images.

## Test suite

Run:

```bash
.venv/bin/python -m unittest discover -s tests
cd frontend
node --test src/evaluationView.test.js
npm run build
```

The tests cover catalog parsing, deterministic subset selection, calibration logic, geometric scoring, retrieval decisions, user-study calculations, OpenRouter schema validation, image API behavior, and visual-evaluation aggregation. Tests verify implementation behavior; they do not replace dataset-level evaluation or human review.

## Main remaining evaluation gap

The next evaluation should use independently captured phone photographs across different angles, distances, backgrounds, and lighting. It should also include visually similar shoes that are absent from the catalog and report their false-confirmation rate. Thresholds should be retuned only on a new calibration subset and then frozen before the new test set is opened.
