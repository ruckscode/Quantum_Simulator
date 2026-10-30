# QMutBench improvement campaign progress ? 2026-10-27

## State

Current champion remains mutation-aware Extra Trees (23 features), five-fold grouped development macro F1 0.7933. Its retrospective frozen-test comparison is 0.8184 vs 0.7884 for the saved full-development 8-feature control; the test has prior reuse and was not used to select this campaign's candidates.

## New experiment: additive circuit structure

The causal hypothesis and eight structure features are recorded in `run_manifest.json`. Fold 1 screen: champion 0.7779, add-on 0.7964, +1.85 pp, so a clean confirmation proceeded. Confirmation training excluded every fold-1 origin from both fit and validation, then trained a matched champion and candidate on identical records for each held fold.

- Fold 2: delta ?0.23 pp.
- Fold 3: delta +0.06 pp.
- Fold 4: delta ?0.22 pp.
- Pooled folds 2?4: candidate macro F1 0.7864, matched champion 0.7878, delta ?0.13 pp; paired origin-bootstrap 95% interval [?0.55,+0.27] pp.
- Fold 5 did not produce predictions. Available memory fell to 458 MiB during the fit, below the 500 MiB stop floor; the active experiment was interrupted. Checkpoints for folds 2?4 were preserved.

**Decision:** do not promote this add-on. Three clean confirmation folds do not show a material gain, and the partial interval includes zero. Do not resume the expensive fifth fit unless there is a new scientific reason and substantially safer resources.

## Hypothesis queue

| Hypothesis | Status | Rationale / next step |
|---|---|---|
| Operation-specialized classifiers | Deferred | Could specialize Add/Replace/Remove error patterns, but three models per grouped fold are expensive and Remove is small. Reconsider only for a resource-safe grouped screen. |
| More native circuit structure | Rejected for now | Add-on above did not improve on three clean folds; historical structure-only screen also lagged. |
| ET parameter / class-weight changes | Already screened | Existing tuning, class-weight, threshold, and nested results show no untouched strong option. Avoid repetition. |
| Gradient boosted trees | Screened / dependency constrained | HGB was weaker; CatBoost, XGBoost, LightGBM absent. Avoid optional installation without stronger evidence. |
| QML | No further run | Existing VQC, QSVC, re-uploading, reservoir results did not beat matched controls; previous expansion was memory-limited. |

No new test evaluation was performed. The frozen test remains reserved from current candidate selection. Current ranking and machine-readable trial ledger are in `leaderboard.json` and `run_manifest.json`.
