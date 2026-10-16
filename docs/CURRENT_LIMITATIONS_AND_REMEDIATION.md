# Current Limitations and Remediation Status

## Verified software status

The most recent user-run Colab inspection (commit `3c45851`) reported 278 passed and 17 skipped tests, a passing offline acceptance test, and a successful controlled simulator-generated ML experiment. Those results do not establish official benchmark accuracy or physical IBM hardware validation.

Additional remediation files have now been added on `fix/statistics-module-collision`:

- `experiments/robust_ml_evaluation.py`: leave-one-circuit-family-out evaluation plus feature ablation (including a run without TVD).
- `experiments/benchmark_label_audit.py`: streaming schema/label audit for large QMutBench CSVs and a file-count audit for extracted Bugs4Q data.
- `ibm_validation.py`: live CLI execution now passes a configured token and optional instance from environment variables into the adapter; token values are not written to reports.
- `notebooks/Real_Data_ML_Evaluation.ipynb`: includes robust ML evaluation and optional benchmark label audit instructions.

These new changes have been committed but must still be executed and tested in Colab. Do not claim they passed until their outputs are reviewed.

## Three main research blockers

### 1. ML validity and generalization

The original controlled experiment produced 80.56% Random Forest accuracy on a single grouped holdout containing only three held-out circuit families. This is a simulator-generated result, not benchmark accuracy. Feature leakage has been identified as a risk in earlier experiments, and TVD is a strong anomaly signal but cannot by itself identify fault cause.

Remediation:
1. Run `python experiments/robust_ml_evaluation.py --repetitions 12 --shots 128 --output-dir reports/robust_ml_evaluation`.
2. Review leave-one-family-out out-of-fold metrics and per-class reports.
3. Compare all features against ablations without TVD and distribution-only/summary-only features.
4. Inspect the generator and features for direct or indirect label construction. Do not choose a model solely by the highest score.
5. Keep controlled-simulation results explicitly labelled as such.

### 2. Benchmark ground truth

QMutBench is a quantum-circuit mutant benchmark. Its mutation operator, mutant survival, or kill-related outcomes are valid only for the tasks those fields define; they do not directly establish the correct hardware-versus-program diagnosis category. Bugs4Q supplies real buggy/fixed Qiskit programs and bug metadata, but its bug-type metadata is not automatically a diagnosis-engine label.

Remediation:
1. Obtain the official public data and preserve provenance/attribution.
2. Run `experiments/benchmark_label_audit.py` against the full CSV and extracted Bugs4Q tree.
3. Report schema, missingness, class counts, unsupported records, and task-specific eligibility.
4. If using QMutBench for mutation-killing/survival prediction, define it as a separate benchmark task with its own leakage-safe features and grouped split.
5. Only compute hardware-versus-program diagnosis metrics for cases with an independent, defensible diagnosis-category reference. If none exist, report that limitation rather than manufacture labels.

### 3. IBM hardware validation

The previous Colab log reported IBM access blocked and no physical hardware execution. The user now reports that their IBM account/token are working, but no live access check has yet been run against the latest code. No hardware job has been authorized.

Remediation:
1. Install compatible optional dependencies in Colab: `qiskit` and `qiskit-ibm-runtime`.
2. Load the token from the Colab secret named `QISKIT_IBM_TOKEN` without printing it; optionally set `QISKIT_IBM_INSTANCE` if the account requires an instance.
3. Run backend discovery only with `python ibm_validation.py --list-backends --output reports/ibm_access_check.json`. This should query backend availability but does not submit a circuit job.
4. Review discovery status, accessible backend names, and any account/instance error. Do not run live execution until the user separately authorizes it.
5. After authorization, execute a small, controlled circuit, save backend/calibration metadata and job ID, compare counts to an ideal and finite-shot baseline, and clearly distinguish observed hardware data from simulator fixtures.

## Acceptance criteria

- ML: family-held-out metrics are reproducible and feature-ablation behavior is reported; no accuracy claim is made beyond the evaluated data source.
- Benchmark: dataset/task/label definitions are explicit; ineligible records are counted; no mutation metadata is silently converted to diagnosis truth.
- IBM: a non-job backend discovery succeeds, followed only after separate user authorization by a real job with traceable result/calibration evidence.
- Tests: run the full test suite after the latest commits. Earlier test success does not automatically cover these new changes.

## Public dataset references

- QMutBench repository: https://github.com/EnautMendi/QMutBench
- Bugs4Q official dataset/framework record: https://zenodo.org/records/8148982
