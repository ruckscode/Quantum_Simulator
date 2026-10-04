# Quantum Diagnostic System

## Problem

This project detects execution anomalies in quantum circuits and combines operation-level program localization with hardware-calibration evidence. Diagnosis results retain their supporting evidence and avoid assigning a cause when the available evidence is insufficient.

## Architecture

Quantum program → tuple-based universal circuit representation → statevector ideal probabilities and observed distribution → statistical comparison → program-fault localization → hardware-fault localization → diagnosis → validation/evidence report.

The Phase K demo uses deterministic offline cases. It does not model hardware noise or claim that controlled observations came from a device.

## Main Modules

- `gates.py`: gate registry and matrices.
- `quantum_simulator.py`: statevector execution, probabilities, measurements, and shots.
- `hardware_model.py`: backend-independent qubit, gate, and connectivity calibration data.
- `statistics.py`: distribution/count comparison and finite-shot guard.
- `fault_localizer.py` and `hardware_fault_localizer.py`: program and hardware candidates.
- `diagnosis_engine.py`: evidence-preserving integrated diagnosis.
- `validation_framework.py`: deterministic controlled scenario validation.
- `bugs4q_validation.py`: Bugs4Q data loading and case-level reporting.
- `ibm_validation.py`: optional IBM adapter with separate offline and live paths.
- `demo.py`: offline end-to-end demonstration.

## Setup

Python 3.12 is used for the validated workspace. Create an environment and install the core runtime/test dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pytest
```

The offline demo and core test suite do not require Qiskit. IBM validation additionally requires compatible `qiskit` and `qiskit-ibm-runtime` packages, network access, and an authorized IBM Quantum account.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Offline Demo

```powershell
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m demo
.\.venv\Scripts\python.exe demo.py --json
```

The demo writes complete diagnosis evidence to `reports/final_demo_report.json`. `--json` also prints machine-readable output. All observed distributions in its controlled fault cases are explicitly identified as deterministic fixtures.

## Reproducible Experiments

The checked-in configurations under `experiments/configs/` are labeled `SYNTHETIC_VALIDATION` and use fixed random seeds. Run one experiment or the full set with:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiment experiments/configs/healthy.json
.\.venv\Scripts\python.exe -m experiments.run_experiment --all --tests-passed 96 --tests-failed 0
```

Individual results are written under `reports/experiments/`; the repeated-run fingerprints and environment/test summary are written to `reports/reproducibility_report.json`. `observed_counts`, when specified, are controlled local fixtures. The runner records but does not apply `noise_parameters`; the project currently has no noise model.

## IBM Validation Requirements

Use `python -m ibm_validation --offline` to exercise local adapter logic without IBM access. `python -m ibm_validation --list-backends` performs discovery only if Qiskit Runtime and credentials are available. Real execution requires a selected backend and a JSON circuit file, for example:

```powershell
python -m ibm_validation --backend YOUR_BACKEND --circuit circuit.json --shots 1024
```

Offline fixture results are not IBM hardware results. Transpilation, backend submission, and diagnosis are reported as separate steps. No IBM execution is claimed in this workspace because Qiskit Runtime, credentials, and backend access are unavailable.

## Bugs4Q Validation Requirements

Place Bugs4Q dataset files under a directory or filename containing `Bugs4Q`; JSON, JSONL/NDJSON, and CSV are supported by the adapter. References are counted as reliable only when explicitly marked both independent and reliable, with no source disagreement.

Focused real-source Bugs4Q integration is complete for 8 selected official cases: 8, 12, 17, 25, 26, 30, 31, and 39. All 8 cases were executed and represented successfully; the results were 3 `NO_ANOMALY` and 5 `PROGRAM_FAULT`. This focused integration does not provide benchmark accuracy, precision, recall, or F1 because independent ground-truth/reference-data evaluation is still unavailable. Separate Bugs4Q reference-data validation reports may therefore still show dataset/reference-data unavailable; that status does not mean the focused integration did not happen. Synthetic validation scenarios are controlled test cases, not Bugs4Q ground truth.

## Multi-Benchmark Validation

Run the independent benchmark adapters and combined summary with:

```powershell
.\.venv\Scripts\python.exe -m benchmark_validation --root .
```

Bugs4Q and QMutBench use separate adapters, case reports, and metrics because their labels may describe different concepts. QMutBench discovery requires verified benchmark provenance and case-like structure; a filename containing only “mutation” is not enough. Mutation operators remain benchmark metadata and are never sent to diagnosis as labels. Ground truth is `RELIABLE` only when it is explicitly marked reliable and independent; uncertain or conflicting labels are preserved and not counted. Temporary fixtures in tests are labeled `TEST_FIXTURE`, not benchmark results.

Reports remain independent at `reports/benchmarks/bugs4q_validation_report.json` and `reports/benchmarks/qmutbench_validation_report.json`; `combined_validation_report.json` summarizes both without cross-benchmark metrics. The QMutBench adapter accepts JSON, JSONL/NDJSON, and CSV case files; it reports missing fields rather than inferring answers. Real QMutBench pilot data was integrated, with 12 paired cases identified: 11 were executed and diagnosed, and 1 was unsupported because it uses `RXX`. All 11 executed cases were diagnosed as `PROGRAM_FAULT`. Benchmark accuracy, precision, recall, F1, and mutation score are not claimed.

## Limitations and Phase Status

- No new hardware noise model is included; hardware observations in the demo are deterministic fixtures.
- No benchmark accuracy, precision, recall, or F1 is asserted without suitable ground truth.
- The `DiagnosisCategory` enum currently represents ambiguous and insufficient evidence with a shared value, `AMBIGUOUS / INSUFFICIENT_EVIDENCE`.
- Phase I: PARTIAL; focused real-source integration is complete for 8 selected official Bugs4Q cases, while independent ground-truth/reference-data evaluation remains unavailable.
- Phase J: PARTIAL; IBM adapter is implemented and offline-tested, but live access is unavailable.
- Phase K: COMPLETE after the full regression suite and all five demo scenarios pass.
