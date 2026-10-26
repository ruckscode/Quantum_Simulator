# Colab-native QMutBench results

This folder is isolated from the quantum simulator's core code. It displays metrics from the saved QMutBench research report inside Google Colab. It does not retrain classifiers or manufacture results.

## Run in Colab

From the repository root:

```python
%cd /content/Quantum_Simulator
!git pull --ff-only origin qmutbench-master-optimization
!pip -q install pandas matplotlib
%run qmutbench_ml/colab_results/show_results.py
```

The viewer reads the specific report at `reports/experiments/qmutbench_final_research_report_20261026.json` and displays a compact comparison of baseline OOF, mutation-aware OOF, and nested grouped evaluation metrics. It also reports whether the expected serialized model files are present.

## Model bundle status and expected location

The viewer checks for these files under `qmutbench_ml/colab_results/models/`:

- `best_model.joblib`
- `best_model_metadata.json`

These files are not currently available in the checked repository location. The report records a historical inference smoke test, but that is not a substitute for having the serialized model bundle available now. Do not claim live inference from the report-only viewer. If the original artifacts cannot be recovered from the training environment or backups, the export must be regenerated through the project's actual training/inference pipeline and verified against the documented 23-feature schema.

## Integrity

- This is a report viewer, not a training pipeline.
- QMutBench mutation-outcome prediction and hardware anomaly diagnosis are different tasks.
- The nested grouped score is development evidence, not independent external validation.
- The 85% Macro F1 target remains unmet.
- Core simulator files are not modified by this viewer.
