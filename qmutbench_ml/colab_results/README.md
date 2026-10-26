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

The viewer reads `reports/experiments/qmutbench_final_research_report_20261026.json` and displays a compact comparison of baseline OOF, mutation-aware OOF, and nested grouped evaluation metrics. It creates the chart from the saved values and checks for the model bundle.

## Model bundle

The documented canonical local-only location is:

```
data/qmutbench_ml/mutation_aware_v1/best_model.joblib
data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json
```

The viewer checks this location first, then `qmutbench_ml/colab_results/models/`. The bundle is excluded from Git, so a clone or Colab runtime may not contain it. The report records a historical inference smoke test, but that does not prove the serialized files are available in the current runtime. The viewer will not fabricate predictions or retrain silently. If the bundle is missing, recover it from the original training environment/backup or rerun the documented training/export pipeline and verify the resulting model hash and feature contract.

## Integrity

- This is a report viewer, not a training pipeline.
- QMutBench mutation-outcome prediction and hardware anomaly diagnosis are different tasks.
- Nested grouped and out-of-fold scores are development evidence, not independent external validation.
- The 85% Macro F1 target remains unmet.
- Core simulator files are not modified by this viewer.
