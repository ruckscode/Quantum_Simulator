# Colab-native QMutBench results

This folder is separate from the quantum simulator's core code. It reads saved JSON experiment reports and displays their real metrics inside Google Colab. It does not change the simulator, retrain classifiers, or manufacture results.

## Run in Colab

From the repository root:

```python
%cd /content/Quantum_Simulator
!git pull --ff-only origin qmutbench-master-optimization
!pip -q install pandas matplotlib
%run qmutbench_ml/colab_results/show_results.py
```

The script recursively discovers JSON reports under `reports/experiments/`, finds recognized numeric metric blocks, displays them in a table, and plots comparable metrics when available. Values are read at runtime, not copied into the script.

## Optional model artifacts

If you have the genuine trained model bundle, place `best_model.joblib` and `best_model_metadata.json` in `models/`. Model files are deliberately not committed to Git. The viewer reports whether they exist and does not claim live inference works when they are absent.

## Integrity

- This is a results viewer, not a training pipeline.
- QMutBench mutation-outcome prediction and hardware anomaly diagnosis are separate tasks.
- Development and out-of-fold scores do not establish independent external generalization.
