# QMutBench ML Classifier Lab

A small Streamlit dashboard to present the QMutBench mutation-outcome classifier results and, when the local model bundle is available, run a real prediction.

## Run in Colab

From the repository root:

```bash
pip install streamlit pandas joblib scikit-learn
streamlit run qmutbench_ml/demo/app.py --server.port 8501
```

In Colab, expose port 8501 using the port-forwarding option available in your notebook environment. Alternatively, copy the app to a local machine and open the Streamlit URL printed in the terminal.

## What the dashboard shows

- Reproduced out-of-fold development metrics for the metadata baseline and mutation-aware Extra Trees model.
- Nested grouped development model-selection metrics.
- A live prediction form that invokes the repository's validated `QMutBenchOutcomeModel` API only if both required local bundle files exist.

## Required local model bundle for predictions

```
data/qmutbench_ml/mutation_aware_v1/best_model.joblib
data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json
```

The model bundle is intentionally excluded from Git. The dashboard remains useful as a results presentation without it, but does **not** fabricate live predictions.

## Important interpretation

The target is the recorded historical QMutBench mutation outcome (Detected vs Survivor), not quantum hardware anomaly diagnosis. Results are development evidence, not independent external validation. The research report documents that the 85% macro-F1 target was not reached and compatible external validation evidence is unavailable.
