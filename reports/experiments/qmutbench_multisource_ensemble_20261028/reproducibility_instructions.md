# Reproduction

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe experiments/qmutbench_multisource_ensemble_audit.py
```

The script reads the three saved three-class development prediction files named in `dataset_provenance.json`, validates candidate and record alignment, performs fixed equal probability means and an outer-fold-held-out logistic stacker (`C=0.1`, seed `20261028`), and writes predictions, metrics, pairwise error overlap, per-origin results, and 2,000-origin-bootstrap intervals. It does not load raw QMutBench data or frozen-test labels.
