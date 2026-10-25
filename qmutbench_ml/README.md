# QMutBench outcome inference

This package applies the saved model for predicting the recorded historical QMutBench mutation outcome. It does not diagnose hardware noise or localize simulator faults.

## Model bundle

The promoted development model is stored outside Git under `data/qmutbench_ml/position_fraction_v1/` as `best_model.joblib` with `best_model_metadata.json`. The previous seven-feature baseline remains preserved under `data/qmutbench_ml/master_optimization_v1/`. Pass the selected bundle's paths explicitly. The loader checks the model SHA-256 against metadata, the input and model feature contracts, target mapping, saved threshold, and fitted categorical vocabularies.

Required columns, in training order:

| Feature | Type | Missing/unknown handling |
|---|---|---|
| `algorithm` | categorical string | Missing and unseen values rejected |
| `operation` | categorical string | Missing and unseen values rejected |
| `gate` | categorical string | Missing and unseen values rejected |
| `dominant_state` | categorical string | Missing and unseen values rejected |
| `qubits` | finite number | Missing and non-finite values rejected |
| `gates` | finite number | Missing and non-finite values rejected |
| `Position_int` | finite number | Missing and non-finite values rejected |
| `position` | QMutBench percentage band, such as `31-40%` | Missing and unseen bands rejected; converted to a fraction using the documented band midpoint |

The promoted model receives the derived numeric feature `position_fraction` in place of source `position`; the API derives it using the bundle's saved mapping. `Killed`, `Origin_program`, row identifiers, and any other extra columns are rejected. The dataset's one-hot preprocessing handles learned categories; the API rejects values outside those learned categories before prediction. No imputation is performed at inference.

## Use

```python
from qmutbench_ml import QMutBenchOutcomeModel

predictor = QMutBenchOutcomeModel(
    "data/qmutbench_ml/position_fraction_v1/best_model.joblib",
    "data/qmutbench_ml/position_fraction_v1/best_model_metadata.json",
)

one = predictor.predict({
    "algorithm": "qft",
    "operation": "Replace",
    "gate": "h",
    "dominant_state": "0",
    "qubits": 5,
    "gates": 30,
    "Position_int": 4,
    "position": "0-10%",
})
many = predictor.predict([one_input_record, another_input_record])
```

Each returned record includes `predicted_class`, `predicted_label`, `probability_detected`, `applied_threshold`, and `model_version`. The probability is an uncalibrated model probability, not calibrated confidence. The default threshold is loaded from metadata (currently 0.5); an explicit threshold override is available through `predict(..., threshold=...)`.

Run focused tests with `.venv/Scripts/python.exe -m pytest tests/test_qmutbench_inference.py -q`.
