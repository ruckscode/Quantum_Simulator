"""Train/export an observable-only Logistic Regression development candidate.

Run in an environment where sklearn's native extensions load successfully.
This legacy full-dataset route does not establish independent deployment
validation or calibrated probabilities. Its strict provenance preflight must
pass before fitting; a successful export is still only a development candidate.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_evaluation import LABELS, load_dataset
from quantum_simulator.ml_diagnosis import CLASSES, FEATURE_NAMES
from quantum_simulator.observable_model_adapter import ARTIFACT_TYPE, DEFAULT_ARTIFACT_PATH

MANIFEST = ROOT / "reports" / "experiments" / "full_dataset" / "dataset_manifest.json"


def export_model(output_path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    # Export is model fitting on the full selected dataset. It must obey the
    # same source/lineage/label gate as cross-validation and stop before loading
    # feature rows when provenance or independent group eligibility fails.
    from ml_training import run_preflight

    preflight = run_preflight()
    if not preflight["ready"]:
        blockers = "; ".join(preflight["blockers"])
        raise RuntimeError(f"Observable model export blocked by dataset preflight: {blockers}")

    rows = load_dataset()
    if len(rows) != 270 or len({row["circuit_family_id"] for row in rows}) != 30:
        raise ValueError("Expected the validated 270-record, 30-family full dataset")
    if Counter(row["ground_truth_label"] for row in rows) != Counter({name: 90 for name in LABELS}):
        raise ValueError("Expected 90 examples per diagnosis class")

    # Build matrices exclusively from the approved independent observables.
    x = np.asarray([
        [float(row[feature]) if row[feature] is not None else np.nan for feature in FEATURE_NAMES]
        for row in rows
    ], dtype=float)
    y = np.asarray([row["ground_truth_label"] for row in rows])
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    pipeline.fit(x, y)

    imputer, scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    class_rows = {str(label): index for index, label in enumerate(classifier.classes_)}
    order = [class_rows[label] for label in CLASSES]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "training_status": "DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT",
        "model_type": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASSES),
        "preprocessing": {
            "imputer": "sklearn.impute.SimpleImputer(strategy='median')",
            "median_imputer_statistics": imputer.statistics_.astype(float).tolist(),
            "scaler": "sklearn.preprocessing.StandardScaler()",
            "standard_scaler_mean": scaler.mean_.astype(float).tolist(),
            "standard_scaler_scale": scaler.scale_.astype(float).tolist(),
        },
        "coefficients": classifier.coef_[order].astype(float).tolist(),
        "intercept": classifier.intercept_[order].astype(float).tolist(),
        "metadata": {
            "dataset_name": manifest.get("dataset_name", "controlled_execution_derived_30_family_dataset"),
            "dataset_source": "reports/experiments/full_dataset (270 execution reports)",
            "training_records": len(rows),
            "training_group_count": 30,
            "evaluation_methodology": "Previously evaluated StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); this final deployable estimator is fit on all 270 records after evaluation.",
            "model_configuration": "Pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))",
            "random_seed": None,
            "inference_features_only": True,
            "labels_or_row_metadata_in_artifact": False,
            "deployment_validation": "NOT_PERFORMED",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(export_model())
