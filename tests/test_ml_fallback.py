import numpy as np
import subprocess
import sys

from ml_evaluation import _features
from ml_fallback_training import BootstrapRandomForest, _add_ideal_circuit_features


def test_custom_fallback_forest_fits_predicts_and_is_deterministic():
    X = np.array([
        [0.0, 0.0], [0.1, 0.2], [0.2, 0.0],
        [1.0, 1.0], [0.9, 0.8], [0.8, 1.0],
        [2.0, 0.0], [1.8, 0.1], [2.0, 0.2],
    ])
    y = np.array(["NORMAL"] * 3 + ["PROGRAM_FAULT"] * 3 + ["HARDWARE_ANOMALY"] * 3)

    first = BootstrapRandomForest(n_estimators=12, max_depth=4, random_state=17).fit(X, y)
    second = BootstrapRandomForest(n_estimators=12, max_depth=4, random_state=17).fit(X, y)

    assert np.array_equal(first.predict(X), second.predict(X))
    assert set(first.predict(X)) <= set(y)
    assert first.feature_importances_.shape == (X.shape[1],)
    assert len(first.estimators_) == 12


def test_fallback_imports_do_not_load_blocked_sklearn_ensemble():
    subprocess.run(
        [sys.executable, "-c", "import sys, ml_fallback_training; assert 'sklearn.ensemble' not in sys.modules"],
        check=True,
    )


def test_gate_mismatch_uses_circuit_operations_not_target_metadata():
    report = {
        "circuit": {"ideal": [["h", 0], ["measure", [0]]],
                    "observed": [["x", 0], ["measure", [0]]]},
        "experiment_metadata": {
            "ground_truth_label": "PROGRAM_FAULT",
            "fault_type": "gate_replacement",
            "mutation_type": "replacement",
            "program_fault_id": "fault-01-gate_replacement",
        },
        "diagnosis": {"category": "PROGRAM_FAULT"},
    }
    expected = _features(report)["gate_mismatch_fraction"]
    report["experiment_metadata"].update({
        "ground_truth_label": "NORMAL",
        "fault_type": "different",
        "mutation_type": "different",
        "program_fault_id": "different-id",
    })
    report["diagnosis"]["category"] = "NORMAL"

    assert _features(report)["gate_mismatch_fraction"] == expected
    assert expected > 0


def test_ideal_circuit_features_read_only_reference_structure(tmp_path, monkeypatch):
    import json
    import ml_fallback_training

    (tmp_path / "run.json").write_text(json.dumps({
        "circuit": {"ideal": [["h", 0], ["rx", 0, 0.2], ["h", 0]],
                    "observed": [["x", 0]]},
        "configuration": {"num_qubits": 2, "ground_truth_label": "PROGRAM_FAULT"},
        "experiment_metadata": {"ground_truth_label": "PROGRAM_FAULT"},
    }), encoding="utf-8")
    monkeypatch.setattr(ml_fallback_training, "ROOT", tmp_path)
    enriched = _add_ideal_circuit_features([{
        "report_path": "run.json", "ground_truth_label": "PROGRAM_FAULT",
    }])[0]

    assert enriched["ideal_operation_count"] == 3
    assert enriched["ideal_unique_gate_count"] == 2
    assert enriched["ideal_qubit_count"] == 2


def test_fallback_runner_blocks_before_loading_or_fitting_ineligible_dataset(tmp_path, monkeypatch, capsys):
    import ml_fallback_training
    import ml_training

    monkeypatch.setattr(ml_training, "run_preflight", lambda: {"ready": False, "blockers": ["missing lineage"]})
    monkeypatch.setattr(ml_fallback_training, "OUT", tmp_path / "blocked-output")
    monkeypatch.setattr(ml_fallback_training, "load_dataset", lambda: (_ for _ in ()).throw(AssertionError("loaded before preflight")))

    assert ml_fallback_training.main() == 2
    assert "missing lineage" in capsys.readouterr().out
    assert not (tmp_path / "blocked-output").exists()
