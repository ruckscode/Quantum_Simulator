import numpy as np
import pandas as pd
import pennylane as qml

from qmutbench_qml.data.loader import FEATURES, TARGET_MAP, load_development_records
from qmutbench_qml.features.reduction import ReducedFeatureEncoder
from qmutbench_qml.models.vqc import DataReuploadingClassifier, VariationalQuantumClassifier
from qmutbench_qml.models.qsvc import QuantumSupportVectorClassifier
from qmutbench_qml.models.quantum_reservoir import QuantumReservoirClassifier
from qmutbench_qml.training.pilot import _classify_score, _small_group_sample, _validate_prediction_alignment, metrics


def test_full_streaming_memmap_handles_close_before_windows_temp_cleanup(tmp_path):
    from qmutbench_qml.training.run_full_streaming import _close_memmaps
    path = tmp_path / "small.dat"
    mapped = np.memmap(path, mode="w+", dtype=np.float64, shape=(4,))
    mapped[:] = np.arange(4)
    mapped.flush()
    _close_memmaps(mapped)
    assert mapped._mmap.closed
    path.unlink()
    assert not path.exists()


def test_verified_target_and_full_development_feature_adapter():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    frame, audit = load_development_records(root)
    assert TARGET_MAP == {"WOO": 1, "OPO": 1, "Survivor": 0}
    assert len(FEATURES) == 23
    assert len(frame) == audit["development_rows"] == 578117
    assert frame.origin_program.nunique() == audit["development_origins"] == 305
    assert frame.record_id.is_unique


def test_reduction_fits_only_training_rows_and_scales_to_angle_domain():
    frame = pd.DataFrame({
        "algorithm": ["a", "b", "a", "b", "a", "b"],
        "operation": ["Add"] * 6, "gate": ["x", "h"] * 3,
        "dominant_state": ["0", "1"] * 3,
        "qasm_diff_removed_gate": ["none"] * 6, "qasm_diff_added_gate": ["x"] * 6,
        "qasm_diff_alignment_class": ["single_insert"] * 6,
        "qasm_diff_local_prev_gate": ["h"] * 6, "qasm_diff_local_next_gate": ["cx"] * 6,
        "qubits": [2, 3, 2, 3, 2, 3], "gates": [3, 4, 3, 4, 3, 4],
        "Position_int": [0, 1, 2, 3, 4, 5], "position_fraction": [.05, .15, .25, .35, .45, .55],
        "qasm_diff_orig_span": [0] * 6, "qasm_diff_mut_span": [1] * 6,
        "qasm_diff_total_span": [1] * 6, "qasm_diff_expected_gate_match": [1, 0] * 3,
        "qasm_diff_qubit_set_equal": [-1] * 6, "qasm_diff_qubits_touched": [1, 2, 1, 2, 1, 2],
        "qasm_diff_removed_arity": [-1] * 6, "qasm_diff_added_arity": [1] * 6,
        "qasm_diff_exact_single_edit": [1, 0] * 3, "qasm_diff_is_local": [1] * 6,
    })
    encoder = ReducedFeatureEncoder().fit(frame.iloc[:4], np.array([0, 1, 0, 1]))
    x = encoder.transform(frame.iloc[4:])
    assert x.shape == (2, 4)
    assert np.isfinite(x).all()
    assert x.min() >= 0 and x.max() <= 1
    assert len(encoder.selected_names) == 4


def test_group_sampling_and_group_fold_have_no_origin_overlap():
    frame = pd.DataFrame({"origin_program": np.repeat([f"o{i}" for i in range(9)], 12),
                          "record_id": np.arange(108), "y": np.tile([0, 1], 54)})
    sampled, origins = _small_group_sample(frame)
    assert sampled.origin_program.nunique() == len(origins) == 9
    assert sampled.groupby("origin_program").size().max() <= 8
    from sklearn.model_selection import GroupKFold
    for train, valid in GroupKFold(3).split(sampled, sampled.y, sampled.origin_program):
        assert not (set(sampled.iloc[train].origin_program) & set(sampled.iloc[valid].origin_program))


def test_margin_models_use_zero_decision_boundary():
    assert _classify_score("rbf_svc_4f", np.array([-.2, .1])).tolist() == [0, 1]
    assert _classify_score("qsvc", np.array([-.2, .1])).tolist() == [0, 1]
    assert _classify_score("vqc", np.array([.49, .51])).tolist() == [0, 1]


def test_metrics_and_comparison_integrity():
    scores = np.array([.2, .7, .4, .8])
    y = np.array([0, 1, 1, 0])
    result = metrics(y, np.array([0, 1, 0, 1]), scores)
    assert result["confusion_matrix_labels_0_1"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == .5
    records = pd.DataFrame([
        {"record_id": 1, "origin_program": "a", "y_true": 0, "fold": 1, "model": "qml", "y_pred": 0, "score": .2},
        {"record_id": 1, "origin_program": "a", "y_true": 0, "fold": 1, "model": "classical", "y_pred": 1, "score": .7},
    ])
    assert _validate_prediction_alignment(records, ["qml", "classical"]).shape[0] == 1
    with np.testing.assert_raises(AssertionError):
        _validate_prediction_alignment(records.iloc[:1], ["qml", "classical"])


def test_vqc_executes_and_updates_trainable_parameters():
    model = VariationalQuantumClassifier(n_qubits=2, depth=1, steps=2, seed=7)
    x = np.array([[.05, .1], [.8, .95], [.2, .7], [.9, .15]])
    y = np.array([0, 1, 0, 1])
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 2)
    assert np.isfinite(probs).all()
    assert model.training_executions > 0
    assert not np.allclose(np.asarray(model.weights), model.initial_weights)
    assert not np.allclose(probs[:, 1], probs[0, 1])
    assert model.weights.shape == (1, 2, 3)


def test_data_reuploading_classifier_executes_a_distinct_trainable_circuit():
    model = DataReuploadingClassifier(n_qubits=2, depth=2, steps=2, seed=8)
    x = np.array([[.05, .1], [.8, .95], [.2, .7], [.9, .15]])
    y = np.array([0, 1, 0, 1])
    model.fit(x, y)
    assert model.data_reuploading is True
    assert np.isfinite(float(model.circuit(x[0], model.weights)))
    assert model.predict_proba(x).shape == (4, 2)
    assert model.training_executions > 0


def test_quantum_reservoir_uses_circuits_and_fits_classical_readout():
    model = QuantumReservoirClassifier(n_qubits=2, seed=9)
    x = np.array([[.05, .1], [.8, .95], [.2, .7], [.9, .15]])
    y = np.array([0, 1, 0, 1])
    model.fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert model.training_executions == len(x)
    assert model.inference_executions == len(x)


def test_qsvc_is_a_quantum_kernel_and_runs_on_pennylane_device():
    model = QuantumSupportVectorClassifier(n_qubits=2, seed=4)
    x = np.array([[.1, .2], [.8, .9], [.3, .7], [.9, .15]])
    kernel = model.kernel_matrix(x[:2], x)
    assert kernel.shape == (2, 4)
    assert np.allclose(kernel[0, 0], 1.0, atol=1e-8)
    assert model.last_kernel_executions == 8
    model.fit_with_features(x, np.array([0, 1, 0, 1]))
    assert model.training_executions == 16
    assert np.isfinite(model.decision_function(x[:2])).all()
