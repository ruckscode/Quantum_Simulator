"""Pure-Python integrity tests for the bounded binary QNN campaign."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qmutbench_qml.training import run_binary_development as binary


def test_binary_label_mapping_keeps_killed_semantics_explicit():
    assert binary.map_label("WOO") == 1
    assert binary.map_label("OPO") == 1
    assert binary.map_label("Survivor") == 0
    with pytest.raises(ValueError, match="Unexpected QMutBench binary label"):
        binary.map_label("unknown")


def test_configuration_accepts_reproducible_defaults_and_rejects_invalid_architecture():
    binary.validate_config(dict(binary.DEFAULT_CONFIG))
    bad = dict(binary.DEFAULT_CONFIG)
    bad["fold_count"] = 3
    with pytest.raises(ValueError, match="five-fold"):
        binary.validate_config(bad)


def test_reference_split_rejects_origin_crossing_folds():
    rows = [
        {"record_id": 1, "origin_program": "a", "y": 0, "fold": 1},
        {"record_id": 2, "origin_program": "b", "y": 1, "fold": 2},
    ]
    assert binary.validate_group_assignments(rows, folds=5) == {"a": 1, "b": 2}
    rows.append({"record_id": 3, "origin_program": "a", "y": 1, "fold": 3})
    with pytest.raises(ValueError, match="crosses reference folds"):
        binary.validate_group_assignments(rows, folds=5)


def test_sampling_is_repeatable_group_capped_and_fold_separated():
    strata = {
        (1, 0, "a"): list(range(0, 20)),
        (1, 1, "a"): list(range(20, 40)),
        (2, 0, "b"): list(range(40, 60)),
        (2, 1, "b"): list(range(60, 80)),
    }
    train = binary.stratified_origin_sample(strata, 1, "train", 5, 10, 31)
    validation = binary.stratified_origin_sample(strata, 1, "validation", 5, 10, 31)
    assert train == binary.stratified_origin_sample(strata, 1, "train", 5, 10, 31)
    assert len(train) == len(set(train)) == 10
    assert len(validation) == len(set(validation)) == 10
    assert set(train).isdisjoint(validation)
    assert set(train) <= set(range(40, 80))
    assert set(validation) <= set(range(0, 40))


def _mock_payload(fold):
    return {"predictions": [{"record_id": fold, "model": "mock", "y_true": 0, "y_pred": 0}],
            "metrics": [{"fold": fold, "model": "mock", "accuracy": 1.0}]}


def test_completed_fold_checkpoint_resumes_after_interruption(tmp_path):
    config = {"seed": 17, "qubits": 2}
    calls = []

    def trainer(fold):
        def run():
            calls.append(fold)
            return _mock_payload(fold)
        return run

    first, resumed = binary.load_or_train_fold(
        tmp_path, 1, "fingerprint-1", config, trainer(1), resume=False)
    assert not resumed and first["payload"] == _mock_payload(1)
    assert calls == [1]
    # Simulate interruption after fold one, before fold two starts.
    one, resumed = binary.load_or_train_fold(
        tmp_path, 1, "fingerprint-1", config, trainer(1), resume=True)
    two, resumed_two = binary.load_or_train_fold(
        tmp_path, 2, "fingerprint-2", config, trainer(2), resume=True)
    assert resumed and not resumed_two
    assert one["payload"] == _mock_payload(1)
    assert two["payload"] == _mock_payload(2)
    assert calls == [1, 2]
    with pytest.raises(FileExistsError, match="pass --resume"):
        binary.load_or_train_fold(tmp_path, 1, "fingerprint-1", config, trainer(1), resume=False)


def test_corrupt_checkpoint_is_rejected(tmp_path):
    config = {"seed": 17}
    binary.save_completed_fold(tmp_path, 1, "fingerprint", config, _mock_payload(1))
    path = tmp_path / "fold_01.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["payload"]["predictions"][0]["y_pred"] = 1
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or incomplete"):
        binary.load_completed_fold(tmp_path, 1, "fingerprint", config)


@pytest.mark.parametrize("changed", [
    {"seed": 18},
    {"seed": 17, "fold_definition": "different"},
    {"seed": 17, "feature_schema": ["different"]},
    {"seed": 17, "model_configuration": {"depth": 3}},
])
def test_stale_configuration_is_rejected(tmp_path, changed):
    config = {"seed": 17}
    binary.save_completed_fold(tmp_path, 1, "fingerprint", config, _mock_payload(1))
    with pytest.raises(ValueError, match="Stale fold checkpoint"):
        binary.load_completed_fold(tmp_path, 1, "fingerprint", changed)


def test_stale_dataset_fingerprint_and_incomplete_checkpoint_are_rejected(tmp_path):
    config = {"seed": 17}
    binary.save_completed_fold(tmp_path, 1, "fingerprint", config, _mock_payload(1))
    with pytest.raises(ValueError, match="Stale fold checkpoint"):
        binary.load_completed_fold(tmp_path, 1, "different-dataset", config)
    (tmp_path / "fold_01.json").unlink()
    with pytest.raises(ValueError, match="Incomplete fold checkpoint"):
        binary.load_completed_fold(tmp_path, 1, "fingerprint", config)


def test_checkpoint_has_required_provenance_fields(tmp_path):
    config = {"seed": 17}
    binary.save_completed_fold(tmp_path, 3, "stable-fingerprint", config, _mock_payload(3))
    payload = json.loads((tmp_path / "fold_03.json").read_text(encoding="utf-8"))
    assert payload["schema"] == binary.SCHEMA
    assert payload["fold"] == 3
    assert payload["configuration"] == config
    assert payload["fingerprint"] == "stable-fingerprint"
    assert len(payload["payload_sha256"]) == 64


def test_interrupted_atomic_write_leaves_no_complete_checkpoint(tmp_path, monkeypatch):
    config = {"seed": 17}
    replace = binary.os.replace
    monkeypatch.setattr(binary.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("simulated interruption")))
    with pytest.raises(OSError, match="simulated interruption"):
        binary.save_completed_fold(tmp_path, 1, "fingerprint", config, _mock_payload(1))
    assert not (tmp_path / "fold_01.json").exists()
    with pytest.raises(ValueError, match="Incomplete fold checkpoint"):
        binary.load_completed_fold(tmp_path, 1, "fingerprint", config)
    monkeypatch.setattr(binary.os, "replace", replace)
    saved = binary.save_completed_fold(tmp_path, 1, "fingerprint", config, _mock_payload(1))
    assert saved.is_file()
    assert binary.load_completed_fold(tmp_path, 1, "fingerprint", config)["payload"] == _mock_payload(1)


def test_binary_vqc_prediction_supplies_neutral_readout_to_quantum_expectations():
    """Catch inference call/signature drift without importing PennyLane under low RAM."""
    model_path = Path(binary.__file__).resolve().parents[1] / "models" / "binary_vqc.py"
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    predict = next(node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == "predict_proba")
    calls = [node for node in ast.walk(predict) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "_logits"]
    assert len(calls) == 1
    assert len(calls[0].args) == 4
    assert isinstance(calls[0].args[2], ast.Call) and calls[0].args[2].func.attr == "array"
    assert calls[0].args[2].args[0].value == 1.0
    assert isinstance(calls[0].args[3], ast.Call) and calls[0].args[3].func.attr == "array"
    assert calls[0].args[3].args[0].value == 0.0


def _epoch_state(epoch):
    return {"completed_epochs": epoch, "weights": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            "readout_scale": 1.1, "readout_bias": -0.1,
            "loss_history": [0.8 - 0.01 * i for i in range(epoch)],
            "rng_state": {"bit_generator": "PCG64", "state": {"state": 123 + epoch, "inc": 456}},
            "optimizer_accumulation": {"fm": [[[0.01, 0.02, 0.03], 0, 0],
                                               [[0.1, 0.2, 0.3], 0, 0],
                                               [0, 0, 0]],
                                        "sm": [[[0.001, 0.002, 0.003], 0, 0],
                                               [[0.01, 0.02, 0.03], 0, 0],
                                               [0, 0, 0]], "t": epoch * 4},
            "training_seconds": float(epoch * 2), "training_executions": epoch * 12,
            "peak_rss_bytes": 123456789}


def test_qnn_epoch_checkpoint_restores_only_complete_validated_boundaries(tmp_path):
    config = {**binary.DEFAULT_CONFIG, "epochs": 8, "qubits": 2, "depth": 1}
    state1 = _epoch_state(1)
    path1 = binary.save_qnn_epoch_checkpoint(tmp_path, 1, "fp", config, state1)
    assert path1.name == "fold_01_qnn_epoch_0001.json"
    assert binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "fp", config) == state1
    state2 = _epoch_state(2)
    binary.save_qnn_epoch_checkpoint(tmp_path, 1, "fp", config, state2)
    assert binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "fp", config) == state2
    with pytest.raises(ValueError, match="Stale QNN epoch checkpoint"):
        binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "changed", config)
    latest = tmp_path / "fold_01_qnn_epoch_0002.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    payload["state"]["readout_bias"] = 9.0
    latest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or incomplete QNN epoch"):
        binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "fp", config)


def test_failed_epoch_checkpoint_replace_preserves_previous_boundary(tmp_path, monkeypatch):
    config = {**binary.DEFAULT_CONFIG, "epochs": 8, "qubits": 2, "depth": 1}
    binary.save_qnn_epoch_checkpoint(tmp_path, 1, "fp", config, _epoch_state(1))
    original_replace = binary.os.replace
    monkeypatch.setattr(binary.os, "replace",
                        lambda *_: (_ for _ in ()).throw(OSError("simulated disk interruption")))
    with pytest.raises(OSError, match="simulated disk interruption"):
        binary.save_qnn_epoch_checkpoint(tmp_path, 1, "fp", config, _epoch_state(2))
    assert binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "fp", config) == _epoch_state(1)
    monkeypatch.setattr(binary.os, "replace", original_replace)
    binary.save_qnn_epoch_checkpoint(tmp_path, 1, "fp", config, _epoch_state(2))
    assert binary.load_latest_qnn_epoch_checkpoint(tmp_path, 1, "fp", config) == _epoch_state(2)


def test_memory_preflight_enforces_startup_gate_and_reports_estimate(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=2047 * 1024 * 1024))
    with pytest.raises(MemoryError, match="Only 2047 MiB available.*requires 2048 MiB"):
        binary.memory_preflight(dict(binary.DEFAULT_CONFIG))
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=2048 * 1024 * 1024))
    preflight = binary.memory_preflight(dict(binary.DEFAULT_CONFIG))
    assert preflight["available_mib"] == 2048
    assert preflight["required_start_mib"] == 2048
    assert preflight["estimated_peak_mib"] >= binary.RAM_FLOOR_MIB


def test_in_run_memory_floor_stops_at_1024_mib(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1023 * 1024 * 1024))
    with pytest.raises(MemoryError, match="1023 MiB, below the 1024 MiB floor"):
        binary.require_memory_floor(psutil, binary.RAM_FLOOR_MIB, "test stage")
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1024 * 1024 * 1024))
    assert binary.require_memory_floor(psutil, binary.RAM_FLOOR_MIB, "test stage") == 1024
