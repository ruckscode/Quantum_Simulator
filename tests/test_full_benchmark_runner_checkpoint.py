import json
from types import SimpleNamespace

import pytest

import full_benchmark_runner as runner


def _case():
    return SimpleNamespace(
        case_id="AddGate_rxx_inGap_1_",
        source_files=["origin.qasm", "AddGate_rxx_inGap_1_.qasm"],
        parse_error=None,
        raw={"mutation": {"operator": "AddGate"}, "num_qubits": 2},
    )


def _dataset():
    return SimpleNamespace(cases=[_case()], issues=[], data_files=["origin.qasm", "AddGate_rxx_inGap_1_.qasm"])


def _result(*, support_status="UNSUPPORTED", execution_status="NOT_EXECUTED", diagnosis=None):
    return {
        "case_id": "AddGate_rxx_inGap_1_",
        "support_status": support_status,
        "supported": support_status == "SUPPORTED",
        "execution_status": execution_status,
        "system_diagnosis": diagnosis,
        "benchmark_ground_truth": {"status": "MISSING", "diagnosis_category": None, "diagnosis_category_compatible": False},
        "diagnosis_ground_truth_compatible": False,
        "errors": ["unsupported simulator operation 'rxx'"] if support_status == "UNSUPPORTED" else [],
        "reference_circuit": [],
        "mutant_circuit": [],
        "original_output": None,
        "mutant_output": None,
        "behavioral_comparison": None,
        "behaviorally_different": None,
    }


def test_checkpoint_writer_atomically_persists_complete_document(tmp_path):
    checkpoint = tmp_path / "nested" / "checkpoint.json"
    runner._write_qmut_checkpoint(
        checkpoint,
        {"schema": 1},
        {"case-1": {"case_id": "case-1", "value": "complete"}},
        {"status": "IN_PROGRESS"},
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["results"] == [{"case_id": "case-1", "value": "complete"}]
    assert list(checkpoint.parent.iterdir()) == [checkpoint]


def test_checkpoint_replace_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"old":"complete"}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise PermissionError("simulated Windows checkpoint lock")

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="checkpoint lock"):
        runner._write_qmut_checkpoint(checkpoint, {"schema": 2}, {}, {"status": "IN_PROGRESS"})

    assert json.loads(checkpoint.read_text(encoding="utf-8")) == {"old": "complete"}
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_benchmark_report_writer_atomically_writes_json_and_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    runner._write("qmutbench", {"records": [{"case_id": "fixture-1", "status": "UNSUPPORTED"}]})

    report_path = tmp_path / "qmutbench_full_results.json"
    csv_path = tmp_path / "qmutbench_full_results.csv"
    assert json.loads(report_path.read_text(encoding="utf-8"))["records"][0]["case_id"] == "fixture-1"
    assert "fixture-1" in csv_path.read_text(encoding="utf-8")
    assert set(path.name for path in tmp_path.iterdir()) == {report_path.name, csv_path.name}


def _configure_runner(monkeypatch, tmp_path, dataset, fresh_result):
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "reports")
    monkeypatch.setattr(runner, "discover_qmutbench_files", lambda _root: dataset)
    monkeypatch.setattr(
        runner,
        "validate_qmutbench_case",
        lambda case, **kwargs: calls.append((case.case_id, kwargs)) or SimpleNamespace(to_dict=lambda: fresh_result),
    )
    monkeypatch.setattr(runner, "_write", lambda *_args, **_kwargs: None)
    return calls


def test_legacy_checkpoint_row_is_recomputed_and_replaced(monkeypatch, tmp_path):
    dataset = _dataset()
    calls = _configure_runner(monkeypatch, tmp_path, dataset, _result())
    checkpoint = tmp_path / "reports" / "qmutbench_full_checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps([{
            "case_id": "AddGate_rxx_inGap_1_",
            "status": "OK",
            "support_status": "SUPPORTED",
            "execution_status": "COMPLETED",
            "diagnosed": True,
            "system_diagnosis": {"category": "PROGRAM_FAULT"},
        }]),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Ignoring incompatible QMutBench checkpoint"):
        report = runner.run_qmut()

    assert len(calls) == 1
    assert calls[0][1] == {"shots": runner.QMUT_SHOTS, "seed": runner.QMUT_SEED, "threshold": runner.QMUT_THRESHOLD}
    assert report["records"][0]["support_status"] == "UNSUPPORTED"
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stored["checkpoint_metadata"]["checkpoint_schema_version"] == runner.CHECKPOINT_SCHEMA_VERSION
    assert stored["checkpoint_metadata"]["dataset_case_ids"] == ["AddGate_rxx_inGap_1_"]
    assert stored["results"][0]["support_status"] == "UNSUPPORTED"
    assert stored["run_provenance"] == {
        "status": "COMPLETED",
        "reused_case_ids": [],
        "fresh_case_ids": ["AddGate_rxx_inGap_1_"],
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("checkpoint_schema_version", -1), ("runner_sources", {"older-runner.py": "old-hash"})],
)
def test_checkpoint_from_incompatible_runner_or_schema_is_recomputed(monkeypatch, tmp_path, field, replacement):
    dataset = _dataset()
    calls = _configure_runner(monkeypatch, tmp_path, dataset, _result())
    checkpoint = tmp_path / "reports" / "qmutbench_full_checkpoint.json"
    metadata = runner._qmut_checkpoint_metadata(dataset)
    metadata[field] = replacement
    runner._write_qmut_checkpoint(checkpoint, metadata, {
        "AddGate_rxx_inGap_1_": {
            "case_id": "AddGate_rxx_inGap_1_",
            "status": "OK",
            "support_status": "SUPPORTED",
            "execution_status": "COMPLETED",
            "diagnosed": True,
            "system_diagnosis": {"category": "PROGRAM_FAULT"},
        }
    }, {"status": "COMPLETED", "reused_case_ids": [], "fresh_case_ids": ["AddGate_rxx_inGap_1_"]})

    with pytest.warns(RuntimeWarning, match="Ignoring incompatible QMutBench checkpoint"):
        report = runner.run_qmut()

    assert len(calls) == 1
    assert report["records"][0]["support_status"] == "UNSUPPORTED"


def test_matching_checkpoint_is_reused(monkeypatch, tmp_path):
    dataset = _dataset()
    calls = _configure_runner(monkeypatch, tmp_path, dataset, _result())
    checkpoint = tmp_path / "reports" / "qmutbench_full_checkpoint.json"
    metadata = runner._qmut_checkpoint_metadata(dataset)
    cached = {
        "case_id": "AddGate_rxx_inGap_1_",
        "status": "OK",
        "support_status": "SUPPORTED",
        "execution_status": "COMPLETED",
        "diagnosed": True,
        "predicted_category": "PROGRAM_FAULT",
        "system_diagnosis": {"category": "PROGRAM_FAULT"},
        "ground_truth_category": None,
        "fault_category": "AddGate",
        "mutation_type": "AddGate",
        "ground_truth_status": "MISSING",
        "diagnosis_ground_truth_compatible": False,
        "errors": [],
    }
    runner._write_qmut_checkpoint(
        checkpoint,
        metadata,
        {cached["case_id"]: cached},
        {"status": "COMPLETED", "reused_case_ids": [], "fresh_case_ids": [cached["case_id"]]},
    )

    report = runner.run_qmut()

    assert calls == []
    assert report["records"][0]["support_status"] == "SUPPORTED"
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stored["run_provenance"] == {
        "status": "COMPLETED",
        "reused_case_ids": ["AddGate_rxx_inGap_1_"],
        "fresh_case_ids": [],
    }
