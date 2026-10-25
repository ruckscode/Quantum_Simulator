import json

import pandas as pd
import pytest

import qmutbench_ml_pipeline as pipeline


def test_qmutbench_default_sources_are_project_relative():
    import qmutbench_ml_extract as extractor

    expected = pipeline.ROOT / "data" / "qmutbench_full" / "all_data.csv"
    assert pipeline.SOURCE_CSV == expected
    assert extractor.SOURCE_CSV == expected


def _frame(size=30):
    return pd.DataFrame(
        [
            {
                "qmutbench_row_index": i,
                "Origin_program": f"circuit_{i % 3}",
                "algorithm": f"circuit_{i % 3}",
                "qubits": 1,
                "gates": 1,
                "operation": "Remove",
                "gate": "X",
                "Position_int": i,
                "Killed": "True",
            }
            for i in range(size)
        ]
    )


def _stub(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "MAX_SIMULATOR_QUBITS", 4)
    monkeypatch.setattr(pipeline, "retrieve_qasm", lambda rel: tmp_path / "fake.qasm")
    monkeypatch.setattr(pipeline, "_read_parse", lambda path: ([], 1))
    monkeypatch.setattr(pipeline, "_run_and_trace", lambda *args: [{"probability_difference": 0.0}])


def test_interrupted_chunk_resumes_without_duplicate_originals(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    frame = _frame()
    checkpoint = tmp_path / "chunk_checkpoint.json"
    identity = {"source_csv_sha256": "abc", "chunk_size": 30, "max_simulator_qubits": 4}
    actual_trace = pipeline._run_and_trace
    calls = 0

    def interrupt_once(*args):
        nonlocal calls
        calls += 1
        if calls == 29:
            raise KeyboardInterrupt()
        return actual_trace(*args)

    monkeypatch.setattr(pipeline, "_run_and_trace", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        pipeline.process_records(frame, checkpoint_path=checkpoint, checkpoint_metadata=identity)
    assert json.loads(checkpoint.read_text())["next_position"] == 25

    monkeypatch.setattr(pipeline, "_run_and_trace", actual_trace)
    records, failures, summary = pipeline.process_records(
        frame, checkpoint_path=checkpoint, checkpoint_metadata=identity
    )
    assert len(failures) == 0
    assert summary["selected_records"] == 30
    originals = [r for r in records if r["record_type"] == "original"]
    mutants = [r for r in records if r["record_type"] == "mutant"]
    assert len(originals) == 3
    assert len(mutants) == 30


def test_stale_checkpoint_is_rejected(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    frame = _frame(1)
    checkpoint = tmp_path / "stale.json"
    checkpoint.write_text(json.dumps({"source_csv_sha256": "old", "row_indices": [0], "next_position": 1, "records": [], "failures": []}))
    records, _, _ = pipeline.process_records(
        frame,
        checkpoint_path=checkpoint,
        checkpoint_metadata={"source_csv_sha256": "new", "chunk_size": 1, "max_simulator_qubits": 4},
    )
    assert len(records) == 2


def test_invalid_checkpoint_is_rejected(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    frame = _frame(1)
    checkpoint = tmp_path / "invalid.json"
    checkpoint.write_text("not json")
    records, _, _ = pipeline.process_records(frame, checkpoint_path=checkpoint)
    assert len(records) == 2


def test_completed_chunk_is_skipped(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_source_digest", lambda: "abc")
    monkeypatch.setattr(pipeline, "_merge_chunks", lambda *args: None)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    prefix = "chunk_00000"
    (chunk_dir / f"{prefix}_results.csv").write_text("record_type\n")
    (chunk_dir / f"{prefix}_failures.csv").write_text("failure_type\n")
    manifest = {
        "source_csv_sha256": "abc",
        "row_indices": [0],
        "chunk_size": 1,
        "max_simulator_qubits": 4,
    }
    (chunk_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(pipeline, "process_records", lambda *a, **k: pytest.fail("completed chunk was reprocessed"))
    pipeline.process_chunks(_frame(1), chunk_size=1, batch_size=1, resume=True)


def test_final_chunk_outputs_merge(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("source")
    monkeypatch.setattr(pipeline, "SOURCE_CSV", source)
    monkeypatch.setattr(pipeline, "_source_digest", lambda: "abc")

    def fake_process(frame, **kwargs):
        row = frame.iloc[0]
        base = pipeline._row_metadata(row, "original", "NORMAL", "org.qasm", None)
        mutant = pipeline._row_metadata(row, "mutant", "PROGRAM_FAULT", "org.qasm", "mut.qasm")
        features = {name: 0.0 for name in pipeline.FEATURE_COLUMNS}
        return [pipeline._record_output(base, features), pipeline._record_output(mutant, features)], [], {
            "selected_records": len(frame),
            "successful_mutant_executions": 1,
            "successful_original_executions": 1,
            "simulator_limit_failures": 0,
        }

    monkeypatch.setattr(pipeline, "process_records", fake_process)
    pipeline.process_chunks(_frame(1), chunk_size=1, batch_size=1, resume=False)
    merged = pd.read_csv(tmp_path / "qmutbench_results.csv")
    assert list(merged.record_type) == ["original", "mutant"]
    assert (tmp_path / "qmutbench_manifest.json").is_file()
