import json

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from experiments import qmutbench_selected_pipeline as pipeline


def _source(path):
    rows = []
    for group in range(10):
        for i, label in enumerate(("Survivor", "WOO", "OPO")):
            rows.append({
                "Unnamed: 0": len(rows), "Origin_program": f"origin_{group}",
                "algorithm": f"algo_{group % 3}", "qubits": 3 + group % 4,
                "gates": 10 + group, "operation": ("Add", "Replace", "Remove")[i],
                "Position_int": 1 + i, "Killed": label, "gate": f"gate_{i}",
                "algorithm_group": f"family_{group % 3}", "dominant_state": f"state_{group % 2}",
                "position": "0-10%",
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def test_prepare_is_group_disjoint_chunked_and_reuses_verified_chunks(tmp_path, monkeypatch):
    data = tmp_path / "source.csv"
    frame = _source(data)
    monkeypatch.setattr(pipeline, "EXPECTED_SHA256", pipeline.sha256_file(data))
    monkeypatch.setattr(pipeline, "group_holdout", lambda y, groups, **kwargs: (
        [i for i, g in enumerate(groups) if g != "origin_9"],
        [i for i, g in enumerate(groups) if g == "origin_9"],
    ))
    out = tmp_path / "selected"
    result = pipeline.prepare(data, out, target_rows=8, chunksize=9, seed=7)
    assert result["selected_rows"] >= 8
    assert result["selected_origin_groups"] < 9
    assert not set(result["selected_groups"]) & set(result["frozen_test_groups"])
    assert all(info["verified"] for info in result["chunks"].values())
    valid_chunk = out / "chunks" / "chunk_00000.csv"
    original_hash = pipeline.sha256_file(valid_chunk)
    # Completed, valid chunks are reused; damage is detected and repaired.
    valid_chunk.write_text("damaged", encoding="utf-8")
    resumed = pipeline.prepare(data, out, target_rows=8, chunksize=9, seed=7)
    assert pipeline.sha256_file(valid_chunk) == original_hash
    assert resumed["selected_rows"] == result["selected_rows"]
    assignments = pd.read_csv(out / "selection_assignments.csv")
    selected = assignments[assignments["split"] == "selected_development"]
    assert selected.record_id.is_unique
    assert set(selected.origin_program) <= set(result["selected_groups"])
    assert json.loads((out / "manifest.json").read_text())["schema"] == pipeline.SCHEMA


def test_status_reports_corrupt_chunks(tmp_path, monkeypatch):
    data = tmp_path / "source.csv"
    _source(data)
    monkeypatch.setattr(pipeline, "EXPECTED_SHA256", pipeline.sha256_file(data))
    monkeypatch.setattr(pipeline, "group_holdout", lambda y, groups, **kwargs: (
        [i for i, g in enumerate(groups) if g != "origin_9"],
        [i for i, g in enumerate(groups) if g == "origin_9"],
    ))
    out = tmp_path / "selected"
    pipeline.prepare(data, out, target_rows=8, chunksize=9)
    (out / "chunks" / "chunk_00000.csv").write_text("partial", encoding="utf-8")
    result = pipeline.status(out)
    assert "chunk_00000.csv" in result["remaining_or_corrupt_chunks"]


def test_extra_trees_candidate_uses_the_requested_estimator():
    assert isinstance(pipeline._model("extra_trees", 12).named_steps["model"], ExtraTreesClassifier)


def test_interruption_after_checkpoint_reuses_completed_chunk_and_rejects_new_config(tmp_path, monkeypatch):
    data = tmp_path / "source.csv"
    _source(data)
    monkeypatch.setattr(pipeline, "EXPECTED_SHA256", pipeline.sha256_file(data))
    monkeypatch.setattr(pipeline, "group_holdout", lambda y, groups, **kwargs: (
        [i for i, g in enumerate(groups) if g != "origin_9"],
        [i for i, g in enumerate(groups) if g == "origin_9"],
    ))
    out = tmp_path / "selected"
    write_manifest = pipeline._atomic_json
    interrupted = {"value": False}

    def interrupt_after_first_checkpoint(path, value):
        write_manifest(path, value)
        if path.name == "manifest.json" and value.get("chunks") and not interrupted["value"]:
            interrupted["value"] = True
            raise KeyboardInterrupt("simulated power loss after durable checkpoint")

    monkeypatch.setattr(pipeline, "_atomic_json", interrupt_after_first_checkpoint)
    try:
        pipeline.prepare(data, out, target_rows=8, chunksize=9, seed=9)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("simulated interruption did not interrupt preparation")
    completed = out / "chunks" / "chunk_00000.csv"
    saved_hash = pipeline.sha256_file(completed)
    original_csv = pipeline._atomic_csv
    rewritten = []

    def track_csv(frame, path):
        if path == completed:
            rewritten.append(path)
        original_csv(frame, path)

    monkeypatch.setattr(pipeline, "_atomic_csv", track_csv)
    monkeypatch.setattr(pipeline, "_atomic_json", write_manifest)
    resumed = pipeline.prepare(data, out, target_rows=8, chunksize=9, seed=9)
    assert not rewritten
    assert pipeline.sha256_file(completed) == saved_hash
    assert resumed["status"] == "prepared"
    try:
        pipeline.prepare(data, out, target_rows=12, chunksize=9, seed=9)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("resume accepted a changed selection configuration")
