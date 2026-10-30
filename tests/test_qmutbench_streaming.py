"""Synthetic contract tests for the disk-indexed QMutBench development stream."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmutbench_qml.data.loader import QASM_CATEGORICAL, QASM_NUMERIC
from qmutbench_qml.data.streaming import DevelopmentStream, sha256
from qmutbench_qml.training import run_binary_development


def _fixture(root: Path, *, duplicate_raw=False, archive_fold_mismatch=False, missing_raw=False):
    raw = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    ref = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    archive = root / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    cache = root / "data/qmutbench_ml/qasm_mutation_diff_v1"
    for path in (raw, ref, archive):
        path.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    ids = np.arange(1, 7, dtype=np.int64)
    ys = np.array([0, 1, 1, 1, 0, 1], dtype=np.int8)
    folds = np.array([1, 1, 2, 2, 3, 3], dtype=np.int8)
    origins = ["origin-a", "origin-a", "origin-b", "origin-b", "origin-c", "origin-c"]
    archive_folds = folds.copy()
    if archive_fold_mismatch:
        archive_folds[0] = 2
    np.savez(archive, record_id=ids, y=ys, fold=archive_folds)
    with ref.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "origin_program", "y_true", "fold"])
        for rid, origin, y, fold in zip(ids, origins, ys, folds):
            writer.writerow([rid, origin, int(y), int(fold)])
    fields = ["", "Origin_program", "Killed", "algorithm", "operation", "gate",
              "dominant_state", "qubits", "gates", "Position_int", "position"]
    records = [(0, "frozen-secret", "PRIVATE", "must-not-be-used"),
        (1, "origin-a", "Survivor", "a"), (2, "origin-a", "WOO", "a"),
        (3, "origin-b", "WOO", "b"), (4, "origin-b", "OPO", "b"),
        (5, "origin-c", "Survivor", "c"), (6, "origin-c", "WOO", "c")]
    if missing_raw:
        records = records[:-1]
    if duplicate_raw:
        records.append(records[1])
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for rid, origin, label, algorithm in records:
            writer.writerow([rid, origin, label, algorithm, "op", "gate", "state", 2, 3, 1, "0-10%"])
    raw_sha = sha256(raw)
    chunks = []
    for ix, start in enumerate((0, 3)):
        rows = []
        for rid in ids[start:start + 3]:
            row = {"record_id": int(rid)}
            row.update({name: float(rid) for name in QASM_NUMERIC})
            row.update({name: "category-a" for name in QASM_CATEGORICAL})
            rows.append(row)
        path = cache / f"diff_{ix:05d}.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
        chunks.append({"file": path.name, "rows": len(rows), "errors": 0, "sha256": sha256(path)})
    (cache / "manifest.json").write_text(json.dumps({"status": "complete",
        "qasm_commit": "01f5e7700c051fe715fc368b307deaa7e6869ba8", "source_sha256": raw_sha,
        "feature_rows": 6, "error_rows": 0, "chunks": chunks}), encoding="utf-8")
    return root


def _stream(root, tmp_path, **kwargs):
    expected_source_rows = kwargs.pop("expected_source_rows", 7)
    return DevelopmentStream(root, tmp_path / "index.sqlite", expected_rows=6,
        expected_origins=3, expected_source_rows=expected_source_rows, expected_source_sha256=None, metadata_chunk_rows=2,
        feature_chunk_rows=2, **kwargs)


def test_stream_joins_all_development_rows_and_never_retains_frozen_row(tmp_path):
    root = _fixture(tmp_path / "project")
    stream = _stream(root, tmp_path)
    audit = stream.build_index()
    assert audit["development_rows_indexed"] == 6
    assert audit["frozen_test_rows_retained"] == 0
    assert audit["source_rows_scanned"] == 7
    valid = pd.concat(list(stream.iter_fold(1, "validation")), ignore_index=True)
    train = pd.concat(list(stream.iter_fold(1, "training")), ignore_index=True)
    assert set(valid.record_id) == {1, 2}
    assert set(train.record_id) == {3, 4, 5, 6}
    assert set(valid.record_id).isdisjoint(train.record_id)
    assert 0 not in set(valid.record_id) | set(train.record_id)
    assert stream.last_stream_audit["unique_ids"] == 4
    assert stream.last_stream_audit["skipped_rows"] == 0


def test_archive_authoritative_mode_uses_five_fold_development_reference_without_csv_labels(tmp_path):
    root = _fixture(tmp_path / "project")
    # The older CSV is intentionally changed; archive-authoritative mode uses
    # only IDs/targets/folds in the registered development NPZ and source groups.
    ref = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    rows = list(csv.reader(ref.open(encoding="utf-8")))
    rows[1][2] = "1" if rows[1][2] == "0" else "0"
    with ref.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    stream = _stream(root, tmp_path, archive_authoritative=True)
    audit = stream.build_index()
    assert audit["development_rows_indexed"] == 6
    assert audit["reference_authority"] == "numeric five-fold development archive"
    assert audit["frozen_test_rows_retained"] == 0
    assert audit["fold_origin_counts"] == {"1": 1, "2": 1, "3": 1, "4": 0, "5": 0}


def test_streamed_feature_values_do_not_depend_on_chunk_boundaries(tmp_path):
    root = _fixture(tmp_path / "project")
    stream = _stream(root, tmp_path)
    stream.build_index()
    first = pd.concat(list(stream.iter_fold(2, "training")), ignore_index=True).sort_values("record_id").reset_index(drop=True)
    stream.feature_chunk_rows = 1
    second = pd.concat(list(stream.iter_fold(2, "training")), ignore_index=True).sort_values("record_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_synthetic_stream_preprocessing_writes_only_bounded_memmap_chunks(tmp_path):
    from qmutbench_qml.training.run_full_streaming import _fit_encoder, _write_memmaps
    root = _fixture(tmp_path / "project")
    stream = _stream(root, tmp_path)
    stream.build_index()
    encoder, fitted = _fit_encoder(stream, fold=1, n_features=2, seed=17, fit_rows=4)
    assert fitted == 3  # Exhaustive train-only class-balanced sample when one class has one row.
    train = _write_memmaps(stream, 1, "training", encoder, tmp_path, "train", rows=4, qubits=2)
    valid = _write_memmaps(stream, 1, "validation", encoder, tmp_path, "valid", rows=2, qubits=2)
    tx = np.memmap(train[0], mode="r", dtype=np.float64, shape=(4, 2))
    vx = np.memmap(valid[0], mode="r", dtype=np.float64, shape=(2, 2))
    assert tx.shape == (4, 2) and vx.shape == (2, 2)
    assert train[-1]["records_seen"] == 4 and valid[-1]["records_seen"] == 2


def test_stream_rejects_duplicate_source_rows_and_split_mismatch(tmp_path):
    root = _fixture(tmp_path / "duplicate", duplicate_raw=True)
    with pytest.raises(Exception):
        _stream(root, tmp_path / "duplicate-db").build_index()
    root = _fixture(tmp_path / "mismatch", archive_fold_mismatch=True)
    with pytest.raises(ValueError, match="disagree"):
        _stream(root, tmp_path / "mismatch-db").build_index()


def test_stream_rejects_origin_fold_leakage_and_incomplete_reference(tmp_path):
    root = _fixture(tmp_path / "leak")
    ref = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    rows = list(csv.reader(ref.open(encoding="utf-8")))
    rows[2][1] = "origin-a"
    rows[2][3] = "2"
    with ref.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="crosses reference folds"):
        _stream(root, tmp_path / "leak-db").build_index()
    root = _fixture(tmp_path / "missing", missing_raw=True)
    with pytest.raises(ValueError, match="reconciliation failed"):
        _stream(root, tmp_path / "missing-db", expected_source_rows=6).build_index()


def test_cli_keeps_sampled_pilot_and_full_development_modes_distinct(capsys):
    assert run_binary_development.main(["--mode", "sampled-pilot", "--validate-config"]) == 0
    sampled = json.loads(capsys.readouterr().out)
    assert sampled["mode"] == "sampled-pilot"
    assert sampled["configuration"]["chunk_batches"] == 8
    assert run_binary_development.main(["--mode", "full-development", "--validate-config"]) == 0
    full = json.loads(capsys.readouterr().out)
    assert full["mode"] == "full-development"
    assert full["configuration"]["chunk_batches"] == 128


def test_completed_fold_oof_resume_artifact_is_checksum_and_split_validated(tmp_path):
    from qmutbench_qml.training.run_full_streaming import _fold_metrics, _validate_saved_oof
    root = _fixture(tmp_path / "project")
    stream = _stream(root, tmp_path)
    stream.build_index()
    path = tmp_path / "fold.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["record_id", "origin_program", "fold", "y_true", "model", "y_pred", "prob_0", "prob_1"])
        writer.writeheader()
        writer.writerows([
            {"record_id": 1, "origin_program": "origin-a", "fold": 1, "y_true": 0, "model": "binary_vqc", "y_pred": 0, "prob_0": .9, "prob_1": .1},
            {"record_id": 2, "origin_program": "origin-a", "fold": 1, "y_true": 1, "model": "binary_vqc", "y_pred": 1, "prob_0": .1, "prob_1": .9},
        ])
    metrics = _fold_metrics(np.array([[1, 0], [0, 1]]), .02, 2)
    metrics["oof_predictions_sha256"] = sha256(path)
    assert _validate_saved_oof(path, 1, stream, metrics, 2)
    path.write_text(path.read_text().replace("0.9", "0.8"), encoding="utf-8")
    assert not _validate_saved_oof(path, 1, stream, metrics, 2)
