from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from prepare_source_isolated_task import prepare  # noqa: E402


FIELDS = ["record_id", "bug_id", "label", "code", "source", "algorithm", "size", "histogram_features"]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path: Path):
    source = tmp_path / "input.csv"
    folds = tmp_path / "folds.csv"
    rows = [
        {"record_id": "a-aer", "bug_id": "g1", "label": 0, "code": " feature ", "source": "aer_simulator", "algorithm": "a", "size": 1, "histogram_features": json.dumps({"entropy": .1, "max_probability": .9, "support_fraction": .4, "mean_hamming_weight": .3, "hamming_variance": .1, "all_ones_probability": .2, "bits": 2, "shots": 20})},
        {"record_id": "a-kyiv", "bug_id": "g1", "label": 1, "code": "feature2", "source": "fake_kyiv", "algorithm": "a", "size": 1, "histogram_features": json.dumps({"entropy": .1, "max_probability": .9, "support_fraction": .4, "mean_hamming_weight": .3, "hamming_variance": .1, "all_ones_probability": .2, "bits": 2, "shots": 20})},
        {"record_id": "b-aer", "bug_id": "g2", "label": 0, "code": "feature", "source": "aer_simulator", "algorithm": "b", "size": 1, "histogram_features": json.dumps({"entropy": .1, "max_probability": .9, "support_fraction": .4, "mean_hamming_weight": .3, "hamming_variance": .1, "all_ones_probability": .2, "bits": 2, "shots": 20})},
        {"record_id": "bad", "bug_id": "g3", "label": 1, "code": "feature", "source": "aer_simulator", "algorithm": "c", "size": 1, "histogram_features": json.dumps({"entropy": .1, "max_probability": .9, "support_fraction": .4, "mean_hamming_weight": .3, "hamming_variance": .1, "all_ones_probability": .2, "bits": 2, "shots": 20})},
    ]
    write_csv(source, FIELDS, rows)
    write_csv(folds, ["record_id", "group_id", "fold", "model"], [
        {"record_id": "a-aer", "group_id": "g1", "fold": 1, "model": "amplitude_hybrid_3q_depth2"},
        {"record_id": "a-kyiv", "group_id": "g1", "fold": 1, "model": "amplitude_hybrid_3q_depth2"},
        {"record_id": "b-aer", "group_id": "g2", "fold": 2, "model": "amplitude_hybrid_3q_depth2"},
        {"record_id": "bad", "group_id": "g3", "fold": 2, "model": "amplitude_hybrid_3q_depth2"},
    ])
    return source, folds, tmp_path / "processed"


def test_chunking_validation_rejections_and_group_folds(tmp_path):
    source, folds, output = fixture(tmp_path)
    report = prepare(source, folds, output, chunk_size=2)
    assert report["counts"] == {"input": 4, "accepted": 3}
    assert report["rejection_reason_counts"] == {"inconsistent_source_label": 1}
    assert report["validation_pass"] is False
    assert report["groups"] == 2
    assert report["group_cross_fold"] is False
    assert [chunk["rows"] for chunk in report["partitions"]] == [2, 1]
    rejected = list(csv.DictReader((output / "rejected/rejection_references.csv").open(encoding="utf-8")))
    assert rejected[0]["reason_codes"] == "inconsistent_source_label"
    split = list(csv.DictReader((output / "manifests/group_folds.csv").open(encoding="utf-8")))
    assert len(split) == 3


def test_valid_task_is_idempotent_and_incompatible_input_refuses_resume(tmp_path):
    source, folds, output = fixture(tmp_path)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))[:3]
    write_csv(source, FIELDS, rows)
    fold_rows = list(csv.DictReader(folds.open(encoding="utf-8")))[:3]
    write_csv(folds, list(fold_rows[0]), fold_rows)
    first = prepare(source, folds, output, chunk_size=2)
    second = prepare(source, folds, output, chunk_size=2)
    assert first["validation_pass"] and second["validation_pass"]
    assert first["partitions"] == second["partitions"]
    rows[0]["code"] = "changed"
    write_csv(source, FIELDS, rows)
    with pytest.raises(RuntimeError, match="incompatible"):
        prepare(source, folds, output, chunk_size=2)


def test_group_crossing_folds_is_rejected(tmp_path):
    source, folds, output = fixture(tmp_path)
    fold_rows = list(csv.DictReader(folds.open(encoding="utf-8")))
    fold_rows[1]["fold"] = "2"
    # Keep this row valid so the cross-fold group violation is independently detected.
    input_rows = list(csv.DictReader(source.open(encoding="utf-8")))
    input_rows[-1]["bug_id"] = "g1"
    input_rows[-1]["label"] = "0"
    input_rows[-1]["source"] = "aer_simulator"
    write_csv(source, FIELDS, input_rows)
    write_csv(folds, list(fold_rows[0]), fold_rows)
    report = prepare(source, folds, output, chunk_size=10)
    assert report["group_cross_fold"] is True
    assert report["rejection_reason_counts"]["group_crosses_folds"] == 2


def test_duplicate_ids_are_rejected_and_counted_even_when_chunked(tmp_path):
    source, folds, output = fixture(tmp_path)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))[:3]
    rows.append(dict(rows[0]))
    write_csv(source, FIELDS, rows)
    fold_rows = list(csv.DictReader(folds.open(encoding="utf-8")))[:3]
    write_csv(folds, list(fold_rows[0]), fold_rows)
    report = prepare(source, folds, output, chunk_size=2)
    assert report["duplicate_record_ids"] == 1
    assert report["rejection_reason_counts"]["duplicate_record_id"] == 1
    assert report["counts"]["accepted"] == 3


def test_numeric_ranges_are_enforced(tmp_path):
    source, folds, output = fixture(tmp_path)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))[:3]
    features = json.loads(rows[0]["histogram_features"])
    features["entropy"] = 4
    rows[0]["histogram_features"] = json.dumps(features)
    write_csv(source, FIELDS, rows)
    fold_rows = list(csv.DictReader(folds.open(encoding="utf-8")))[:3]
    write_csv(folds, list(fold_rows[0]), fold_rows)
    report = prepare(source, folds, output, chunk_size=2)
    assert report["rejection_reason_counts"]["numeric_out_of_range:histogram_features.entropy"] == 1
