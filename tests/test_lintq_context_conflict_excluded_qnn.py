import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/lintq_context_v2_conflict_excluded_20261029"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lintq_conflicting_context_component_is_explicitly_rejected():
    source = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/lintq_context_v2/normalized_manifest.csv"
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    accepted = list(csv.DictReader((OUT / "accepted_manifest.csv").open(encoding="utf-8-sig", newline="")))
    rejected = list(csv.DictReader((OUT / "rejected_conflict_component.csv").open(encoding="utf-8-sig", newline="")))
    assert manifest["status"] == "completed_grouped_qnn_oof"
    assert manifest["source_manifest_sha256"] == sha256(source)
    assert len(accepted) == 287 and len(rejected) == 3
    assert {row["record_id"] for row in rejected} == {
        "lintq-warning-88", "lintq-warning-99", "lintq-warning-101"}
    assert {row["rejection_reason"] for row in rejected} == {
        "byte_identical_model_input_has_conflicting_manual_labels"}
    assert len({row["label"] for row in rejected}) == 2


def test_lintq_conflict_excluded_oof_is_complete_and_group_disjoint():
    accepted = {row["record_id"]: row for row in csv.DictReader(
        (OUT / "accepted_manifest.csv").open(encoding="utf-8-sig", newline=""))}
    predictions = list(csv.DictReader((OUT / "oof_predictions.csv").open(encoding="utf-8-sig", newline="")))
    groups = defaultdict(set)
    assert len(predictions) == len(accepted) == 287
    assert {row["record_id"] for row in predictions} == set(accepted)
    for row in predictions:
        assert int(row["y_true"]) == int(accepted[row["record_id"]]["label"])
        assert row["group_id"] == accepted[row["record_id"]]["model_group_id"]
        groups[row["group_id"]].add(row["fold"])
    assert all(len(folds) == 1 for folds in groups.values())
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert all(sha256(OUT / "angle_reupload_4q_depth2_lr0035" / f"fold_{fold}" / "checkpoint.npz") == digest
               for fold, digest in manifest["checkpoint_hashes"].items())
    assert manifest["frozen_test_accessed"] is False
