import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/qbugs_qasm_v2_candidate_sweep_20261030"
MANIFEST = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qbugs_qasm_v2/normalized_manifest.csv"
SPLIT = ROOT / "data/processed/qbugs_hornbro/correct_buggy_circuit_state_qasm_v2/manifests/group_folds.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qbugs_candidate_sweep_has_matching_source_and_split_hashes():
    campaign = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert campaign["status"] == "VERIFIED"
    assert campaign["source_hashes"]["manifest"] == sha256(MANIFEST)
    assert campaign["source_hashes"]["split"] == sha256(SPLIT)
    assert campaign["rows"] == 1496
    assert campaign["groups"] == 748
    assert campaign["frozen_test_accessed"] is False


def test_qbugs_candidate_oof_and_checkpoints_cover_all_grouped_folds():
    campaign = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    source = {row["record_id"]: row for row in csv.DictReader(
        MANIFEST.open(encoding="utf-8-sig", newline=""))}
    for model in campaign["models"]:
        model_dir = OUT / model["configuration"]["name"]
        predictions = list(csv.DictReader((model_dir / "oof_predictions.csv").open(
            encoding="utf-8-sig", newline="")))
        assert len(predictions) == len(source) == 1496
        assert {row["record_id"] for row in predictions} == set(source)
        group_folds = defaultdict(set)
        for row in predictions:
            assert int(row["y_true"]) == int(source[row["record_id"]]["label"])
            assert row["group_id"] == source[row["record_id"]]["bug_id"]
            group_folds[row["group_id"]].add(row["fold"])
        assert all(len(folds) == 1 for folds in group_folds.values())
        run_manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert sha256(model_dir / "oof_predictions.csv") == run_manifest["oof_sha256"]
        assert all(sha256(model_dir / f"fold_{fold}" / "checkpoint.npz") == digest
                   for fold, digest in run_manifest["checkpoint_hashes"].items())
        assert len(run_manifest["checkpoint_hashes"]) == 5
