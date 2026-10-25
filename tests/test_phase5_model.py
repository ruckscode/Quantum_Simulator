import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase5_model import train_phase5_artifact
from quantum_simulator.ml_diagnosis import CLASSES, FEATURE_NAMES
from quantum_simulator.observable_model_adapter import PortableLogisticRegressionAdapter


def _rows():
    rows = []
    for family in ("F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"):
        for index, label in enumerate(CLASSES):
            base = float(index * 2)
            rows.append({
                "family_id": family,
                "features": {name: base + offset * .1 for offset, name in enumerate(FEATURE_NAMES)},
                "label": label,
            })
    return rows


def test_training_fails_closed_on_non_development_family(tmp_path):
    rows = _rows()
    rows.append({**rows[0], "family_id": "F3_SWAP_ROUTING"})
    with pytest.raises(ValueError, match="only F1/F2"):
        train_phase5_artifact(rows, tmp_path / "model.json")


def test_f1_f2_training_exports_roundtrippable_uncalibrated_model(tmp_path):
    path = tmp_path / "model.json"
    artifact = train_phase5_artifact(_rows(), path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == artifact
    assert saved["feature_names"] == list(FEATURE_NAMES)
    assert saved["class_names"] == list(CLASSES)
    assert saved["probabilities_calibrated"] is False
    assert saved["probability_calibration"] == "none"
    assert saved["training_status"] == "DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT"
    assert saved["confidence_policy"]["cutoff"] == .60
    assert saved["metadata"]["training_families"] == ["F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"]
    assert saved["metadata"]["excluded_families"] == ["F3_SWAP_ROUTING", "F4_PARITY_TREE", "F5_PHASE_LAYER"]
    assert saved["metadata"]["deployment_validation"] == "NOT_PERFORMED"

    model = PortableLogisticRegressionAdapter(path)
    assert model.deployment_validated is False
    prediction = model.predict(_rows()[0]["features"])
    assert tuple(prediction.class_probabilities) == CLASSES
    assert sum(prediction.class_probabilities.values()) == pytest.approx(1.0)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_phase5_artifact(_rows(), path)


def test_stale_temporary_file_does_not_block_artifact_export_or_get_removed(tmp_path):
    path = tmp_path / "model.json"
    stale = tmp_path / "model.json.tmp"
    stale.write_text("preserve unrelated stale artifact", encoding="utf-8")

    artifact = train_phase5_artifact(_rows(), path)

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert stale.read_text(encoding="utf-8") == "preserve unrelated stale artifact"
    assert set(item.name for item in tmp_path.iterdir()) == {"model.json", "model.json.tmp"}


def test_training_rejects_wrong_feature_order_and_multilabel_rows(tmp_path):
    rows = _rows()
    rows[0]["features"] = dict(reversed(tuple(rows[0]["features"].items())))
    with pytest.raises(ValueError, match="names/order"):
        train_phase5_artifact(rows, tmp_path / "bad-order.json")

    rows = _rows()
    rows[0]["label"] = ("PROGRAM_FAULT", "HARDWARE_ANOMALY")
    with pytest.raises(ValueError, match="single-cause"):
        train_phase5_artifact(rows, tmp_path / "multilabel.json")
