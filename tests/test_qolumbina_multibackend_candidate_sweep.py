from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_qolumbina_multibackend_candidate_sweep import (  # noqa: E402
    OUTPUT,
    TASK_NAME,
    validate_outputs,
    verify_inputs,
)


def test_qolumbina_candidate_inputs_match_pinned_target_and_group_split():
    audit = verify_inputs()
    assert audit["manifest_rows"] == 546
    assert audit["groups"] == 26
    assert audit["folds"] == 5
    assert audit["label_counts"] == {"1": 506, "0": 40}
    assert audit["cross_fold_groups"] == 0
    assert audit["frozen_test_accessed"] is False


def test_qolumbina_candidate_outputs_cover_each_validated_row_and_checkpoint():
    audit = verify_inputs()
    result = validate_outputs(audit, OUTPUT)
    assert result["prediction_rows"] == 1092
    assert len(result["model_names"]) == 2
    assert all(len(folds) == 5 for folds in result["checkpoint_hashes"].values())
    assert TASK_NAME == "Qolumbina multibackend"
