import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generate_phase5_training_rows import (
    ALLOWED_FAMILIES,
    ALLOWED_SUFFIXES,
    EXPECTED_LABEL,
    _expected_config_keys,
    _profile,
    validate_training_records,
)
from experiments.phase5_benchmark_infrastructure import SEEDS, SHOT_COUNTS, build_ledger, validate_calibration


def _records():
    records = []
    for family, suffix, seed, shots in sorted(_expected_config_keys()):
        label = EXPECTED_LABEL.get(suffix, "HARDWARE_ANOMALY")
        records.append({"family_id": family, "suffix": suffix,
                        "case_id": f"{family}-{suffix}", "seed": seed,
                        "shots": shots, "label": label})
    return records


def test_development_grid_count_suffixes_labels_and_partitions_are_frozen():
    records = _records()
    validate_training_records(records)
    assert ALLOWED_FAMILIES == {"F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"}
    assert len(ALLOWED_SUFFIXES) == 11 and "C15" not in ALLOWED_SUFFIXES
    assert len(SEEDS) == 5 and len(SHOT_COUNTS) == 3
    assert len(records) == 330
    assert {label: sum(row["label"] == label for row in records)
            for label in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")} == {
                "NORMAL": 30, "PROGRAM_FAULT": 60, "HARDWARE_ANOMALY": 240,
            }


def test_all_selected_case_profiles_cover_both_circuits():
    cases = [case for case in build_ledger()
             if case.family_id in ALLOWED_FAMILIES and case.suffix in ALLOWED_SUFFIXES]
    assert len(cases) == 22
    for case in cases:
        validate_calibration(_profile(case), case.expected_circuit, case.observed_circuit, case=case)


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.append({**rows[0], "family_id": "F3_SWAP_ROUTING", "case_id": "F3_SWAP_ROUTING-N0"}), "only F1/F2"),
    (lambda rows: rows.append({**rows[0], "suffix": "C15", "case_id": "F1_GHZ_CHAIN-C15"}), "excluded or unsupported"),
    (lambda rows: rows.append(rows[0]), "duplicate"),
    (lambda rows: rows.pop(), "missing"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "label": "NOT_A_LABEL"}), "label is invalid"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "case_id": "F2_TOFFOLI_COMPUTE-N0"}), "does not match"),
])
def test_training_grid_rejects_leakage_and_corruption(mutation, message):
    records = _records()
    mutation(records)
    with pytest.raises(ValueError, match=message):
        validate_training_records(records)
