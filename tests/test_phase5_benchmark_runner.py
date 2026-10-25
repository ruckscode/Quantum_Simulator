from __future__ import annotations

import copy
from functools import lru_cache
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import phase5_benchmark_infrastructure as p5
from experiments.phase5_benchmark_runner import (
    SPECIFICITY_CASES,
    F3_GATE_DIAGNOSTIC_CASES,
    build_phase5_configurations,
    evaluate_phase5_acceptance,
    build_phase5_reproducibility_manifest,
    preflight_phase5_benchmark,
    adjudicated_class_from_disposition,
)
from quantum_simulator.ml_diagnosis import CLASSES


def _observations():
    cases = {case.case_id: case for case in p5.build_ledger()}
    fingerprints = {(row["case_id"], row["seed"], row["shots"]): row["input_fingerprint"] for row in _pairs()}
    rows = []
    for case in cases.values():
        for seed in p5.SEEDS:
            for shots in p5.SHOT_COUNTS:
                targets = []
                causes = []
                if case.suffix in ("P1", "P2"):
                    index = p5._PROGRAM_MUTATIONS[case.family_id][0 if case.suffix == "P1" else 1][0]
                    targets.append(f"program_index:{index}")
                    causes.append("PROGRAM_FAULT")
                elif case.suffix.startswith("R"):
                    targets.append(f"qubit:{p5._R_TARGET[case.family_id]}")
                    causes.append("HARDWARE_ANOMALY")
                elif case.suffix.startswith("G"):
                    gate = p5._G_TARGET[case.family_id]
                    targets.append(f"gate:{gate[0]}:{','.join(map(str, gate[1:]))}")
                    causes.append("HARDWARE_ANOMALY")
                elif case.suffix == "C15":
                    index = p5._PROGRAM_MUTATIONS[case.family_id][0][0]
                    targets.extend((f"program_index:{index}", f"qubit:{p5._R_TARGET[case.family_id]}"))
                    causes.extend(("PROGRAM_FAULT", "HARDWARE_ANOMALY"))
                if case.family_id == p5.FAMILIES[2] and case.suffix in ("P1", "P2"):
                    raw_prediction, disposition = "NORMAL", "PROGRAM_FAULT"
                elif case.family_id == p5.FAMILIES[4] and case.suffix in ("P1", "P2"):
                    raw_prediction, disposition = "HARDWARE_ANOMALY", "AMBIGUOUS / INSUFFICIENT_EVIDENCE"
                else:
                    raw_prediction = case.truth if isinstance(case.truth, str) else "NORMAL"
                    disposition = "AMBIGUOUS" if case.suffix == "C15" else "NO_ANOMALY" if case.suffix == "N0" else case.truth
                rows.append({
                    "case_id": case.case_id, "seed": seed, "shots": shots,
                    "anomaly_detected": case.case_id in p5._SENSITIVITY_CASES,
                    "raw_model_prediction": raw_prediction,
                    "raw_model_confidence": .9,
                    "disposition": disposition,
                    "adjudicated_class": adjudicated_class_from_disposition(disposition),
                    "predicted_targets": targets, "predicted_causes": causes,
                    "artifact_fingerprint": f"artifact-{case.case_id}-{seed}-{shots}",
                    "input_fingerprint": fingerprints[(case.case_id, seed, shots)],
                })
    return rows


@lru_cache(maxsize=1)
def _pairs(equal=True):
    rows = [dict(row) for row in build_phase5_reproducibility_manifest()]
    if not equal:
        rows[0]["regenerated_reference_fingerprint"] = "different-reference"
    return rows


def test_builds_all_900_configurations_with_complete_profiles_and_split():
    configs = build_phase5_configurations()
    assert len(configs) == 900
    assert len({item.key for item in configs}) == 900
    assert {item.case.case_id for item in configs} == {case.case_id for case in p5.build_ledger()}
    assert {item.case.family_id for item in configs if item.case.family_id in p5.PARTITIONS["development"]} == p5.PARTITIONS["development"]
    assert {item.case.family_id for item in configs} == set().union(*p5.PARTITIONS.values())
    assert not (p5.PARTITIONS["development"] & p5.PARTITIONS["calibration"])
    assert not (p5.PARTITIONS["locked_test"] & (p5.PARTITIONS["development"] | p5.PARTITIONS["calibration"]))
    for item in configs:
        p5.validate_calibration(item.config.hardware_parameters, item.case.expected_circuit, item.case.observed_circuit, case=item.case)
    c15 = [item for item in configs if item.case.suffix == "C15"]
    assert len(c15) == 75 and all(item.config.ground_truth_label is None for item in c15)


def test_specificity_is_only_contract_named_n0_and_masked_controls():
    assert len(SPECIFICITY_CASES) == 8
    assert F3_GATE_DIAGNOSTIC_CASES == {f"{p5.FAMILIES[2]}-{suffix}" for suffix in ("G02", "G06", "G15", "G30")}
    assert {f"{family}-N0" for family in p5.FAMILIES} <= SPECIFICITY_CASES
    assert {f"{p5.FAMILIES[2]}-P1", f"{p5.FAMILIES[2]}-P2", f"{p5.FAMILIES[4]}-P2"} <= SPECIFICITY_CASES
    assert not (SPECIFICITY_CASES & F3_GATE_DIAGNOSTIC_CASES)


def test_read_only_dry_run_preflights_all_profiles_and_model():
    destination = Path(__file__).resolve().parents[1] / "reports" / "experiments" / "__phase5_preflight_must_not_create__"
    summary = preflight_phase5_benchmark(result_dir=destination)
    assert summary["status"] == "PREFLIGHT_READY"
    assert summary["configuration_count"] == 900
    assert summary["profile_count_validated"] == 60
    assert summary["sensitivity_per_shot_denominator"] == 105
    assert summary["specificity_per_shot_denominator"] == 40
    assert summary["reserved_family_f3_gate_diagnostic_templates"] == 4
    assert summary["cases_executed"] is False
    assert not destination.exists()


def test_acceptance_reports_each_metric_and_c15_separately():
    result = evaluate_phase5_acceptance(_observations(), reproducibility_manifest=_pairs())
    assert result["overall_acceptance_pass"] is True
    assert all(result["acceptance_checks"].values())
    assert result["classification"]["denominator"] == 675
    assert result["localization"]["denominator"] > 0
    assert result["abstention"]["denominator"] == 900
    assert result["abstention"]["c15_denominator"] == 75
    assert result["localization"]["c15_combined_target_recovery"]["denominator"] == 75
    assert result["c15_excluded_from_three_class_metrics"] is True
    assert all(v["sensitivity_denominator"] == 105 and v["specificity_denominator"] == 40 for v in result["output_detection_by_shot"].values())
    assert result["specificity_includes_f3_gate_cases"] is False
    assert len(result["reserved_family_f3_gate_diagnostic"][256]) == 4
    assert {v["denominator"] for v in result["reserved_family_f3_gate_diagnostic"][256].values()} == {5}


def test_acceptance_rejects_missing_and_duplicate_observations():
    rows = _observations()
    with pytest.raises(ValueError, match="missing scoring rows"):
        evaluate_phase5_acceptance(rows[:-1], reproducibility_manifest=_pairs())
    with pytest.raises(ValueError, match="duplicate scoring row"):
        evaluate_phase5_acceptance(rows + [rows[0]], reproducibility_manifest=_pairs())


@pytest.mark.parametrize("criterion", ["sensitivity", "specificity", "classification", "localization", "c15_ambiguous", "seed_sd", "repeat"])
def test_each_numeric_acceptance_gate_fails_closed(criterion):
    rows = _observations()
    if criterion == "sensitivity":
        for row in rows:
            if row["shots"] == 256 and row["case_id"] in sorted(p5._SENSITIVITY_CASES)[:5]:
                row["anomaly_detected"] = False
    elif criterion == "specificity":
        for row in rows:
            if row["shots"] == 256 and row["case_id"] in sorted(SPECIFICITY_CASES)[:5]:
                row["anomaly_detected"] = True
    elif criterion == "classification":
        program_rows = [r for r in rows if r["case_id"].endswith(("-P1", "-P2"))]
        for row in program_rows[:60]:
            row["disposition"] = "INSUFFICIENT_EVIDENCE"
            row["adjudicated_class"] = "ABSTAIN"
    elif criterion == "localization":
        for row in rows:
            row["predicted_targets"] = []
    elif criterion == "c15_ambiguous":
        for row in rows:
            if row["case_id"].endswith("-C15") and row["seed"] in p5.SEEDS[:2]:
                row["disposition"] = "INSUFFICIENT_EVIDENCE"
    elif criterion == "seed_sd":
        for row in rows:
            if row["shots"] == 256 and row["seed"] == p5.SEEDS[0] and row["case_id"] in p5._SENSITIVITY_CASES:
                row["anomaly_detected"] = False
    manifest = [dict(row) for row in _pairs()]
    if criterion == "repeat":
        manifest[0]["regenerated_configuration_fingerprint"] = "different-config"
    result = evaluate_phase5_acceptance(rows, reproducibility_manifest=manifest)
    failed = {
        "sensitivity": "sensitivity_each_shot",
        "specificity": "specificity_each_shot",
        "classification": "classification",
        "localization": "localization",
        "c15_ambiguous": "c15_ambiguous_recall",
        "seed_sd": "reproducibility_seed_sd",
        "repeat": "reproducibility_deterministic_config_reference_artifacts",
    }[criterion]
    assert result["acceptance_checks"][failed] is False


def test_reproducibility_manifest_covers_all_configs_without_resampling():
    manifest = build_phase5_reproducibility_manifest()
    assert len(manifest) == 900
    assert all(row["configuration_fingerprint"] == row["regenerated_configuration_fingerprint"] for row in manifest)
    assert all(row["reference_fingerprint"] == row["regenerated_reference_fingerprint"] for row in manifest)
    assert evaluate_phase5_acceptance(_observations(), reproducibility_manifest=manifest)["acceptance_checks"]["reproducibility_deterministic_config_reference_artifacts"]
    assert not evaluate_phase5_acceptance(_observations(), reproducibility_manifest=manifest)["reproducibility"]["stochastic_sampled_artifacts_repeated"]


def test_reproducibility_manifest_rejects_a_changed_configuration_input():
    configurations = list(build_phase5_configurations())
    changed = copy.deepcopy(configurations[0].config)
    changed.experiment_name += "_changed"
    configurations[0] = type(configurations[0])(
        configurations[0].case, configurations[0].seed, configurations[0].shots, changed,
    )
    with pytest.raises(ValueError, match="regeneration mismatch"):
        build_phase5_reproducibility_manifest(configurations)


def test_owner_decision_resolves_specificity_membership():
    result = evaluate_phase5_acceptance(_observations(), reproducibility_manifest=_pairs())
    assert len(result["specificity_templates"]) == 8
    assert result["specificity_includes_f3_gate_cases"] is False
    assert set(result["reserved_family_f3_gate_diagnostic"][256]) == F3_GATE_DIAGNOSTIC_CASES


@pytest.mark.parametrize(("disposition", "expected"), [
    ("NO_ANOMALY", "NORMAL"),
    ("PROGRAM_FAULT", "PROGRAM_FAULT"),
    ("HARDWARE_ANOMALY", "HARDWARE_ANOMALY"),
    ("AMBIGUOUS", "ABSTAIN"),
    ("AMBIGUOUS / INSUFFICIENT_EVIDENCE", "ABSTAIN"),
    ("INSUFFICIENT_EVIDENCE", "ABSTAIN"),
    ("ABSTAIN", "ABSTAIN"),
])
def test_disposition_mapping_is_contract_consistent(disposition, expected):
    assert adjudicated_class_from_disposition(disposition) == expected


def test_masked_f3_program_faults_score_from_adjudicated_disposition():
    rows = _observations()
    selected = [r for r in rows if r["case_id"] in {f"{p5.FAMILIES[2]}-P1", f"{p5.FAMILIES[2]}-P2"}]
    assert len(selected) == 30
    assert {r["raw_model_prediction"] for r in selected} == {"NORMAL"}
    assert {r["adjudicated_class"] for r in selected} == {"PROGRAM_FAULT"}
    scored = evaluate_phase5_acceptance(rows, reproducibility_manifest=_pairs())
    assert scored["classification"]["confusion_matrix"]["PROGRAM_FAULT"]["PROGRAM_FAULT"] == 120
    assert scored["classification"]["raw_model_confusion_matrix"]["PROGRAM_FAULT"]["NORMAL"] == 30


def test_f5_ambiguous_program_cases_are_abstentions_not_hardware_predictions():
    rows = _observations()
    selected = [r for r in rows if r["case_id"] in {f"{p5.FAMILIES[4]}-P1", f"{p5.FAMILIES[4]}-P2"}]
    assert len(selected) == 30
    assert {r["raw_model_prediction"] for r in selected} == {"HARDWARE_ANOMALY"}
    assert {r["adjudicated_class"] for r in selected} == {"ABSTAIN"}
    scored = evaluate_phase5_acceptance(rows, reproducibility_manifest=_pairs())
    assert scored["classification"]["confusion_matrix"]["PROGRAM_FAULT"]["ABSTAIN"] == 30
    assert scored["classification"]["confusion_matrix"]["PROGRAM_FAULT"]["HARDWARE_ANOMALY"] == 0
    assert scored["classification"]["raw_model_confusion_matrix"]["PROGRAM_FAULT"]["HARDWARE_ANOMALY"] == 30


def test_normal_hardware_and_abstention_dispositions_remain_distinct():
    rows = _observations()
    scored = evaluate_phase5_acceptance(rows, reproducibility_manifest=_pairs())
    matrix = scored["classification"]["confusion_matrix"]
    assert matrix["NORMAL"]["NORMAL"] == 75
    assert matrix["HARDWARE_ANOMALY"]["HARDWARE_ANOMALY"] == 450
    assert scored["abstention"]["adjudicated_abstentions"] == 105
