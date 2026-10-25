import pytest
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase5_benchmark_infrastructure import (
    FAMILIES,
    aggregate_output_detection,
    build_inference_payload,
    build_ledger,
    classifier_inference,
    observable_features,
    score_observation,
    validate_calibration,
    validate_family_split,
    validate_ledger,
)
from quantum_simulator.ml_diagnosis import CLASSES, FEATURE_NAMES, ClassifierPrediction
from quantum_simulator.observable_model_adapter import ARTIFACT_TYPE, PortableLogisticRegressionAdapter


def _profile(case):
    q_target = 2
    readout = .15 if case.suffix == "C15" else {"R02": .02, "R06": .06, "R15": .15, "R30": .30}.get(case.suffix, 0.0)
    qubits = {q: {"readout_error": readout if q == q_target and (case.suffix.startswith("R") or case.suffix == "C15") else 0.0,
                  "t1": 1e6, "t2": 1e6, "status": "available", "available": True} for q in range(4)}
    keys = set()
    for circuit in (case.expected_circuit, case.observed_circuit):
        for name, *qs in circuit:
            keys.add((name, tuple(qs)))
    return {"qubits": qubits,
            "gates": [{"gate": name, "qubits": qs, "error": 0.0, "duration": 0.0, "supported": True} for name, qs in sorted(keys)],
            "coupling_map": [], "edges": []}


def test_exact_ledger_membership_and_duplicate_rejection():
    cases = build_ledger()
    assert len(cases) == 60
    assert len({c.case_id for c in cases}) == 60
    with pytest.raises(ValueError):
        validate_ledger(cases[:-1] + (cases[0],))


def test_calibration_requires_explicit_expected_observed_union():
    case = next(c for c in build_ledger() if c.case_id == f"{FAMILIES[0]}-P2")
    profile = _profile(case)
    validate_calibration(profile, case.expected_circuit, case.observed_circuit, case=case)
    profile["gates"].pop()
    with pytest.raises(ValueError, match="cover"):
        validate_calibration(profile, case.expected_circuit, case.observed_circuit, case=case)


def test_calibration_missing_qubit_field_fails_closed():
    case = build_ledger()[0]
    profile = _profile(case)
    del profile["qubits"][0]["t1"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_calibration(profile, case.expected_circuit, case.observed_circuit, case=case)


def test_inference_payload_rejects_leakage_and_missing_evidence():
    base = {"expected_distribution": {"0": 1.0}, "observed_distribution": {"0": 1.0}, "shots": 256,
            "expected_circuit": [], "observed_circuit": [], "hardware_model": {}, "threshold": .05}
    with pytest.raises(ValueError, match="forbidden"):
        build_inference_payload({**base, "family_id": FAMILIES[0]})
    with pytest.raises(ValueError, match="missing"):
        build_inference_payload({k: v for k, v in base.items() if k != "hardware_model"})


def test_frozen_feature_order_uses_only_distribution_observables():
    features = observable_features({"0": 1.0, "1": 0.0}, {"0": .75, "1": .25})
    assert tuple(features) == ("total_variation_distance", "js_divergence", "hellinger_distance",
                               "observed_entropy_bits", "observed_max_probability")
    assert features["total_variation_distance"] == pytest.approx(.25)
    assert features["observed_max_probability"] == pytest.approx(.75)


def test_family_partitions_are_disjoint_and_reject_unknown_family():
    validate_family_split(FAMILIES)
    with pytest.raises(ValueError):
        validate_family_split((*FAMILIES, "F6_UNKNOWN"))


def test_scoring_keeps_d_truth_and_c15_combined_recovery_separate():
    cases = build_ledger()
    f3g = next(c for c in cases if c.case_id == f"{FAMILIES[2]}-G30")
    f3g_score = score_observation(f3g, anomaly_detected=False)
    assert f3g_score["output_detection_eligible"] is True
    assert f3g_score["output_positive_truth"] is False
    c15 = next(c for c in cases if c.case_id == f"{FAMILIES[0]}-C15")
    combined = score_observation(c15, predicted_cause=("PROGRAM_FAULT", "HARDWARE_ANOMALY"), disposition="AMBIGUOUS")
    single = score_observation(c15, predicted_cause="PROGRAM_FAULT", disposition="PROGRAM_FAULT")
    assert combined["combined_recovery"] and combined["categorical_ambiguous"]
    assert not single["combined_recovery"] and single["single_cause_prediction"]


def test_21_template_aggregation_uses_105_per_shot_and_315_total():
    observations = [
        {"case_id": case.case_id, "seed": seed, "shots": shots,
         "anomaly_detected": case.suffix in ("R15", "R30", "G15", "G30", "C15")}
        for case in build_ledger() for seed in (20263001, 20263002, 20263003, 20263004, 20263005)
        for shots in (256, 1024, 4096)
    ]
    result = aggregate_output_detection(observations)
    assert result["sensitivity_templates"] == 21
    assert result["observations_per_shot_level"] == 105
    assert result["total_sensitivity_observations"] == 315
    assert result["specificity_templates"] == 8
    assert result["specificity_membership"] == "owner decision 2026-10-13: N0 plus masked D− controls; F3 G cases reported separately"
    assert len(result["reserved_family_f3_gate_diagnostic"][256]) == 4
    assert {value["denominator"] for value in result["reserved_family_f3_gate_diagnostic"][256].values()} == {5}
    assert all(value["sensitivity_denominator"] == 105 for value in result["per_shot"].values())
    assert result["per_shot"][256]["sensitivity_acceptance"]
    assert result["repeated_configurations_are_independent_templates"] is False
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_output_detection(observations + [observations[0]])


class _ConfidenceClassifier:
    feature_names = FEATURE_NAMES

    def __init__(self, confidence):
        self.confidence = confidence

    def predict(self, features):
        probabilities = {label: (1 - self.confidence) / 2 for label in CLASSES}
        probabilities["NORMAL"] = self.confidence
        return ClassifierPrediction("NORMAL", probabilities)


def test_inference_keeps_fixed_060_abstention_boundary():
    args = dict(expected_distribution={"0": 1.0, "1": 0.0}, observed_distribution={"0": 1.0, "1": 0.0},
                expected_circuit=[("x", 0)], observed_circuit=[("x", 0)], hardware_model=None, shots=256)
    low = classifier_inference(**args, classifier=_ConfidenceClassifier(.59))
    at = classifier_inference(**args, classifier=_ConfidenceClassifier(.60))
    assert low["disposition"] == "AMBIGUOUS / INSUFFICIENT_EVIDENCE"
    assert at["confidence_cutoff"] == .60
    assert at["probabilities_calibrated"] is False
    assert "not a guarantee of 60% empirical correctness" in at["confidence_policy"]
    assert at["disposition"] == "NO_ANOMALY"


def test_portable_phase5_artifact_loading_and_ordered_inference():
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "training_status": "FINAL DEPLOYABLE MODEL",
        "model_type": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASSES),
        "preprocessing": {
            "median_imputer_statistics": [0.0] * 5,
            "standard_scaler_mean": [0.0] * 5,
            "standard_scaler_scale": [1.0] * 5,
        },
        "coefficients": [[0.0] * 5 for _ in CLASSES],
        "intercept": [0.0] * len(CLASSES),
        "probability_calibration": "none",
        "probabilities_calibrated": False,
        "confidence_policy": {
            "cutoff": .60,
            "meaning": "operational confidence rule; not a guarantee of 60% empirical correctness",
            "probability_source": "uncalibrated Logistic Regression predict_proba output",
        },
        "metadata": {"dataset_source": "synthetic unit fixture", "evaluation_methodology": "unit test",
                     "training_families": ["F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"],
                     "excluded_families": ["F3_SWAP_ROUTING", "F4_PARITY_TREE", "F5_PHASE_LAYER"]},
    }
    with tempfile.TemporaryDirectory(prefix=".phase5-artifact-test-", dir=Path(__file__).parent) as work:
        path = Path(work) / "model.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        model = PortableLogisticRegressionAdapter(path)
        prediction = model.predict(dict(zip(FEATURE_NAMES, [0.1] * 5)))
        assert prediction.predicted_class == "NORMAL"
        assert set(prediction.class_probabilities) == set(CLASSES)
