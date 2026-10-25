import pytest

from experiments.diagnosis_evaluation_protocol import (
    ProtocolRecordError,
    audit_research_protocol,
)


def _truth(
    *,
    review="INDEPENDENTLY_ADJUDICATED",
    distribution="CHANGE",
    disposition="SINGLE_CAUSE",
    causes=None,
    source_status="KNOWN",
    source_targets=None,
    hardware_status="NOT_APPLICABLE",
    hardware_targets=None,
):
    return {
        "review_status": review,
        "distribution_status": distribution,
        "cause_disposition": disposition,
        "possible_causes": causes if causes is not None else ["PROGRAM"],
        "source_localization_status": source_status,
        "source_targets": source_targets if source_targets is not None else ["operation:2"],
        "hardware_localization_status": hardware_status,
        "hardware_targets": hardware_targets if hardware_targets is not None else [],
        "adjudicator_ids": ["reviewer-A"],
        "evidence_refs": ["evidence:sha256:abc"],
    }


def _case(case_id, root, partition, *, truth=None, observable=None):
    return {
        "case_id": case_id,
        "family_root_id": f"family-{root}",
        "lineage_root_id": f"lineage-{root}",
        "partition": partition,
        "family_fingerprint_sha256": (root[0].lower() * 64),
        "observable_inputs": observable or {
            "reference_operations": [],
            "submitted_operations": [],
            "reference_counts": {"00": 8, "11": 2},
            "submitted_counts": {"00": 5, "11": 5},
            "sampling_design": {"reference_shots": 10, "submitted_shots": 10},
            "measurement_schema": {"width": 2, "classical_bit_order": "msb_first"},
            "hardware_telemetry": None,
        },
        "provenance": {
            "source_artifact_sha256": "a" * 64,
            "observation_artifact_sha256": "b" * 64,
            "protocol_sha256": "c" * 64,
            "runtime": {
                "python_version": "3.12.10",
                "python_implementation": "CPython",
                "libraries": {"numpy": "2.5.3"},
            },
            "generation_id": "gen-fixture-v1",
            "source_revision": "fixture-source-v1",
            "ancestor_root_ids": [],
            "circuit_sha256": "4" * 64,
            "configuration_sha256": "5" * 64,
            "seed": 17,
            "noise_configuration": {"model": "ideal"},
            "reproduction_record_sha256": "6" * 64,
            "execution_provenance": {
                "execution_id": f"exec-{case_id}",
                "backend_kind": "simulator",
                "executor_source_sha256": "7" * 64,
                "execution_configuration_sha256": "8" * 64,
            },
        },
        "adjudicator_only": truth or _truth(),
    }


def _confirmatory_protocol():
    return {
        "protocol_id": "reviewed-protocol-v1",
        "stage": "confirmatory",
        "target_population": "versioned eligible circuit-family frame",
        "intended_use": "separately specified software fault triage",
        "required_endpoints": ["detection", "cause_disposition", "single_cause_attribution", "abstention"],
        "sampling_frame_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "preregistration_sha256": "f" * 64,
        "acceptance_criteria_sha256": "1" * 64,
        "power_analysis_sha256": "2" * 64,
        "resource_budget_sha256": "3" * 64,
        "independent_review_status": "APPROVED",
        "independent_review_id": "review-record-1",
        "acceptance_criteria_id": "approved-criteria-v1",
    }


def _verified_review(protocol):
    """Test-only stand-in for an external trust-policy verifier."""
    return {
        "status": "APPROVED",
        "review_id": protocol["independent_review_id"],
        "reviewed_protocol_sha256": protocol.get("preregistration_sha256"),
        "verified_by": "test-trust-policy-verifier",
    }


def _audit(protocol, cases):
    return audit_research_protocol(
        protocol, cases, verified_review_attestation=_verified_review(protocol)
    )


def test_protocol_contract_requires_independent_reviewed_inputs():
    cases = [
        _case("train-1", "A", "train"),
        _case("cal-1", "B", "calibration"),
        _case("test-1", "C", "test"),
    ]
    result = _audit(_confirmatory_protocol(), cases)
    assert result["schema_valid"] is True
    assert result["confirmatory_prerequisites_recorded"] is True
    assert result["endpoint_eligible_test_case_ids"]["detection"] == ["test-1"]
    assert result["endpoint_eligible_test_case_ids"]["cause_disposition"] == ["test-1"]
    assert result["endpoint_eligible_test_case_ids"]["single_cause_attribution"] == ["test-1"]
    assert "label quality, statistical power" in result["note"]


@pytest.mark.parametrize("field,value", [
    ("ground_truth", {"cause": "PROGRAM"}),
    ("mutation_id", "mutant-007"),
    ("family_root_id", "family-C"),
    ("simulator_state", [1.0, 0.0]),
])
def test_adjudicator_and_identity_fields_cannot_enter_observable_inputs(field, value):
    case = _case("test-1", "C", "test")
    case["observable_inputs"]["nested"] = {field: value}
    with pytest.raises(ProtocolRecordError, match="leaked into observable_inputs"):
        _audit(_confirmatory_protocol(), [case])


@pytest.mark.parametrize("root_field", ["family_root_id", "lineage_root_id"])
def test_related_roots_cannot_cross_partitions(root_field):
    left = _case("train-1", "A", "train")
    right = _case("test-1", "B", "test")
    right[root_field] = left[root_field]
    with pytest.raises(ProtocolRecordError, match="crosses partitions"):
        _audit(_confirmatory_protocol(), [left, right])


def test_exact_duplicate_family_fingerprint_cannot_be_relabelled_as_an_independent_root():
    left = _case("train-1", "A", "train")
    right = _case("test-1", "B", "test")
    right["family_fingerprint_sha256"] = left["family_fingerprint_sha256"]
    with pytest.raises(ProtocolRecordError, match="exact duplicate family fingerprint"):
        _audit(_confirmatory_protocol(), [left, right])


def test_observation_counts_must_match_the_declared_sampling_design():
    case = _case("test-1", "C", "test")
    case["observable_inputs"]["sampling_design"]["submitted_shots"] = 11
    with pytest.raises(ProtocolRecordError, match="does not match recorded count total"):
        _audit(_confirmatory_protocol(), [case])


def test_hardware_localization_requires_a_declared_telemetry_input_slot():
    protocol = _confirmatory_protocol()
    protocol["required_endpoints"].append("hardware_localization")
    case = _case("test-1", "C", "test")
    del case["observable_inputs"]["hardware_telemetry"]
    with pytest.raises(ProtocolRecordError, match="hardware_telemetry"):
        audit_research_protocol(protocol, [case])


@pytest.mark.parametrize("disposition,causes", [
    ("AMBIGUOUS", ["PROGRAM"]),
    ("OBSERVATIONALLY_EQUIVALENT", ["HARDWARE"]),
    ("SINGLE_CAUSE", ["PROGRAM", "HARDWARE"]),
])
def test_label_dispositions_cannot_be_silently_collapsed(disposition, causes):
    case = _case("test-1", "C", "test", truth=_truth(disposition=disposition, causes=causes))
    with pytest.raises(ProtocolRecordError, match="competing causes|exactly one possible cause"):
        _audit(_confirmatory_protocol(), [case])


def test_ambiguous_and_equivalent_truth_is_kept_for_abstention_not_single_cause_accuracy():
    truth = _truth(
        disposition="OBSERVATIONALLY_EQUIVALENT",
        causes=["PROGRAM", "HARDWARE"],
        source_status="OBSERVATIONALLY_EQUIVALENT",
        source_targets=[],
        hardware_status="OBSERVATIONALLY_EQUIVALENT",
        hardware_targets=[],
    )
    cases = [
        _case("train-1", "A", "train"),
        _case("cal-1", "B", "calibration"),
        _case("test-equivalent", "C", "test", truth=truth),
    ]
    protocol = _confirmatory_protocol()
    protocol["required_endpoints"].remove("single_cause_attribution")
    result = _audit(protocol, cases)
    assert result["confirmatory_prerequisites_recorded"] is True
    assert result["endpoint_eligible_test_case_ids"]["abstention"] == ["test-equivalent"]
    assert result["endpoint_eligible_test_case_ids"]["cause_disposition"] == ["test-equivalent"]
    assert result["endpoint_eligible_test_case_ids"]["single_cause_attribution"] == []


def test_single_cause_acceptance_endpoint_blocks_ambiguous_truth_instead_of_forcing_a_label():
    truth = _truth(
        disposition="AMBIGUOUS",
        causes=["PROGRAM", "HARDWARE"],
        source_status="AMBIGUOUS",
        source_targets=[],
        hardware_status="AMBIGUOUS",
        hardware_targets=[],
    )
    cases = [
        _case("train-1", "A", "train"),
        _case("cal-1", "B", "calibration"),
        _case("test-ambiguous", "C", "test", truth=truth),
    ]
    result = _audit(_confirmatory_protocol(), cases)
    assert result["confirmatory_prerequisites_recorded"] is False
    assert any("single_cause_attribution" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("review,distribution,disposition", [
    ("SYNTHETIC_ONLY", "CHANGE", "SINGLE_CAUSE"),
    ("MISSING", "MISSING", "MISSING"),
    ("CONFLICTING", "CHANGE", "CONFLICTING"),
    ("INDEPENDENTLY_ADJUDICATED", "UNRESOLVED", "UNRESOLVED"),
])
def test_missing_synthetic_conflicting_or_unresolved_truth_blocks_confirmatory_evaluation(review, distribution, disposition):
    truth = _truth(
        review=review,
        distribution=distribution,
        disposition=disposition,
        causes=["PROGRAM"] if disposition == "SINGLE_CAUSE" else [],
        source_status="UNRESOLVED",
        source_targets=[],
        hardware_status="UNRESOLVED",
        hardware_targets=[],
    )
    cases = [
        _case("train-1", "A", "train"),
        _case("cal-1", "B", "calibration"),
        _case("test-1", "C", "test", truth=truth),
    ]
    result = _audit(_confirmatory_protocol(), cases)
    assert result["confirmatory_prerequisites_recorded"] is False
    assert any("lack independent truth" in blocker for blocker in result["blockers"])


def test_missing_preregistration_or_provenance_fails_closed():
    protocol = _confirmatory_protocol()
    protocol.pop("preregistration_sha256")
    cases = [
        _case("train-1", "A", "train"),
        _case("cal-1", "B", "calibration"),
        _case("test-1", "C", "test"),
    ]
    cases[-1]["provenance"]["observation_artifact_sha256"] = None
    result = _audit(protocol, cases)
    assert result["confirmatory_prerequisites_recorded"] is False
    assert any("preregistration_sha256" in blocker for blocker in result["blockers"])
    assert any("observation_artifact_sha256" in blocker for blocker in result["blockers"])


def test_development_protocol_never_claims_confirmatory_prerequisites():
    cases = [_case("dev-1", "A", "train")]
    result = audit_research_protocol({"stage": "development"}, cases)
    assert result["schema_valid"] is True
    assert result["confirmatory_prerequisites_recorded"] is False


def test_self_attested_approval_fields_do_not_satisfy_external_review_gate():
    protocol = _confirmatory_protocol()
    cases = [_case("train-1", "A", "train"), _case("cal-1", "B", "calibration"), _case("test-1", "C", "test")]
    result = audit_research_protocol(protocol, cases)
    assert result["confirmatory_prerequisites_recorded"] is False
    assert any("externally verified" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("field,value", [
    ("generation_id", None),
    ("ancestor_root_ids", None),
    ("circuit_sha256", "not-a-hash"),
    ("configuration_sha256", None),
    ("seed", None),
    ("noise_configuration", None),
    ("reproduction_record_sha256", "bad"),
    ("execution_provenance", None),
])
def test_incomplete_per_case_provenance_blocks_confirmatory_protocol(field, value):
    protocol = _confirmatory_protocol()
    cases = [_case("train-1", "A", "train"), _case("cal-1", "B", "calibration"), _case("test-1", "C", "test")]
    cases[-1]["provenance"][field] = value
    result = _audit(protocol, cases)
    assert result["confirmatory_prerequisites_recorded"] is False
    assert any(field in blocker for blocker in result["blockers"])
