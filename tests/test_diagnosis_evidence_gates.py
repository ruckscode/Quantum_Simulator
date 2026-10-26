import hashlib
import json
from pathlib import Path

import pytest

from experiments.diagnosis_evidence_gates import (
    audit_confirmatory_study_preflight,
    audit_family_split_alignment,
    reconcile_saved_evidence,
    validate_independent_adjudication_records,
    validate_inventory_conflict_reviews,
)
from experiments.diagnosis_evaluation_protocol import ProtocolRecordError


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _seal(document, field):
    document.pop(field, None)
    document[field] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return document


def _record(case_id="case-1", *, protocol_sha=None, judgments=None, resolution=None, family="family-1"):
    judgments = judgments or ["BEHAVIORAL_PROGRAM_FAULT", "BEHAVIORAL_PROGRAM_FAULT"]
    return {
        "case_id": case_id,
        "family_root_id": family,
        "lineage_root_id": f"lineage-{family}",
        "protocol_version": "independent-diagnosis-v1",
        "protocol_sha256": protocol_sha or _sha("protocol-v1"),
        "evidence_refs": [{"evidence_id": "immutable-observation", "sha256": _sha("evidence")}],
        "execution_ids": [f"exec-{case_id}-1", f"exec-{case_id}-2"],
        "independent_judgments": [
            {
                "adjudicator_token_sha256": _sha(f"reviewer-{idx}"),
                "blinding_status": "BLINDED_TO_SYSTEM_OUTPUTS",
                "judgment": value,
                "evidence_refs": [_sha("evidence")],
            }
            for idx, value in enumerate(judgments)
        ],
        "resolution": resolution or {"status": "AGREED"},
    }


def _attestations(record, *, protocol_sha=None):
    return [
        {
            "status": "VERIFIED",
            "case_id": record["case_id"],
            "protocol_sha256": protocol_sha or record["protocol_sha256"],
            "adjudicator_token_sha256": item["adjudicator_token_sha256"],
            "identity_tokenization_verified": True,
            "blinding_verified": True,
            "independence_verified": True,
            "attestation_sha256": _sha(f"attestation-{idx}"),
            "verified_by": "external-trust-verifier-fixture",
        }
        for idx, item in enumerate(record["independent_judgments"])
    ]


def _blocked_lineage_fixture(tmp_path):
    """Return synthetic-only fixtures for deterministic, fail-closed gate tests.

    These records exercise reconciliation behavior; they are not an inventory of
    real repository artifacts and must never be treated as external provenance.
    """
    families = []
    conflicts = []
    source_rules = []

    for index in range(1, 4):
        root_id = f"conflicting-root-{index}"
        families.append({
            "lineage_root_id": root_id,
            "family_ids": [f"conflicting-family-{index}"],
            "topology_sha256s": [_sha(f"topology-a-{index}"), _sha(f"topology-b-{index}")],
            "artifact_paths": [],
            "ancestor_values": [],
            "lineage_status": "observed_unadjudicated",
        })
        conflicts.append({
            "kind": "lineage_root_maps_to_multiple_fingerprints",
            "identifier": root_id,
            "fingerprints": [_sha(f"topology-a-{index}"), _sha(f"topology-b-{index}")],
        })

    for index in range(1, 4):
        source = tmp_path / f"declared-source-{index}.py"
        source.write_text(f"# synthetic test source {index}\n", encoding="utf-8")
        root_id = f"design_recipe_{index:02d}"
        families.append({
            "lineage_root_id": root_id,
            "family_ids": [],
            "topology_sha256s": [],
            "artifact_paths": [],
            "ancestor_values": [],
            "lineage_status": "source_declared_not_observed_in_saved_rows",
        })
        source_rules.append({
            "source_path": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "declared_possible_roots": [root_id],
            "status": "synthetic fixture declaration; not external lineage proof",
        })

    inventory = {
        "schema_version": 1,
        "scan_truncated": True,
        "complete": False,
        "missing_scan_roots": [],
        "skipped_large_file_paths": [],
        "bytes_scanned": 0,
        "artifacts": [],
        "families": families,
        "source_lineage_rules": source_rules,
        "conflicts": conflicts,
        "summary": {
            "artifact_count": 0,
            "bytes_scanned": 0,
            "circuit_record_count": 5366,
            "unknown_lineage_circuit_count": 5186,
            "duplicate_content_count": 457,
            "family_only_without_root_record_count": 1671,
            "source_declared_possible_root_count": 6,
            "observed_lineage_root_count": 3,
            "conflict_count": 3,
        },
    }
    _seal(inventory, "inventory_sha256")

    frame = {
        "schema_version": 1,
        "inventory_sha256": inventory["inventory_sha256"],
        "status": "blocked_incomplete_prior_inventory",
        "candidates": [],
        "selected": {"development": [], "validation": []},
    }
    _seal(frame, "frame_sha256")

    recovery_records = [
        {
            "circuit_family_id": f"synthetic-family-{family_index:02d}",
            "run_id": f"synthetic-run-{family_index:02d}-{run_index:02d}",
            "family_recipe_root": None,
            "source_sha256_recorded": None,
        }
        for family_index in range(30)
        for run_index in range(9)
    ]
    recovery = {
        "dataset_manifest_sha256": _sha("synthetic-only-recovery-manifest"),
        "records": recovery_records,
        "summary": {
            "record_count": len(recovery_records),
            "total_report_bytes_read": 0,
            "missing_lineage_rows": len(recovery_records),
            "missing_run_source_hash_rows": len(recovery_records),
            "independent_adjudication_rows": 0,
        },
    }
    return inventory, frame, recovery


def test_missing_evidence_and_self_attested_independence_block_case():
    record = _record()
    record["evidence_refs"] = []
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=record["protocol_sha256"]
    )
    assert not result["independent_adjudication_ready"]
    assert any("evidence references are missing" in item for item in result["blockers"])
    assert any("external blinding/independence attestation" in item for item in result["blockers"])


def test_matching_external_attestations_allow_only_bound_and_blinded_agreed_case():
    record = _record()
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=record["protocol_sha256"],
        verified_adjudicator_attestations=_attestations(record),
    )
    assert result["independent_adjudication_ready"]
    assert result["eligible_case_ids"] == ["case-1"]
    assert result["independent_family_count"] == 1


@pytest.mark.parametrize("disposition", [
    "BEHAVIORAL_PROGRAM_FAULT", "HARDWARE_FAULT", "EQUIVALENT_BEHAVIOR", "INSUFFICIENT_EVIDENCE",
])
def test_independent_truth_uses_explicit_behavior_hardware_equivalence_and_insufficient_categories(disposition):
    record = _record(judgments=[disposition, disposition])
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=record["protocol_sha256"],
        verified_adjudicator_attestations=_attestations(record),
    )
    assert result["independent_adjudication_ready"]


def test_missing_evidence_digest_and_invalid_protocol_binding_block_adjudication():
    record = _record()
    record["evidence_refs"] = [{"evidence_id": "evidence-without-digest"}]
    attestations = _attestations(record)
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=_sha("other-protocol"),
        verified_adjudicator_attestations=_attestations(record),
    )
    assert not result["independent_adjudication_ready"]
    assert any("stable IDs and SHA-256" in item for item in result["blockers"])
    assert any("not bound to the frozen protocol hash" in item for item in result["blockers"])
    assert any("attestation is missing or mismatched" in item for item in result["blockers"])


def test_conflicting_judgments_remain_unresolved_without_fabricated_consensus():
    record = _record(judgments=["BEHAVIORAL_PROGRAM_FAULT", "HARDWARE_FAULT"], resolution={"status": "UNRESOLVED"})
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=record["protocol_sha256"],
        verified_adjudicator_attestations=_attestations(record),
    )
    assert not result["independent_adjudication_ready"]
    assert any("conflicting judgments remain unresolved" in item for item in result["blockers"])


def test_explicit_resolution_preserves_disagreement_and_requires_evidence():
    resolver_token = _sha("reviewer-resolver")
    record = _record(
        judgments=["BEHAVIORAL_PROGRAM_FAULT", "HARDWARE_FAULT"],
        resolution={
            "status": "DISAGREEMENT_RESOLVED",
            "method": "blinded_third_review",
            "resolution_evidence_sha256": _sha("resolution"),
            "resolved_judgment": "INSUFFICIENT_EVIDENCE",
            "resolver_token_sha256": resolver_token,
        },
    )
    attestations = _attestations(record)
    attestations.append({
        "status": "VERIFIED",
        "case_id": record["case_id"],
        "protocol_sha256": record["protocol_sha256"],
        "adjudicator_token_sha256": resolver_token,
        "identity_tokenization_verified": True,
        "blinding_verified": True,
        "independence_verified": True,
        "attestation_sha256": _sha("resolver-attestation"),
        "verified_by": "external-trust-verifier-fixture",
    })
    result = validate_independent_adjudication_records(
        [record], protocol_sha256=record["protocol_sha256"],
        verified_adjudicator_attestations=attestations,
    )
    assert result["independent_adjudication_ready"]


def test_simulator_labels_or_system_predictions_are_never_adjudication_truth():
    record = _record()
    record["scenario_label"] = "PROGRAM_FAULT"
    with pytest.raises(ProtocolRecordError, match="not adjudication truth"):
        validate_independent_adjudication_records([record], protocol_sha256=record["protocol_sha256"])


def test_raw_reviewer_identity_is_rejected_in_favor_of_opaque_external_token():
    record = _record()
    record["independent_judgments"][0]["reviewer_name"] = "invented reviewer"
    with pytest.raises(ProtocolRecordError, match="raw adjudicator identities"):
        validate_independent_adjudication_records([record], protocol_sha256=record["protocol_sha256"])


def test_repeated_executions_in_same_family_cannot_inflate_independent_case_count():
    left = _record("case-1", family="shared-family")
    right = _record("case-2", family="shared-family")
    with pytest.raises(ProtocolRecordError, match="not independent cases"):
        validate_independent_adjudication_records(
            [left, right], protocol_sha256=left["protocol_sha256"],
            verified_adjudicator_attestations=_attestations(left) + _attestations(right),
        )


def test_reconciliation_is_deterministic_and_does_not_promote_inferred_roots(tmp_path):
    inventory, frame, recovery = _blocked_lineage_fixture(tmp_path)
    first = reconcile_saved_evidence(tmp_path, inventory, frame, recovery)
    second = reconcile_saved_evidence(tmp_path, inventory, frame, recovery)
    assert first == second
    assert first["summary"]["candidate_root_count"] == 6
    assert first["summary"]["conflicting_root_count"] == 3
    assert first["summary"]["inferred_root_count"] == 3
    assert first["summary"]["recovered_synthetic_family_id_count"] == 30
    assert first["summary"]["recovered_synthetic_record_count"] == 270
    assert first["summary"]["eligible_independent_families"] == 0
    assert all(not item["eligible"] for item in first["candidate_roots"])
    assert first["resource_boundary"]["no_new_filesystem_discovery"] is True


def test_conflict_review_ledger_is_hash_bound_resumable_and_never_self_verifies():
    conflict = {"kind": "lineage_root_maps_to_multiple_fingerprints", "identifier": "root-a",
                "fingerprints": [_sha("a"), _sha("b")]}
    inventory = {"inventory_sha256": _sha("inventory"), "conflicts": [conflict]}
    conflict_sha = _sha(json.dumps(conflict, sort_keys=True, separators=(",", ":")))
    review = {
        "conflict_sha256": conflict_sha,
        "classification": "EXTERNALLY_ATTESTED",
        "reviewer_token_sha256": _sha("opaque-reviewer"),
        "rationale": "Custodian compared the cited source records.",
        "evidence_refs": [{"evidence_id": "source-record-a", "sha256": _sha("source-a")}],
    }
    first = validate_inventory_conflict_reviews(inventory, [review])
    second = validate_inventory_conflict_reviews(inventory, [review])
    assert first == second
    assert first["reviews"][0]["provenance_classification"] == "CONFLICTING"
    assert first["reviews"][0]["external_attestation_status"] == "MISSING"
    assert first["reviews"][0]["resolved"] is False
    assert first["summary"]["unreviewed_conflict_count"] == 0
    assert first["blockers"]
    assert first["review_queue"][0]["conflict_sha256"] == conflict_sha
    assert first["review_queue"][0]["review_status"] == "REVIEWED"

    attestation = {
        "status": "VERIFIED", "inventory_sha256": inventory["inventory_sha256"],
        "conflict_sha256": conflict_sha, "attestation_sha256": _sha("external-attestation"),
        "review_sha256": _sha(json.dumps(review, sort_keys=True, separators=(",", ":"))),
        "verified_by": "independent-trust-service",
    }
    attested = validate_inventory_conflict_reviews(inventory, [review], verified_attestations=[attestation])
    assert attested["reviews"][0]["provenance_classification"] == "EXTERNALLY_ATTESTED"
    assert attested["reviews"][0]["resolved"] is False
    changed_review = {**review, "rationale": "Changed after attestation."}
    stale_attestation = validate_inventory_conflict_reviews(
        inventory, [changed_review], verified_attestations=[attestation]
    )
    assert stale_attestation["reviews"][0]["provenance_classification"] == "CONFLICTING"

    queued = validate_inventory_conflict_reviews(inventory)
    assert queued["review_queue"][0]["review_status"] == "UNREVIEWED"
    assert queued["review_queue"][0]["record_template"]["evidence_refs"] == []


@pytest.mark.parametrize("mutation", ["unknown_conflict", "empty_rationale", "missing_evidence", "duplicate"])
def test_conflict_review_ledger_rejects_malformed_or_stale_reviews(mutation):
    conflict = {"kind": "family_id_maps_to_multiple_roots", "identifier": "family-a",
                "lineage_roots": ["root-a", "root-b"]}
    inventory = {"inventory_sha256": _sha("inventory"), "conflicts": [conflict]}
    conflict_sha = _sha(json.dumps(conflict, sort_keys=True, separators=(",", ":")))
    review = {"conflict_sha256": conflict_sha, "classification": "INFERRED",
              "reviewer_token_sha256": _sha("reviewer"), "rationale": "Compared records.",
              "evidence_refs": [{"evidence_id": "record", "sha256": _sha("record-bytes")}]}
    reviews = [review]
    if mutation == "unknown_conflict":
        review["conflict_sha256"] = _sha("stale inventory conflict")
    elif mutation == "empty_rationale":
        review["rationale"] = "  "
    elif mutation == "missing_evidence":
        review["evidence_refs"] = []
    elif mutation == "duplicate":
        reviews = [review, dict(review)]
    with pytest.raises(ProtocolRecordError):
        validate_inventory_conflict_reviews(inventory, reviews)


def test_reconciliation_eligibility_requires_complete_attested_frame_and_exact_source_hashes(tmp_path):
    source = tmp_path / "source.json"
    generator = tmp_path / "generator.py"
    config = tmp_path / "config.json"
    evidence = tmp_path / "observation.json"
    for path, value in ((source, "circuit"), (generator, "generator"), (config, "config"), (evidence, "observation")):
        path.write_text(value, encoding="utf-8")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, generator, config, evidence)}
    inventory = {
        "complete": True, "scan_truncated": False, "missing_scan_roots": [], "skipped_large_file_paths": [],
        "summary": {"unknown_lineage_circuit_count": 0, "conflict_count": 0},
        "families": [{"lineage_root_id": "old-root", "family_ids": ["old-family"], "topology_sha256s": [_sha("old-topology")], "artifact_paths": [], "ancestor_values": [], "lineage_status": "observed_unadjudicated"}],
        "artifacts": [], "source_lineage_rules": [], "conflicts": [],
    }
    _seal(inventory, "inventory_sha256")
    candidate = {
        "family_id": "fresh-family", "lineage_root_id": "fresh-root", "topology_sha256": _sha("fresh-topology"),
        "ancestor_root_ids": [], "partition": "development",
        "source_path": "source.json", "source_sha256": hashes["source.json"],
        "generator_path": "generator.py", "generator_sha256": hashes["generator.py"],
        "configuration_path": "config.json", "configuration_sha256": hashes["config.json"],
        "supporting_evidence": [{"path": "observation.json", "sha256": hashes["observation.json"]}],
    }
    frame = {
        "inventory_sha256": inventory["inventory_sha256"], "status": "ready",
        "candidates": [candidate], "selected": {"development": ["fresh-family"], "validation": []},
    }
    _seal(frame, "frame_sha256")
    inventory_attestation = {
        "status": "VERIFIED_COMPLETE", "inventory_sha256": inventory["inventory_sha256"],
        "attestation_sha256": _sha("inventory-attestation"), "unresolved_record_count": 0, "verified_by": "external-verifier",
    }
    frame_attestation = {
        "status": "VERIFIED_FRAME", "inventory_sha256": inventory["inventory_sha256"],
        "frame_sha256": frame["frame_sha256"], "attestation_sha256": _sha("frame-attestation"), "verified_by": "external-verifier",
    }
    report = reconcile_saved_evidence(
        tmp_path, inventory, frame, {"records": [], "summary": {"record_count": 0, "total_report_bytes_read": 0}},
        verified_inventory_attestation=inventory_attestation, verified_frame_attestation=frame_attestation,
    )
    assert report["summary"]["eligible_independent_families"] == 1
    assert report["new_frame_candidates"][0]["provenance_classification"] == "DIRECTLY_VERIFIED"

    candidate["generator_sha256"] = _sha("stale-generator")
    _seal(frame, "frame_sha256")
    stale = reconcile_saved_evidence(
        tmp_path, inventory, frame, {"records": [], "summary": {"record_count": 0, "total_report_bytes_read": 0}},
        verified_inventory_attestation=inventory_attestation,
        verified_frame_attestation={**frame_attestation, "frame_sha256": frame["frame_sha256"]},
    )
    assert stale["summary"]["eligible_independent_families"] == 0
    assert any("generator_path bytes missing or hash mismatch" in item for item in stale["new_frame_candidates"][0]["eligibility_reasons"])


def test_current_confirmatory_preflight_reports_missing_external_gates(tmp_path):
    inventory, frame, recovery = _blocked_lineage_fixture(tmp_path)
    result = audit_confirmatory_study_preflight(
        repository=tmp_path, inventory=inventory, frame=frame, recovery=recovery,
        protocol={"stage": "confirmatory", "preregistration_sha256": _sha("protocol")},
        protocol_cases=[], adjudication_records=[],
    )
    assert not result["ready"]
    assert any("zero eligible independent families" in item for item in result["blockers"])
    assert any("externally verified inventory attestation" in item for item in result["blockers"])
    assert any("protocol gate" in item for item in result["blockers"])


def test_family_frame_lineages_map_only_to_train_calibration_or_untouched_test():
    frame = {
        "candidates": [
            {"family_id": "dev-family", "lineage_root_id": "dev-root", "partition": "development"},
            {"family_id": "test-family", "lineage_root_id": "test-root", "partition": "validation"},
        ],
        "selected": {"development": ["dev-family"], "validation": ["test-family"]},
    }
    good = [
        {"case_id": "train-case", "lineage_root_id": "dev-root", "partition": "train"},
        {"case_id": "test-case", "lineage_root_id": "test-root", "partition": "test"},
    ]
    assert audit_family_split_alignment(frame, good) == []
    bad = [
        {"case_id": "train-case", "lineage_root_id": "test-root", "partition": "train"},
        {"case_id": "test-case", "lineage_root_id": "dev-root", "partition": "test"},
    ]
    assert len(audit_family_split_alignment(frame, bad)) == 2
