"""Fail-closed provenance reconciliation and independent adjudication gates.

This module consumes persisted bounded inventories; it does not scan the
repository, run circuits, infer missing lineage, or validate external signatures.
Callers may pass attestation records only after an external trust verifier has
validated them against an independently managed trust policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.diagnosis_evaluation_protocol import ProtocolRecordError
from experiments.diagnosis_lineage_inventory import verify_hash

MAX_RECOVERY_RECORDS = 1000
MAX_RECOVERY_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_FILES = 20_000
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
MAX_SINGLE_EVIDENCE_BYTES = 25 * 1024 * 1024
JUDGMENTS = {
    "BEHAVIORAL_PROGRAM_FAULT", "HARDWARE_FAULT", "EQUIVALENT_BEHAVIOR",
    "NO_BEHAVIORAL_CHANGE", "INSUFFICIENT_EVIDENCE", "UNRESOLVED",
}
SYNTHETIC_OR_PREDICTIVE_FIELDS = {
    "scenario_label", "mutation_id", "mutant_id", "injected_fault",
    "simulator_label", "synthetic_label", "system_prediction", "model_prediction",
    "diagnostic_result", "predicted_cause", "predicted_fault",
}
RAW_IDENTITY_FIELDS = {"adjudicator_name", "reviewer_name", "email", "username", "reviewer_id", "adjudicator_id"}
CONFLICT_REVIEW_CLASSIFICATIONS = {"VERIFIED", "INFERRED", "MISSING", "CONFLICTING", "EXTERNALLY_ATTESTED"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _forbidden_record_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in SYNTHETIC_OR_PREDICTIVE_FIELDS | RAW_IDENTITY_FIELDS:
                found.add(normalized)
            found.update(_forbidden_record_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_record_keys(child))
    return found


def _safe_repo_file(root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (root / relative).resolve()
    return candidate if candidate.is_relative_to(root) and candidate.is_file() else None


def validate_inventory_conflict_reviews(
    inventory: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]] = (),
    *,
    verified_attestations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Bind custodian notes to exact inventory conflicts without resolving them.

    Local classifications are reviewer assessments, not verified provenance.
    EXTERNALLY_ATTESTED is reported only when an independently verified
    attestation is supplied through this trust boundary.
    """
    conflicts = inventory.get("conflicts", [])
    if not isinstance(conflicts, list):
        raise ProtocolRecordError("inventory conflicts must be a list")
    if not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes)):
        raise ProtocolRecordError("conflict reviews must be a sequence")
    if not isinstance(verified_attestations, Sequence) or isinstance(verified_attestations, (str, bytes)):
        raise ProtocolRecordError("verified conflict attestations must be a sequence")
    conflict_ids = {_sha(_canonical(item)): item for item in conflicts if isinstance(item, Mapping)}
    attestations = {}
    for attestation in verified_attestations:
        if not isinstance(attestation, Mapping):
            continue
        key = (attestation.get("inventory_sha256"), attestation.get("conflict_sha256"), attestation.get("review_sha256"))
        attestations[key] = attestation
    seen: set[str] = set()
    results = []
    blockers = []
    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            raise ProtocolRecordError(f"conflict review {index} must be a mapping")
        conflict_id = review.get("conflict_sha256")
        if not _is_sha(conflict_id) or conflict_id not in conflict_ids:
            raise ProtocolRecordError(f"conflict review {index} does not match an exact inventory conflict")
        if conflict_id in seen:
            raise ProtocolRecordError(f"duplicate review for inventory conflict {conflict_id}")
        seen.add(conflict_id)
        classification = review.get("classification")
        if classification not in CONFLICT_REVIEW_CLASSIFICATIONS:
            raise ProtocolRecordError(f"conflict review {index} has unsupported classification")
        rationale = review.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProtocolRecordError(f"conflict review {index} requires a decision rationale")
        token = review.get("reviewer_token_sha256")
        if not _is_sha(token):
            raise ProtocolRecordError(f"conflict review {index} requires an opaque reviewer token SHA-256")
        evidence = review.get("evidence_refs")
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("evidence_id"), str) or not item["evidence_id"].strip()
            or not _is_sha(item.get("sha256"))
            for item in evidence
        ):
            raise ProtocolRecordError(f"conflict review {index} requires stable evidence IDs and SHA-256 references")
        review_sha256 = _sha(_canonical(dict(review)))
        attestation = attestations.get((inventory.get("inventory_sha256"), conflict_id, review_sha256), {})
        externally_verified = (
            classification == "EXTERNALLY_ATTESTED"
            and attestation.get("status") == "VERIFIED"
            and _is_sha(attestation.get("attestation_sha256"))
            and attestation.get("inventory_sha256") == inventory.get("inventory_sha256")
            and attestation.get("conflict_sha256") == conflict_id
            and attestation.get("review_sha256") == review_sha256
            and isinstance(attestation.get("verified_by"), str)
            and bool(attestation.get("verified_by", "").strip())
        )
        if classification == "EXTERNALLY_ATTESTED" and not externally_verified:
            blockers.append(f"conflict {conflict_id}: external attestation missing or mismatched")
        results.append({
            "conflict_sha256": conflict_id,
            "review_sha256": review_sha256,
            "conflict": dict(conflict_ids[conflict_id]),
            "reviewer_assessment": classification,
            "provenance_classification": "EXTERNALLY_ATTESTED" if externally_verified else "CONFLICTING",
            "external_attestation_status": "VERIFIED" if externally_verified else "MISSING",
            "reviewer_token_sha256": token,
            "rationale": rationale.strip(),
            "evidence_refs": [dict(item) for item in evidence],
            "resolved": False,
            "resolution_note": "Review is recorded; source conflict remains unresolved until independently reconciled.",
        })
    unresolved = sorted(set(conflict_ids) - seen)
    reviewed_by_id = {item["conflict_sha256"]: item for item in results}
    review_queue = []
    for conflict_sha256, conflict in sorted(conflict_ids.items()):
        review_queue.append({
            "conflict_sha256": conflict_sha256,
            "conflict": dict(conflict),
            "review_status": "REVIEWED" if conflict_sha256 in reviewed_by_id else "UNREVIEWED",
            "record_template": reviewed_by_id.get(conflict_sha256, {
                "classification": "",
                "reviewer_token_sha256": "",
                "rationale": "",
                "evidence_refs": [],
            }),
        })
    return {
        "schema_version": 1,
        "inventory_sha256": inventory.get("inventory_sha256"),
        "reviews": sorted(results, key=lambda item: item["conflict_sha256"]),
        "review_queue": review_queue,
        "unreviewed_conflict_sha256s": unresolved,
        "blockers": blockers,
        "summary": {
            "inventory_conflict_count": len(conflict_ids),
            "reviewed_conflict_count": len(results),
            "unreviewed_conflict_count": len(unresolved),
            "externally_attested_review_count": sum(item["external_attestation_status"] == "VERIFIED" for item in results),
            "resolved_conflict_count": 0,
        },
    }


def _hash_bounded_files(
    root: Path, paths: Sequence[str], byte_budget: list[int], *, max_total_bytes: int = MAX_EVIDENCE_BYTES
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in sorted(set(paths)):
        if len(result) >= MAX_EVIDENCE_FILES:
            result[relative] = {"status": "MISSING", "reason": "evidence file cap reached"}
            continue
        path = _safe_repo_file(root, relative)
        if path is None:
            result[relative] = {"status": "MISSING", "reason": "not present inside repository"}
            continue
        size = path.stat().st_size
        if size > MAX_SINGLE_EVIDENCE_BYTES or byte_budget[0] + size > max_total_bytes:
            result[relative] = {"status": "MISSING", "reason": "configured evidence byte cap exceeded"}
            continue
        digest = _sha(path.read_bytes())
        byte_budget[0] += size
        result[relative] = {"status": "HASHED_CURRENT_BYTES", "sha256": digest, "size_bytes": size}
    return result


def reconcile_saved_evidence(
    repository: str | Path,
    inventory: Mapping[str, Any],
    frame: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    verified_inventory_attestation: Mapping[str, Any] | None = None,
    verified_frame_attestation: Mapping[str, Any] | None = None,
    conflict_reviews: Sequence[Mapping[str, Any]] = (),
    verified_conflict_attestations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create a deterministic evidence-linked reconciliation from saved files.

    A root is never eligible merely because it was inferred or its saved bytes
    still hash correctly. Eligibility requires complete externally attested
    inventory, a reconciled root and lineage, no conflicts/duplicates, and a
    complete exact-generation record. This repository has no such attestation.
    """
    root = Path(repository).resolve()
    inventory_valid = verify_hash(dict(inventory), "inventory_sha256")
    frame_valid = verify_hash(dict(frame), "frame_sha256")
    review_report = validate_inventory_conflict_reviews(
        inventory, conflict_reviews, verified_attestations=verified_conflict_attestations
    )
    summary = inventory.get("summary", {})
    inventory_roots = inventory.get("families", [])
    frame_candidates = frame.get("candidates", [])
    roots: dict[str, dict[str, Any]] = {}
    evidence_paths: set[str] = set()
    for item in inventory_roots if isinstance(inventory_roots, list) else []:
        if not isinstance(item, Mapping) or not isinstance(item.get("lineage_root_id"), str):
            continue
        rid = item["lineage_root_id"]
        roots[rid] = {
            "lineage_root_id": rid,
            "family_ids": sorted(set(item.get("family_ids", []))),
            "topology_sha256s": sorted(set(item.get("topology_sha256s", []))),
            "ancestor_values": item.get("ancestor_values", []),
            "lineage_status": item.get("lineage_status", "MISSING"),
            "evidence_paths": sorted(set(item.get("artifact_paths", []))),
            "source_evidence": [],
        }
        evidence_paths.update(item.get("artifact_paths", []))
    for candidate in frame_candidates if isinstance(frame_candidates, list) else []:
        if isinstance(candidate, Mapping) and isinstance(candidate.get("lineage_root_id"), str):
            rid = candidate["lineage_root_id"]
            roots.setdefault(rid, {
                "lineage_root_id": rid,
                "family_ids": [candidate.get("family_id")] if candidate.get("family_id") else [],
                "topology_sha256s": [candidate.get("topology_sha256")] if candidate.get("topology_sha256") else [],
                "ancestor_values": candidate.get("ancestor_root_ids", []),
                "lineage_status": "frame_candidate_unadjudicated",
                "evidence_paths": [],
                "source_evidence": [],
            })

    inventory_artifacts = {
        item.get("path"): item for item in inventory.get("artifacts", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    rules = inventory.get("source_lineage_rules", [])
    sources_by_root: dict[str, list[Mapping[str, Any]]] = {}
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, Mapping):
            continue
        for rid in rule.get("declared_possible_roots", []):
            sources_by_root.setdefault(rid, []).append(rule)
    # Hash only exact referenced evidence and the small set of generator files;
    # no discovery scan is performed by this reconciliation utility.
    referenced = set(evidence_paths)
    referenced.update(
        str(rule["source_path"]) for rule in rules
        if isinstance(rule, Mapping) and isinstance(rule.get("source_path"), str)
    )
    for candidate in frame_candidates if isinstance(frame_candidates, list) else []:
        if not isinstance(candidate, Mapping):
            continue
        for field in ("source_path", "generator_path", "configuration_path"):
            if isinstance(candidate.get(field), str):
                referenced.add(candidate[field])
        for item in candidate.get("supporting_evidence", []):
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                referenced.add(item["path"])
    current_file_hashes = _hash_bounded_files(root, sorted(referenced), [0])

    artifact_by_path = inventory_artifacts
    exact_duplicate_pairs = []
    for path, captured in artifact_by_path.items():
        duplicate = captured.get("duplicate_of")
        if duplicate and (path in referenced or duplicate in referenced):
            exact_duplicate_pairs.append({"path": path, "duplicate_of": duplicate,
                                          "artifact_sha256": captured.get("artifact_sha256")})

    conflicts = inventory.get("conflicts", [])
    conflict_roots = {
        conflict.get("identifier") for conflict in conflicts
        if isinstance(conflict, Mapping) and conflict.get("kind") == "lineage_root_maps_to_multiple_fingerprints"
    }
    ambiguous_family_ids = {
        conflict.get("identifier") for conflict in conflicts
        if isinstance(conflict, Mapping) and conflict.get("kind") == "family_id_maps_to_multiple_roots"
    }
    for rid, record in roots.items():
        evidence = []
        for path in record["evidence_paths"]:
            artifact = artifact_by_path.get(path, {})
            current = current_file_hashes.get(path, {})
            evidence.append({
                "path": path,
                "inventory_artifact_sha256": artifact.get("artifact_sha256"),
                "current_sha256": current.get("sha256"),
                "integrity_status": (
                    "DIRECTLY_VERIFIED_CURRENT_BYTES_MATCH_CAPTURED_HASH"
                    if current.get("sha256") and current.get("sha256") == artifact.get("artifact_sha256")
                    else "CONFLICTING_CURRENT_BYTES"
                    if current.get("sha256") and artifact.get("artifact_sha256")
                    else current.get("status", "MISSING")
                ),
                "duplicate_of": artifact.get("duplicate_of"),
            })
        source_evidence = []
        for rule in sources_by_root.get(rid, []):
            current = current_file_hashes.get(rule.get("source_path"), {})
            source_evidence.append({
                "source_path": rule.get("source_path"),
                "captured_source_sha256": rule.get("source_sha256"),
                "current_source_sha256": current.get("sha256"),
                "source_hash_status": (
                    "DIRECTLY_VERIFIED_CURRENT_BYTES_MATCH_CAPTURED_HASH"
                    if current.get("sha256") and current.get("sha256") == rule.get("source_sha256")
                    else "CONFLICTING_STALE_SOURCE_HASH"
                    if current.get("sha256") and rule.get("source_sha256")
                    else current.get("status", "MISSING")
                ),
                "declared_root_is_observed": bool(record["evidence_paths"] and any(
                    path != rule.get("source_path") for path in record["evidence_paths"]
                )),
                "declaration_status": rule.get("status"),
            })
        record["supporting_evidence"] = evidence
        record["source_evidence"] = source_evidence
        byte_conflict = any(item.get("integrity_status") == "CONFLICTING_CURRENT_BYTES" for item in evidence)
        source_conflict = any(item.get("source_hash_status") == "CONFLICTING_STALE_SOURCE_HASH" for item in source_evidence)
        record["provenance_classification"] = (
            "CONFLICTING" if rid in conflict_roots or set(record["family_ids"]) & ambiguous_family_ids or byte_conflict or source_conflict
            else "DIRECTLY_VERIFIED" if record["lineage_status"] == "observed_unadjudicated"
                and evidence and all(item.get("integrity_status") == "DIRECTLY_VERIFIED_CURRENT_BYTES_MATCH_CAPTURED_HASH" for item in evidence)
            else "INFERRED" if any(item.get("source_hash_status") == "DIRECTLY_VERIFIED_CURRENT_BYTES_MATCH_CAPTURED_HASH" for item in source_evidence)
            else "MISSING"
        )
        record["external_attestation_status"] = "MISSING"
        record["eligible"] = False
        record["eligibility_reason"] = (
            "Prior root or source-declared possible root; no externally attested completeness/reconciliation, "
            "unresolved lineage/fingerprint conflict or missing ancestry, and no fresh frame candidate."
        )

    prior_roots = {item.get("lineage_root_id") for item in inventory_roots if isinstance(item, Mapping)}
    prior_topologies = {
        fingerprint for item in inventory_roots if isinstance(item, Mapping)
        for fingerprint in item.get("topology_sha256s", []) if _is_sha(fingerprint)
    }
    prior_family_ids = {
        family_id for item in inventory_roots if isinstance(item, Mapping)
        for family_id in item.get("family_ids", []) if isinstance(family_id, str)
    }
    inventory_attested = (
        isinstance(verified_inventory_attestation, Mapping)
        and verified_inventory_attestation.get("status") == "VERIFIED_COMPLETE"
        and verified_inventory_attestation.get("inventory_sha256") == inventory.get("inventory_sha256")
        and _is_sha(verified_inventory_attestation.get("attestation_sha256"))
        and verified_inventory_attestation.get("unresolved_record_count") == 0
        and isinstance(verified_inventory_attestation.get("verified_by"), str)
        and bool(verified_inventory_attestation.get("verified_by", "").strip())
    )
    frame_attested = (
        isinstance(verified_frame_attestation, Mapping)
        and verified_frame_attestation.get("status") == "VERIFIED_FRAME"
        and verified_frame_attestation.get("inventory_sha256") == inventory.get("inventory_sha256")
        and verified_frame_attestation.get("frame_sha256") == frame.get("frame_sha256")
        and _is_sha(verified_frame_attestation.get("attestation_sha256"))
        and isinstance(verified_frame_attestation.get("verified_by"), str)
        and bool(verified_frame_attestation.get("verified_by", "").strip())
    )
    inventory_complete = (
        inventory_valid and inventory.get("complete") is True
        and inventory.get("scan_truncated") is False
        and not inventory.get("missing_scan_roots")
        and not inventory.get("skipped_large_file_paths")
        and isinstance(summary, Mapping)
        and summary.get("unknown_lineage_circuit_count") == 0
        and summary.get("conflict_count") == 0
    )
    frame_integrity = frame_valid and frame.get("inventory_sha256") == inventory.get("inventory_sha256")
    new_candidates = []
    proposed_roots: set[str] = set()
    proposed_families: set[str] = set()
    proposed_fingerprints: set[str] = set()
    candidates_list = frame_candidates if isinstance(frame_candidates, list) else []
    selected = frame.get("selected", {})
    selected_ids = set()
    if isinstance(selected, Mapping):
        for partition in ("development", "validation"):
            for item in selected.get(partition, []):
                if isinstance(item, str):
                    selected_ids.add(item)
                elif isinstance(item, Mapping) and isinstance(item.get("family_id"), str):
                    selected_ids.add(item["family_id"])
    for candidate in candidates_list:
        if not isinstance(candidate, Mapping):
            new_candidates.append({"eligible": False, "eligibility_reason": "malformed candidate record"})
            continue
        reasons = []
        family_id, rid = candidate.get("family_id"), candidate.get("lineage_root_id")
        fingerprint, ancestors = candidate.get("topology_sha256"), candidate.get("ancestor_root_ids")
        partition = candidate.get("partition")
        if not isinstance(family_id, str) or not family_id.strip(): reasons.append("family_id missing")
        if not isinstance(rid, str) or not rid.strip(): reasons.append("lineage_root_id missing")
        if not _is_sha(fingerprint): reasons.append("canonical topology fingerprint missing/invalid")
        if not isinstance(ancestors, list) or any(not isinstance(value, str) or not value.strip() for value in ancestors): reasons.append("complete ancestor_root_ids missing/invalid")
        if partition not in {"development", "validation"}: reasons.append("partition must be development or validation")
        if family_id not in selected_ids: reasons.append("family is absent from the frame's frozen selected partitions")
        if frame.get("status") != "ready": reasons.append("sampling frame is not marked ready")
        if rid in prior_roots or (isinstance(ancestors, list) and set(ancestors) & prior_roots): reasons.append("root or ancestry overlaps a previously inspected root")
        if fingerprint in prior_topologies: reasons.append("exact topology fingerprint duplicates prior evidence")
        if family_id in prior_family_ids: reasons.append("family ID duplicates prior evidence")
        if rid in proposed_roots or family_id in proposed_families or fingerprint in proposed_fingerprints: reasons.append("duplicate candidate family/root/fingerprint")
        if isinstance(rid, str): proposed_roots.add(rid)
        if isinstance(family_id, str): proposed_families.add(family_id)
        if _is_sha(fingerprint): proposed_fingerprints.add(fingerprint)
        bound_sources = (("source_path", "source_sha256"), ("generator_path", "generator_sha256"),
                         ("configuration_path", "configuration_sha256"))
        for path_key, hash_key in bound_sources:
            path_value, expected_hash = candidate.get(path_key), candidate.get(hash_key)
            current = current_file_hashes.get(path_value, {}) if isinstance(path_value, str) else {}
            if not isinstance(path_value, str) or not _is_sha(expected_hash):
                reasons.append(f"{path_key}/{hash_key} provenance missing")
            elif current.get("sha256") != expected_hash:
                reasons.append(f"{path_key} bytes missing or hash mismatch")
        evidence_items = candidate.get("supporting_evidence")
        if not isinstance(evidence_items, list) or not evidence_items:
            reasons.append("exact supporting evidence references missing")
        else:
            for evidence_item in evidence_items:
                if not isinstance(evidence_item, Mapping) or not isinstance(evidence_item.get("path"), str) or not _is_sha(evidence_item.get("sha256")):
                    reasons.append("supporting evidence path/hash malformed")
                    continue
                if current_file_hashes.get(evidence_item["path"], {}).get("sha256") != evidence_item["sha256"]:
                    reasons.append(f"supporting evidence hash mismatch: {evidence_item['path']}")
        if not inventory_complete: reasons.append("prior inventory is incomplete, truncated, unresolved, or lacks external completeness attestation")
        if not inventory_attested: reasons.append("external inventory reconciliation attestation missing or mismatched")
        if not frame_integrity: reasons.append("sampling frame hash is invalid or bound to another inventory")
        if not frame_attested: reasons.append("external sampling-frame attestation missing or mismatched")
        new_candidates.append({
            "family_id": family_id, "lineage_root_id": rid, "partition": partition,
            "topology_sha256": fingerprint, "ancestor_root_ids": ancestors,
            "provenance_classification": "DIRECTLY_VERIFIED" if not reasons else "MISSING_OR_UNRESOLVED",
            "external_attestation_status": "VERIFIED" if inventory_attested and frame_attested else "MISSING",
            "supporting_evidence": [dict(item) for item in evidence_items if isinstance(item, Mapping)] if isinstance(evidence_items, list) else [],
            "eligible": not reasons,
            "eligibility_reasons": reasons,
        })
    proposed_root_set = {item.get("lineage_root_id") for item in new_candidates if isinstance(item.get("lineage_root_id"), str)}
    for item in new_candidates:
        ancestors = item.get("ancestor_root_ids")
        if isinstance(ancestors, list) and proposed_root_set.intersection(ancestors):
            item["eligible"] = False
            item["provenance_classification"] = "CONFLICTING"
            item["eligibility_reasons"].append("candidate family is descended from another proposed family root")

    recovery_records = recovery.get("records", [])
    recovery_summary = recovery.get("summary", {})
    recovery_cap_exceeded = (
        not isinstance(recovery_records, list)
        or len(recovery_records) > MAX_RECOVERY_RECORDS
        or not isinstance(recovery_summary, Mapping)
        or recovery_summary.get("record_count", 0) > MAX_RECOVERY_RECORDS
        or recovery_summary.get("total_report_bytes_read", MAX_RECOVERY_BYTES + 1) > MAX_RECOVERY_BYTES
    )
    recovered_families: dict[str, list[Mapping[str, Any]]] = {}
    for row in recovery_records[:MAX_RECOVERY_RECORDS] if isinstance(recovery_records, list) else []:
        if isinstance(row, Mapping) and isinstance(row.get("circuit_family_id"), str):
            recovered_families.setdefault(row["circuit_family_id"], []).append(row)
    recovery_paths = sorted({
        row.get("report_path") for rows in recovered_families.values() for row in rows
        if isinstance(row.get("report_path"), str)
    })
    # The recovery table declares these caps; refuse to hash if it exceeds them.
    recovery_current = _hash_bounded_files(
        root, recovery_paths, [0], max_total_bytes=MAX_RECOVERY_BYTES
    ) if not recovery_cap_exceeded else {}
    recovered = []
    for family_id, rows in sorted(recovered_families.items()):
        reports = []
        for row in sorted(rows, key=lambda item: str(item.get("run_id", ""))):
            report_path = row.get("report_path")
            relative = None
            if isinstance(report_path, str):
                try:
                    relative = Path(report_path).resolve().relative_to(root).as_posix()
                except ValueError:
                    pass
            current = recovery_current.get(relative, {}) if relative else {"status": "MISSING"}
            recorded = row.get("report_sha256_current_artifact_integrity_only")
            reports.append({
                "run_id": row.get("run_id"), "path": relative,
                "recorded_current_artifact_sha256": recorded,
                "recomputed_current_sha256": current.get("sha256"),
                "integrity_status": "DIRECTLY_VERIFIED_CURRENT_BYTES_MATCH_RECORDED_HASH"
                    if current.get("sha256") and current.get("sha256") == recorded
                    else "CONFLICTING_CURRENT_BYTES" if current.get("sha256") and recorded
                    else current.get("status", "MISSING"),
            })
        recovered.append({
            "candidate_family_id": family_id,
            "record_count": len(rows),
            "run_ids": sorted(str(row.get("run_id", "")) for row in rows),
            "supporting_reports": reports,
            "lineage_status": "MISSING" if all(row.get("family_recipe_root") is None for row in rows) else "RECORDED_UNRECONCILED",
            "source_status": "MISSING" if all(row.get("source_sha256_recorded") is None for row in rows) else "RECORDED_UNRECONCILED",
            "label_status": "SYNTHETIC_ONLY; NOT INDEPENDENT ADJUDICATION",
            "independent_family_count": 0,
            "eligible": False,
            "eligibility_reason": "Run/family IDs do not establish recipe ancestry; generator source hashes and independent adjudication are absent.",
        })

    report = {
        "schema_version": 1,
        "reconciliation_id": "diagnosis-saved-evidence-reconciliation-v1",
        "input_hashes": {
            "inventory_sha256": inventory.get("inventory_sha256"),
            "inventory_hash_valid": inventory_valid,
            "sampling_frame_sha256": frame.get("frame_sha256"),
            "sampling_frame_hash_valid": frame_valid,
            "recovery_manifest_sha256": recovery.get("dataset_manifest_sha256"),
        },
        "resource_boundary": {
            "inventory_scan_reused": True,
            "inventory_was_truncated": inventory.get("scan_truncated") is True,
            "saved_inventory_artifact_count": summary.get("artifact_count"),
            "saved_inventory_bytes_scanned": inventory.get("bytes_scanned"),
            "recovery_record_cap": MAX_RECOVERY_RECORDS,
            "recovery_byte_cap": MAX_RECOVERY_BYTES,
            "current_evidence_rehash_file_cap": MAX_EVIDENCE_FILES,
            "current_evidence_rehash_byte_cap": MAX_EVIDENCE_BYTES,
            "recovery_cap_exceeded": recovery_cap_exceeded,
            "no_new_filesystem_discovery": True,
        },
        "candidate_roots": [roots[key] for key in sorted(roots)],
        "new_frame_candidates": new_candidates,
        "recovered_synthetic_families": recovered,
        "exact_duplicate_content_relationships": sorted(exact_duplicate_pairs, key=lambda row: (row["path"], row["duplicate_of"])),
        "unresolved_inventory_conflicts": conflicts,
        "conflict_review_ledger": review_report,
        "summary": {
            "candidate_root_count": len(roots),
            "directly_verified_root_count": sum(r["provenance_classification"] == "DIRECTLY_VERIFIED" for r in roots.values()),
            "inferred_root_count": sum(r["provenance_classification"] == "INFERRED" for r in roots.values()),
            "missing_root_count": sum(r["provenance_classification"] == "MISSING" for r in roots.values()),
            "conflicting_root_count": sum(r["provenance_classification"] == "CONFLICTING" for r in roots.values()),
            "externally_attested_root_count": 0,
            "recovered_synthetic_family_id_count": len(recovered),
            "recovered_synthetic_record_count": sum(row["record_count"] for row in recovered),
            "eligible_independent_families": sum(item.get("eligible") is True for item in new_candidates),
            "inventory_conflict_count": len(conflicts) if isinstance(conflicts, list) else None,
            "exact_duplicate_relationship_count": len(exact_duplicate_pairs),
            "all_inventory_exact_duplicate_content_count": summary.get("duplicate_content_count"),
            "prior_circuit_record_count": summary.get("circuit_record_count"),
            "unknown_lineage_circuit_record_count": summary.get("unknown_lineage_circuit_count"),
            "family_only_without_root_record_count": summary.get("family_only_without_root_record_count"),
            "source_declared_possible_root_count": summary.get("source_declared_possible_root_count"),
            "observed_lineage_root_count": summary.get("observed_lineage_root_count"),
            "frame_candidate_count": len(frame_candidates) if isinstance(frame_candidates, list) else None,
            "ambiguous_family_id_conflict_count": sum(
                isinstance(item, Mapping) and item.get("kind") == "family_id_maps_to_multiple_roots"
                for item in conflicts
            ) if isinstance(conflicts, list) else None,
            "recovery_rows_missing_recipe_ancestry": recovery_summary.get("missing_lineage_rows"),
            "recovery_rows_missing_per_run_source_hash": recovery_summary.get("missing_run_source_hash_rows"),
            "recovery_rows_missing_independent_adjudication": recovery_summary.get("independent_adjudication_rows") == 0,
        },
        "disposition": "BLOCKED; saved inventory incomplete and truncated, family frame empty, lineage conflicts unresolved, recovery rows lack source ancestry and independent truth.",
    }
    reconciliation_basis = {
        "input_hashes": report["input_hashes"],
        "current_evidence_hashes": current_file_hashes,
        "current_recovery_hashes": recovery_current,
        "unresolved_inventory_conflicts": conflicts,
        "conflict_review_ledger": review_report,
    }
    # This stable basis excludes decision fields that depend on the attestation
    # itself, avoiding a circular signature while binding all source evidence.
    report["reconciliation_sha256"] = _sha(_canonical(reconciliation_basis))
    report["report_sha256"] = _sha(_canonical(report))
    return report


def validate_independent_adjudication_records(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
    verified_adjudicator_attestations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate blinded adjudication evidence; trust attestations externally.

    Raw pseudonymous tokens are hashes. Self-declared blinding/independence
    fields are insufficient: each judgment must be covered by an externally
    verified case-, protocol-, and token-bound attestation.
    """
    if not _is_sha(protocol_sha256):
        raise ProtocolRecordError("adjudication gate requires a valid protocol SHA-256")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ProtocolRecordError("adjudication records must be a sequence")
    if not isinstance(verified_adjudicator_attestations, Sequence) or isinstance(verified_adjudicator_attestations, (str, bytes)):
        raise ProtocolRecordError("verified adjudicator attestations must be a sequence")
    attestations = {}
    for attestation in verified_adjudicator_attestations:
        if not isinstance(attestation, Mapping):
            continue
        key = (attestation.get("case_id"), attestation.get("adjudicator_token_sha256"))
        attestations[key] = attestation
    seen_cases: set[str] = set()
    family_owner: dict[str, str] = {}
    lineage_owner: dict[str, str] = {}
    execution_owner: dict[str, str] = {}
    eligible: list[str] = []
    blockers: list[str] = []
    if not records:
        blockers.append("no independent blinded adjudication records were supplied")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolRecordError("each adjudication case must be a mapping")
        forbidden = _forbidden_record_keys(record)
        if forbidden & SYNTHETIC_OR_PREDICTIVE_FIELDS:
            raise ProtocolRecordError("scenario labels and system predictions are not adjudication truth: " + ", ".join(sorted(forbidden)))
        if forbidden & RAW_IDENTITY_FIELDS:
            raise ProtocolRecordError("raw adjudicator identities are prohibited; use externally verified opaque tokens")
        case_id, family, lineage = record.get("case_id"), record.get("family_root_id"), record.get("lineage_root_id")
        if not all(isinstance(value, str) and value.strip() for value in (case_id, family, lineage)):
            raise ProtocolRecordError("case_id, family_root_id, and lineage_root_id are required")
        if case_id in seen_cases:
            raise ProtocolRecordError(f"duplicate case_id {case_id!r}; repeated executions belong inside one case")
        seen_cases.add(case_id)
        for mapping, key, label in ((family_owner, family, "family"), (lineage_owner, lineage, "lineage")):
            previous = mapping.setdefault(key, case_id)
            if previous != case_id:
                raise ProtocolRecordError(f"repeated executions/cases share one {label} root; they are not independent cases")
        if record.get("protocol_sha256") != protocol_sha256:
            blockers.append(f"case {case_id}: adjudication record is not bound to the frozen protocol hash")
        if not isinstance(record.get("protocol_version"), str) or not record["protocol_version"].strip():
            blockers.append(f"case {case_id}: adjudication protocol version is missing")
        evidence = record.get("evidence_refs")
        if not isinstance(evidence, list) or not evidence:
            blockers.append(f"case {case_id}: immutable evidence references are missing")
        elif any(not isinstance(item, Mapping) or not _is_sha(item.get("sha256")) or not isinstance(item.get("evidence_id"), str) or not item["evidence_id"].strip() for item in evidence):
            blockers.append(f"case {case_id}: evidence references need stable IDs and SHA-256 digests")
        judgments = record.get("independent_judgments")
        if not isinstance(judgments, list) or len(judgments) < 2:
            blockers.append(f"case {case_id}: at least two separately recorded independent judgments are required")
            continue
        values = []
        tokens: set[str] = set()
        all_attested = True
        for judgment in judgments:
            if not isinstance(judgment, Mapping):
                blockers.append(f"case {case_id}: malformed judgment record")
                all_attested = False
                continue
            token = judgment.get("adjudicator_token_sha256")
            value = judgment.get("judgment")
            if not _is_sha(token) or token in tokens:
                blockers.append(f"case {case_id}: adjudicator tokens must be distinct privacy-preserving SHA-256 identifiers")
                all_attested = False
            tokens.add(token)
            if value not in JUDGMENTS:
                blockers.append(f"case {case_id}: unsupported judgment disposition")
                all_attested = False
            values.append(value)
            if judgment.get("blinding_status") != "BLINDED_TO_SYSTEM_OUTPUTS":
                blockers.append(f"case {case_id}: judgment is not recorded as blinded to system outputs")
                all_attested = False
            att = attestations.get((case_id, token), {})
            if not (
                att.get("status") == "VERIFIED"
                and att.get("protocol_sha256") == protocol_sha256
                and att.get("case_id") == case_id
                and att.get("adjudicator_token_sha256") == token
                and att.get("identity_tokenization_verified") is True
                and att.get("blinding_verified") is True
                and att.get("independence_verified") is True
                and _is_sha(att.get("attestation_sha256"))
                and isinstance(att.get("verified_by"), str) and att.get("verified_by").strip()
            ):
                blockers.append(f"case {case_id}: external blinding/independence attestation is missing or mismatched")
                all_attested = False
            judgment_evidence = judgment.get("evidence_refs")
            if not isinstance(judgment_evidence, list) or not judgment_evidence or any(not _is_sha(value) for value in judgment_evidence):
                blockers.append(f"case {case_id}: each judgment needs hashed evidence references")
                all_attested = False
        resolution = record.get("resolution")
        if not isinstance(resolution, Mapping):
            blockers.append(f"case {case_id}: disagreement disposition/resolution is missing")
            continue
        status = resolution.get("status")
        if len(set(values)) == 1:
            if status != "AGREED":
                blockers.append(f"case {case_id}: identical judgments must be recorded as AGREED")
                all_attested = False
        elif status == "UNRESOLVED":
            all_attested = False
            blockers.append(f"case {case_id}: conflicting judgments remain unresolved")
        elif status == "DISAGREEMENT_RESOLVED":
            resolver = resolution.get("resolver_token_sha256")
            resolution_attestation = attestations.get((case_id, resolver), {})
            resolver_verified = (
                _is_sha(resolver) and resolver not in tokens
                and resolution_attestation.get("status") == "VERIFIED"
                and resolution_attestation.get("protocol_sha256") == protocol_sha256
                and resolution_attestation.get("identity_tokenization_verified") is True
                and resolution_attestation.get("blinding_verified") is True
                and resolution_attestation.get("independence_verified") is True
                and _is_sha(resolution_attestation.get("attestation_sha256"))
                and isinstance(resolution_attestation.get("verified_by"), str)
                and bool(resolution_attestation.get("verified_by", "").strip())
            )
            if (
                not isinstance(resolution.get("method"), str) or not resolution["method"].strip()
                or not _is_sha(resolution.get("resolution_evidence_sha256"))
                or resolution.get("resolved_judgment") not in JUDGMENTS
                or not resolver_verified
            ):
                blockers.append(
                    f"case {case_id}: resolved disagreement needs method, hashed evidence, resolved disposition, and a separately verified blinded resolver"
                )
                all_attested = False
        else:
            blockers.append(f"case {case_id}: conflicting judgments require explicit resolution or UNRESOLVED")
            all_attested = False
        executions = record.get("execution_ids")
        if not isinstance(executions, list) or not executions or any(not isinstance(item, str) or not item.strip() for item in executions):
            blockers.append(f"case {case_id}: execution_ids are required and remain nested within this family case")
            all_attested = False
        else:
            if len(set(executions)) != len(executions):
                blockers.append(f"case {case_id}: duplicate execution ID")
                all_attested = False
            for execution in executions:
                previous = execution_owner.setdefault(execution, case_id)
                if previous != case_id:
                    raise ProtocolRecordError("one execution ID cannot support multiple independent cases")
        if all_attested and status in {"AGREED", "DISAGREEMENT_RESOLVED"} and len(evidence or []) > 0:
            eligible.append(case_id)
    return {
        "schema_valid": True,
        "independent_adjudication_ready": bool(records) and len(eligible) == len(records) and not blockers,
        "eligible_case_ids": sorted(eligible),
        "independent_family_count": len({record.get("family_root_id") for record in records if isinstance(record, Mapping) and record.get("case_id") in eligible}),
        "blockers": list(dict.fromkeys(blockers)),
        "note": "External attestation validity is assumed only at this API boundary; this validator does not authenticate signatures or reviewer identities.",
    }


def audit_family_split_alignment(
    frame: Mapping[str, Any], protocol_cases: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require confirmatory cases to use exactly the frame's disjoint roots.

    The sampling frame's ``development`` families may feed training or
    calibration; its ``validation`` families map only to the untouched ``test``.
    """
    selected = frame.get("selected", {})
    selected_ids: set[str] = set()
    if isinstance(selected, Mapping):
        for partition in ("development", "validation"):
            for item in selected.get(partition, []):
                value = item if isinstance(item, str) else item.get("family_id") if isinstance(item, Mapping) else None
                if isinstance(value, str):
                    selected_ids.add(value)
    candidate_map: dict[str, str] = {}
    blockers: list[str] = []
    for candidate in frame.get("candidates", []) if isinstance(frame.get("candidates"), list) else []:
        if not isinstance(candidate, Mapping) or candidate.get("family_id") not in selected_ids:
            continue
        lineage = candidate.get("lineage_root_id")
        partition = candidate.get("partition")
        if not isinstance(lineage, str) or partition not in {"development", "validation"}:
            blockers.append("selected family candidate lacks a lineage root or valid development/validation partition")
            continue
        if lineage in candidate_map:
            blockers.append(f"selected lineage root {lineage!r} is assigned more than once")
        candidate_map[lineage] = partition
    observed: set[str] = set()
    for case in protocol_cases:
        if not isinstance(case, Mapping):
            blockers.append("protocol case is malformed during frame/partition alignment")
            continue
        lineage, partition = case.get("lineage_root_id"), case.get("partition")
        if not isinstance(lineage, str) or lineage not in candidate_map:
            blockers.append(f"protocol case {case.get('case_id')!r} lineage is not in the selected family frame")
            continue
        expected = "development" if partition in {"train", "calibration"} else "validation" if partition == "test" else None
        if expected is None or candidate_map[lineage] != expected:
            blockers.append(f"protocol case {case.get('case_id')!r} partition is inconsistent with its selected family lineage")
        if lineage in observed:
            blockers.append(f"protocol lineage {lineage!r} appears in more than one case/partition")
        observed.add(lineage)
    missing = sorted(set(candidate_map) - observed)
    if missing:
        blockers.append("selected family lineages are absent from protocol cases: " + ", ".join(missing))
    return list(dict.fromkeys(blockers))


def audit_confirmatory_study_preflight(
    *, repository: str | Path, inventory: Mapping[str, Any], frame: Mapping[str, Any],
    recovery: Mapping[str, Any], protocol: Mapping[str, Any],
    protocol_cases: Sequence[Mapping[str, Any]], adjudication_records: Sequence[Mapping[str, Any]],
    verified_inventory_attestation: Mapping[str, Any] | None = None,
    verified_frame_attestation: Mapping[str, Any] | None = None,
    verified_review_attestation: Mapping[str, Any] | None = None,
    verified_adjudicator_attestations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run every evidence gate used before confirmatory study execution."""
    from experiments.diagnosis_development_study import StudyBlockedError, audit_materialized_sampling_frame
    from experiments.diagnosis_evaluation_protocol import audit_research_protocol

    blockers: list[str] = []
    reconciliation = reconcile_saved_evidence(
        repository, inventory, frame, recovery,
        verified_inventory_attestation=verified_inventory_attestation,
        verified_frame_attestation=verified_frame_attestation,
    )
    inventory_attestation_ok = (
        isinstance(verified_inventory_attestation, Mapping)
        and verified_inventory_attestation.get("status") == "VERIFIED_COMPLETE"
        and verified_inventory_attestation.get("reconciliation_sha256") == reconciliation.get("reconciliation_sha256")
        and verified_inventory_attestation.get("inventory_sha256") == inventory.get("inventory_sha256")
        and verified_inventory_attestation.get("frame_sha256") == frame.get("frame_sha256")
        and _is_sha(verified_inventory_attestation.get("attestation_sha256"))
        and isinstance(verified_inventory_attestation.get("verified_by"), str)
        and bool(verified_inventory_attestation.get("verified_by", "").strip())
    )
    if not inventory_attestation_ok:
        blockers.append("complete family/lineage reconciliation lacks a matching externally verified inventory attestation")
    if reconciliation["summary"]["eligible_independent_families"] <= 0:
        blockers.append("reconciliation identifies zero eligible independent families; resolve each root/source/duplicate conflict and independently attest completeness")
    try:
        audit_materialized_sampling_frame(
            inventory, frame, verified_reconciliation_attestation=verified_inventory_attestation
        )
    except (StudyBlockedError, ValueError, TypeError) as exc:
        blockers.append(f"sampling frame preflight: {exc}")
    protocol_hash = protocol.get("preregistration_sha256") if isinstance(protocol, Mapping) else None
    adjudication = validate_independent_adjudication_records(
        adjudication_records, protocol_sha256=protocol_hash if isinstance(protocol_hash, str) else "",
        verified_adjudicator_attestations=verified_adjudicator_attestations,
    ) if _is_sha(protocol_hash) else {
        "independent_adjudication_ready": False, "eligible_case_ids": [],
        "independent_family_count": 0, "blockers": ["protocol preregistration SHA-256 is missing or invalid"],
    }
    blockers.extend(f"adjudication gate: {item}" for item in adjudication["blockers"])
    try:
        protocol_result = audit_research_protocol(
            protocol, protocol_cases, verified_review_attestation=verified_review_attestation
        )
    except (ProtocolRecordError, TypeError, ValueError) as exc:
        protocol_result = {"confirmatory_prerequisites_recorded": False, "blockers": [str(exc)]}
    blockers.extend(f"protocol gate: {item}" for item in protocol_result.get("blockers", []))
    blockers.extend(f"family split gate: {item}" for item in audit_family_split_alignment(frame, protocol_cases))
    case_ids = {case.get("case_id") for case in protocol_cases if isinstance(case, Mapping)}
    if adjudication.get("eligible_case_ids") != sorted(case_ids):
        blockers.append("independent adjudication evidence does not cover exactly the protocol case set")
    return {
        "ready": not blockers and protocol_result.get("confirmatory_prerequisites_recorded") is True
            and adjudication.get("independent_adjudication_ready") is True and inventory_attestation_ok,
        "blockers": list(dict.fromkeys(blockers)),
        "reconciliation": reconciliation,
        "adjudication": adjudication,
        "protocol": protocol_result,
        "note": "Preflight is diagnostic only. No study execution authorization is implied.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--inventory", default="reports/experiments/diagnosis_prior_lineage_inventory_20261015.json")
    parser.add_argument("--frame", default="reports/experiments/diagnosis_family_sampling_frame_20261015.json")
    parser.add_argument("--recovery", default="reports/experiments/ml_historical_record_recovery_20261015_v2.json")
    parser.add_argument("--output", required=True, help="Write report to a new path; existing paths are refused")
    parser.add_argument("--reviews", help="Optional hash-bound custodian conflict review JSON; original inventory remains unchanged")
    args = parser.parse_args()
    root = Path(args.repository).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    docs = [json.loads((root / path).read_text(encoding="utf-8")) for path in (args.inventory, args.frame, args.recovery)]
    reviews = json.loads((root / args.reviews).read_text(encoding="utf-8"))["reviews"] if args.reviews else []
    report = reconcile_saved_evidence(root, *docs, conflict_reviews=reviews)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    print(report["disposition"])


if __name__ == "__main__":
    main()


__all__ = [
    "audit_confirmatory_study_preflight", "audit_family_split_alignment", "reconcile_saved_evidence",
    "validate_independent_adjudication_records", "validate_inventory_conflict_reviews",
]
