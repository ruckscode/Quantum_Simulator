"""Fail-closed record checks for independent diagnosis evaluations.

This module validates protocol/data separation only. It does not implement a
diagnosis rule, certify independent truth, or establish scientific validity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


PARTITIONS = {"train", "calibration", "test"}
REVIEW_STATUSES = {
    "INDEPENDENTLY_ADJUDICATED", "SYNTHETIC_ONLY", "MISSING", "CONFLICTING",
}
DISTRIBUTION_STATUSES = {"CHANGE", "NO_CHANGE", "UNRESOLVED", "MISSING", "SYNTHETIC_ONLY"}
CAUSE_DISPOSITIONS = {
    "SINGLE_CAUSE", "AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT",
    "CONFLICTING", "UNRESOLVED", "MISSING", "SYNTHETIC_ONLY",
}
TARGET_DISPOSITIONS = {
    "KNOWN", "NO_TARGET", "AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT",
    "CONFLICTING", "UNRESOLVED", "MISSING", "NOT_APPLICABLE", "SYNTHETIC_ONLY",
}
ENDPOINTS = {
    "detection", "cause_disposition", "single_cause_attribution", "source_localization",
    "hardware_localization", "abstention",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OBSERVABLE_KEYS = {
    "adjudicator_only", "case_id", "family_root_id", "lineage_root_id",
    "partition", "truth", "ground_truth", "label", "labels", "synthetic_label",
    "true_label", "ground_truth_label", "diagnosis_label", "cause_label", "target_label",
    "mutation_id", "mutant_id", "fault_label", "fault_type", "fault_class",
    "fault_target", "fault_targets", "source_target_truth", "hardware_target_truth",
    "simulator_state", "internal_state", "statevector", "internal_run_report",
    "diagnostic_result", "program_evidence_candidates", "hardware_evidence_candidates",
    "trace_deviation_candidates", "localization_candidates", "anomaly_decision",
    "anomaly_detected", "category", "prediction", "confidence", "injected_fault",
    "scenario_label", "fault_manifest", "adjudication", "adjudication_status",
    "review_status", "reviewer_id", "reviewer_ids", "adjudicator_ids", "evidence_refs",
    "distribution_truth", "cause_disposition", "possible_causes", "source_targets",
    "hardware_targets", "source_localization_status", "hardware_localization_status",
}


class ProtocolRecordError(ValueError):
    """Raised when record structure, blinding, or family separation is invalid."""


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_observable_inputs(value: Any, case_id: str, required_endpoints: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolRecordError(f"case {case_id!r}: observable_inputs must be a mapping")
    needed = {
        "reference_operations", "submitted_operations", "reference_counts",
        "submitted_counts", "sampling_design", "measurement_schema",
    }
    if "hardware_localization" in required_endpoints:
        needed.add("hardware_telemetry")
    missing = sorted(needed - set(value))
    if missing:
        raise ProtocolRecordError(
            f"case {case_id!r}: observable_inputs missing required evidence: {', '.join(missing)}"
        )
    for field in ("reference_operations", "submitted_operations"):
        if not isinstance(value[field], Sequence) or isinstance(value[field], (str, bytes)):
            raise ProtocolRecordError(f"case {case_id!r}: {field} must be an operation sequence")
    for field in ("reference_counts", "submitted_counts"):
        counts = value[field]
        if not isinstance(counts, Mapping) or not counts:
            raise ProtocolRecordError(f"case {case_id!r}: {field} must be a nonempty outcome-count mapping")
        if any(not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0
               for key, count in counts.items()):
            raise ProtocolRecordError(f"case {case_id!r}: {field} must contain nonnegative integer counts")
        if sum(counts.values()) <= 0:
            raise ProtocolRecordError(f"case {case_id!r}: {field} must contain at least one shot")
    design = value["sampling_design"]
    if not isinstance(design, Mapping):
        raise ProtocolRecordError(f"case {case_id!r}: sampling_design must be a mapping")
    for role, count_field in (("reference_shots", "reference_counts"), ("submitted_shots", "submitted_counts")):
        shots = design.get(role)
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ProtocolRecordError(f"case {case_id!r}: sampling_design.{role} must be a positive integer")
        if shots != sum(value[count_field].values()):
            raise ProtocolRecordError(f"case {case_id!r}: {role} does not match recorded count total")
    if not isinstance(value["measurement_schema"], Mapping) or not value["measurement_schema"]:
        raise ProtocolRecordError(f"case {case_id!r}: measurement_schema must be a nonempty mapping")
    if "hardware_telemetry" in needed and value["hardware_telemetry"] is not None and not isinstance(
        value["hardware_telemetry"], Mapping
    ):
        raise ProtocolRecordError(f"case {case_id!r}: hardware_telemetry must be a mapping or null")


def _forbidden_paths(value: Any, prefix: str = "observable_inputs") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            path = f"{prefix}.{key}"
            if normalized in _FORBIDDEN_OBSERVABLE_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def _require_adjudication(record: Any, case_id: str) -> None:
    if not isinstance(record, Mapping):
        raise ProtocolRecordError(f"case {case_id!r}: adjudicator_only must be a mapping")
    required = {
        "review_status", "distribution_status", "cause_disposition",
        "possible_causes", "source_localization_status", "source_targets",
        "hardware_localization_status", "hardware_targets", "adjudicator_ids",
        "evidence_refs",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ProtocolRecordError(f"case {case_id!r}: missing adjudication fields: {', '.join(missing)}")
    if record["review_status"] not in REVIEW_STATUSES:
        raise ProtocolRecordError(f"case {case_id!r}: unsupported review_status")
    if record["distribution_status"] not in DISTRIBUTION_STATUSES:
        raise ProtocolRecordError(f"case {case_id!r}: unsupported distribution_status")
    if record["cause_disposition"] not in CAUSE_DISPOSITIONS:
        raise ProtocolRecordError(f"case {case_id!r}: unsupported cause_disposition")
    for field in ("source_localization_status", "hardware_localization_status"):
        if record[field] not in TARGET_DISPOSITIONS:
            raise ProtocolRecordError(f"case {case_id!r}: unsupported {field}")
    causes = record["possible_causes"]
    if not isinstance(causes, list) or any(not _nonempty(item) for item in causes):
        raise ProtocolRecordError(f"case {case_id!r}: possible_causes must be a list of nonempty names")
    if len(set(causes)) != len(causes):
        raise ProtocolRecordError(f"case {case_id!r}: possible_causes contains duplicates")
    disposition = record["cause_disposition"]
    if disposition == "SINGLE_CAUSE" and len(causes) != 1:
        raise ProtocolRecordError(f"case {case_id!r}: SINGLE_CAUSE requires exactly one possible cause")
    if disposition in {"AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT"} and len(causes) < 2:
        raise ProtocolRecordError(
            f"case {case_id!r}: {disposition} must preserve at least two competing causes"
        )
    for status_field, targets_field in (
        ("source_localization_status", "source_targets"),
        ("hardware_localization_status", "hardware_targets"),
    ):
        targets = record[targets_field]
        if not isinstance(targets, list) or any(not _nonempty(item) for item in targets):
            raise ProtocolRecordError(f"case {case_id!r}: {targets_field} must be a list of nonempty names")
        status = record[status_field]
        if status == "KNOWN" and not targets:
            raise ProtocolRecordError(f"case {case_id!r}: {status_field}=KNOWN requires target evidence")
        if status in {"NO_TARGET", "NOT_APPLICABLE"} and targets:
            raise ProtocolRecordError(f"case {case_id!r}: {status_field}={status} cannot contain targets")
    for field in ("adjudicator_ids", "evidence_refs"):
        values = record[field]
        if not isinstance(values, list) or any(not _nonempty(item) for item in values):
            raise ProtocolRecordError(f"case {case_id!r}: {field} must be a list of nonempty values")
    if record["review_status"] == "INDEPENDENTLY_ADJUDICATED":
        if not record["adjudicator_ids"] or not record["evidence_refs"]:
            raise ProtocolRecordError(
                f"case {case_id!r}: independent adjudication requires reviewer identity and evidence references"
            )


def audit_research_protocol(
    protocol: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    verified_review_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate blinded case records and report confirmatory blockers.

    Invalid record structure or leakage raises :class:`ProtocolRecordError`.
    Missing preregistration, review, or independent truth is returned as an
    actionable blocker; no prediction or outcome is computed here. The optional
    ``verified_review_attestation`` must be supplied by a caller that has
    checked an external reviewer attestation against its trust policy; protocol
    JSON fields alone are never treated as proof of approval.
    """
    if not isinstance(protocol, Mapping):
        raise ProtocolRecordError("protocol must be a mapping")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ProtocolRecordError("cases must be a nonempty sequence of records")
    stage = protocol.get("stage", "development")
    if stage not in {"development", "confirmatory"}:
        raise ProtocolRecordError("protocol stage must be 'development' or 'confirmatory'")
    required_endpoints = protocol.get("required_endpoints", [])
    if not isinstance(required_endpoints, list) or any(item not in ENDPOINTS for item in required_endpoints):
        raise ProtocolRecordError("required_endpoints must contain only registered endpoint names")

    seen_ids: set[str] = set()
    family_partitions: dict[str, set[str]] = defaultdict(set)
    lineage_partitions: dict[str, set[str]] = defaultdict(set)
    family_digests: dict[str, str] = {}
    blockers: list[str] = []
    endpoint_eligible: dict[str, list[str]] = {endpoint: [] for endpoint in ENDPOINTS}
    test_cases: list[Mapping[str, Any]] = []

    required_case_fields = {
        "case_id", "family_root_id", "lineage_root_id", "partition",
        "family_fingerprint_sha256", "observable_inputs", "provenance", "adjudicator_only",
    }
    for case in cases:
        if not isinstance(case, Mapping):
            raise ProtocolRecordError("every case must be a mapping")
        missing = sorted(required_case_fields - set(case))
        if missing:
            raise ProtocolRecordError(f"case missing required fields: {', '.join(missing)}")
        case_id = case["case_id"]
        if not _nonempty(case_id):
            raise ProtocolRecordError("case_id must be a nonempty string")
        if case_id in seen_ids:
            raise ProtocolRecordError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        family, lineage, partition = case["family_root_id"], case["lineage_root_id"], case["partition"]
        if not _nonempty(family) or not _nonempty(lineage):
            raise ProtocolRecordError(f"case {case_id!r}: family and lineage roots are required")
        if partition not in PARTITIONS:
            raise ProtocolRecordError(f"case {case_id!r}: partition must be train, calibration, or test")
        family_partitions[family].add(partition)
        lineage_partitions[lineage].add(partition)
        digest = case["family_fingerprint_sha256"]
        if not _sha256(digest):
            raise ProtocolRecordError(f"case {case_id!r}: family_fingerprint_sha256 must be a SHA-256 digest")
        prior_family = family_digests.setdefault(digest, family)
        if prior_family != family:
            raise ProtocolRecordError(
                f"case {case_id!r}: exact duplicate family fingerprint appears under distinct family roots"
            )
        observable = case["observable_inputs"]
        _validate_observable_inputs(observable, case_id, required_endpoints)
        leaked = _forbidden_paths(observable)
        if leaked:
            raise ProtocolRecordError(
                f"case {case_id!r}: adjudicator-only or identity fields leaked into observable_inputs: "
                + ", ".join(leaked)
            )
        provenance = case["provenance"]
        if not isinstance(provenance, Mapping):
            raise ProtocolRecordError(f"case {case_id!r}: provenance must be a mapping")
        for field in ("source_artifact_sha256", "observation_artifact_sha256", "protocol_sha256"):
            if not _sha256(provenance.get(field)):
                blockers.append(f"case {case_id}: missing or invalid {field}")
        runtime = provenance.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or not _nonempty(runtime.get("python_version"))
            or not _nonempty(runtime.get("python_implementation"))
            or not isinstance(runtime.get("libraries"), Mapping)
            or not runtime.get("libraries")
            or any(not _nonempty(name) or not _nonempty(version) for name, version in runtime.get("libraries", {}).items())
        ):
            blockers.append(f"case {case_id}: runtime provenance is missing")
        required_provenance = (
            "generation_id", "source_revision", "circuit_sha256",
            "configuration_sha256", "reproduction_record_sha256",
        )
        for field in required_provenance:
            if not _nonempty(provenance.get(field)):
                blockers.append(f"case {case_id}: missing {field}")
        for field in ("circuit_sha256", "configuration_sha256", "reproduction_record_sha256"):
            if not _sha256(provenance.get(field)):
                blockers.append(f"case {case_id}: missing or invalid {field}")
        ancestors = provenance.get("ancestor_root_ids")
        if not isinstance(ancestors, list) or any(not _nonempty(item) for item in ancestors):
            blockers.append(f"case {case_id}: lineage ancestor_root_ids are missing or malformed")
        seed = provenance.get("seed")
        if isinstance(seed, bool) or not (isinstance(seed, int) or _nonempty(seed)):
            blockers.append(f"case {case_id}: random seed provenance is missing")
        noise = provenance.get("noise_configuration")
        if not isinstance(noise, Mapping) or not noise:
            blockers.append(f"case {case_id}: noise_configuration provenance is missing")
        execution = provenance.get("execution_provenance")
        if (
            not isinstance(execution, Mapping)
            or not _nonempty(execution.get("execution_id"))
            or not _nonempty(execution.get("backend_kind"))
            or not _sha256(execution.get("executor_source_sha256"))
            or not _sha256(execution.get("execution_configuration_sha256"))
        ):
            blockers.append(
                f"case {case_id}: execution_provenance needs execution ID, backend kind, executor source hash, and execution configuration hash"
            )
        _require_adjudication(case["adjudicator_only"], case_id)
        if partition == "test":
            test_cases.append(case)

    split_errors = [f"family root {root!r} crosses partitions" for root, parts in family_partitions.items() if len(parts) > 1]
    split_errors += [f"lineage root {root!r} crosses partitions" for root, parts in lineage_partitions.items() if len(parts) > 1]
    if split_errors:
        raise ProtocolRecordError("; ".join(split_errors))

    if stage == "confirmatory":
        if not _nonempty(protocol.get("target_population")):
            blockers.append("target population is not defined")
        if not _nonempty(protocol.get("intended_use")):
            blockers.append("intended use is not defined")
        for field in (
            "sampling_frame_sha256", "split_manifest_sha256", "preregistration_sha256",
            "acceptance_criteria_sha256", "power_analysis_sha256", "resource_budget_sha256",
        ):
            if not _sha256(protocol.get(field)):
                blockers.append(f"{field} is missing or invalid")
        review_ok = (
            isinstance(verified_review_attestation, Mapping)
            and verified_review_attestation.get("status") == "APPROVED"
            and _nonempty(verified_review_attestation.get("review_id"))
            and verified_review_attestation.get("review_id") == protocol.get("independent_review_id")
            and _sha256(verified_review_attestation.get("reviewed_protocol_sha256"))
            and verified_review_attestation.get("reviewed_protocol_sha256") == protocol.get("preregistration_sha256")
            and _nonempty(verified_review_attestation.get("verified_by"))
        )
        if not review_ok:
            blockers.append(
                "independent review approval is not backed by an externally verified, protocol-bound attestation"
            )
        if not _nonempty(protocol.get("acceptance_criteria_id")):
            blockers.append("use-specific acceptance criteria are not identified")
        if not required_endpoints:
            blockers.append("no confirmatory endpoints are declared")
        present_partitions = {case["partition"] for case in cases}
        if present_partitions != PARTITIONS:
            blockers.append("train, calibration, and untouched test partitions must all be present")
        if not test_cases:
            blockers.append("untouched test partition is empty")

    for case in cases:
        if case["partition"] != "test":
            continue
        case_id = case["case_id"]
        truth = case["adjudicator_only"]
        independent = truth["review_status"] == "INDEPENDENTLY_ADJUDICATED"
        distribution_known = truth["distribution_status"] in {"CHANGE", "NO_CHANGE"}
        cause_known = truth["cause_disposition"] in {
            "SINGLE_CAUSE", "AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT",
        }
        if independent and distribution_known:
            endpoint_eligible["detection"].append(case_id)
        if independent and cause_known:
            endpoint_eligible["cause_disposition"].append(case_id)
            endpoint_eligible["abstention"].append(case_id)
        if independent and truth["cause_disposition"] == "SINGLE_CAUSE":
            endpoint_eligible["single_cause_attribution"].append(case_id)
        if independent and truth["source_localization_status"] in {
            "KNOWN", "NO_TARGET", "AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT", "NOT_APPLICABLE",
        }:
            endpoint_eligible["source_localization"].append(case_id)
        if independent and truth["hardware_localization_status"] in {
            "KNOWN", "NO_TARGET", "AMBIGUOUS", "OBSERVATIONALLY_EQUIVALENT", "NOT_APPLICABLE",
        }:
            endpoint_eligible["hardware_localization"].append(case_id)

    if stage == "confirmatory":
        for endpoint in required_endpoints:
            absent = sorted(set(case["case_id"] for case in test_cases) - set(endpoint_eligible[endpoint]))
            if absent:
                blockers.append(f"test cases lack independent truth for required endpoint {endpoint}: {', '.join(absent)}")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_valid": True,
        "confirmatory_prerequisites_recorded": stage == "confirmatory" and not blockers,
        "blockers": blockers,
        "test_case_count": len(test_cases),
        "test_lineage_roots": len({case["lineage_root_id"] for case in test_cases}),
        "endpoint_eligible_test_case_ids": endpoint_eligible,
        "note": "Recorded prerequisites do not validate label quality, statistical power, method performance, or external validity.",
    }


__all__ = ["ProtocolRecordError", "audit_research_protocol"]
