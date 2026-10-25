"""Fail-closed structural readiness check for the QMutBench evidence manifest.

This checker verifies declared fields, local file presence, SHA-256 values, and
coverage declarations. It cannot authenticate custodians, reviewers, signatures,
or scientific truth. Those gates remain BLOCKED until independent review.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"PASS", "BLOCKED", "INVALID"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(manifest_path: Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Return per-gate structural and evidence status without changing eligibility."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"schema": "qmutbench-evidence-readiness-result-v1", "readiness_status": "INVALID",
                "error": f"manifest cannot be parsed: {exc}", "gates": [],
                "eligibility_decision_changed": False}
    root = Path(base_dir) if base_dir is not None else manifest_path.resolve().parents[2]
    errors: list[str] = []
    if not isinstance(manifest, dict) or manifest.get("schema") != "qmutbench-external-evidence-acquisition-v1":
        return {"schema": "qmutbench-evidence-readiness-result-v1", "readiness_status": "INVALID",
                "error": "unsupported manifest schema or root value", "gates": [],
                "eligibility_decision_changed": False}
    decision = manifest.get("current_eligibility_decision")
    if not isinstance(decision, dict) or decision.get("eligible_independent_recipe_roots") != 0 or decision.get("evaluation_manifest_justified") is not False:
        errors.append("baseline eligibility decision is missing or inconsistent; checker will not update it")
    gates = manifest.get("evidence_gates")
    if not isinstance(gates, list):
        return {"schema": "qmutbench-evidence-readiness-result-v1", "readiness_status": "INVALID",
                "error": "evidence_gates must be a list", "gates": [],
                "eligibility_decision_changed": False}
    result_gates = []
    seen_gates: set[str] = set()
    any_invalid = False
    any_blocked = bool(errors)
    for gate in gates:
        if not isinstance(gate, dict) or not isinstance(gate.get("gate_id"), str) or not isinstance(gate.get("required_artifacts"), list):
            result_gates.append({"gate_id": None, "status": "INVALID", "structural_status": "INVALID",
                                 "independent_verification_status": "BLOCKED", "reasons": ["gate shape is invalid"]})
            any_invalid = True
            continue
        gate_id = gate["gate_id"]
        reasons: list[str] = []
        invalid: list[str] = []
        missing: list[str] = []
        if gate_id in seen_gates:
            invalid.append("duplicate gate_id")
        seen_gates.add(gate_id)
        artifacts = gate["required_artifacts"]
        artifact_ids: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("artifact_id"), str) or not isinstance(artifact.get("required_fields"), list):
                invalid.append("artifact descriptor is malformed")
                continue
            artifact_id = artifact["artifact_id"]
            if artifact_id in artifact_ids:
                invalid.append(f"duplicate artifact_id: {artifact_id}")
            artifact_ids.add(artifact_id)
            required_fields = artifact["required_fields"]
            if any(not isinstance(field, str) or not field for field in required_fields):
                invalid.append(f"invalid required_fields for {artifact_id}")
            provided = artifact.get("provided_fields", [])
            if not isinstance(provided, list) or any(not isinstance(field, str) for field in provided):
                invalid.append(f"provided_fields malformed for {artifact_id}")
                provided = []
            absent = sorted(set(required_fields) - set(provided))
            if absent:
                missing.append(f"{artifact_id}: fields not evidenced: {', '.join(absent)}")
            path_value, digest = artifact.get("path"), artifact.get("sha256")
            if path_value is None and digest is None:
                missing.append(f"{artifact_id}: artifact path and checksum unresolved")
            elif not isinstance(path_value, str) or not path_value or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                invalid.append(f"{artifact_id}: path/checksum pair is malformed")
            else:
                path = (root / path_value).resolve()
                try:
                    path.relative_to(root.resolve())
                except ValueError:
                    invalid.append(f"{artifact_id}: path escapes evidence root")
                    continue
                if not path.is_file():
                    missing.append(f"{artifact_id}: file not present: {path_value}")
                elif _sha256(path) != digest:
                    invalid.append(f"{artifact_id}: SHA-256 mismatch")
            if artifact.get("coverage_complete") is not True:
                missing.append(f"{artifact_id}: coverage not explicitly complete")
        structural_status = "INVALID" if invalid else "BLOCKED" if missing else "PASS"
        if invalid:
            reasons.extend(invalid)
        if missing:
            reasons.extend(missing)
        review = gate.get("external_review")
        if not isinstance(review, dict):
            review_status = "BLOCKED"
            reasons.append("external_review record is missing")
        else:
            review_status = "BLOCKED"
            reasons.append("independent reviewer identity, independence, and attestation authenticity require external verification")
            # Locally self-asserted status values are intentionally not accepted.
        status = "INVALID" if invalid else "BLOCKED"  # evidence gates fail closed pending external verification
        result_gates.append({"gate_id": gate_id, "status": status, "structural_status": structural_status,
                             "independent_verification_status": review_status, "reasons": reasons})
        any_invalid |= bool(invalid)
        any_blocked |= status == "BLOCKED"
    if not gates:
        any_blocked = True
        errors.append("no evidence gates were provided")
    return {
        "schema": "qmutbench-evidence-readiness-result-v1",
        "readiness_status": "INVALID" if any_invalid else "BLOCKED" if any_blocked else "PASS",
        "baseline_eligible_independent_recipe_roots": 0,
        "evaluation_manifest_justified": False,
        "eligibility_decision_changed": False,
        "checker_scope": "structural checks only; external identities, signatures, attestations, and truth are not authenticated",
        "errors": errors,
        "gates": result_gates,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    manifest = here / "reports/experiments/qmutbench_external_evidence_manifest_20261015.json"
    result = check_manifest(manifest, base_dir=here)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
