import hashlib
import json

from qmutbench_evidence_readiness import check_manifest


def _case(tmp_path, *, provided=None, checksum=None, coverage=True, review=None):
    payload = b"pinned evidence\n"
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(payload)
    required_digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "qmutbench-external-evidence-acquisition-v1",
        "current_eligibility_decision": {"eligible_independent_recipe_roots": 0, "evaluation_manifest_justified": False},
        "evidence_gates": [{
            "gate_id": "source",
            "required_artifacts": [{
                "artifact_id": "release", "required_fields": ["commit", "sha256"],
                "provided_fields": ["commit", "sha256"] if provided is None else provided,
                "path": "evidence.bin", "sha256": required_digest if checksum is None else checksum,
                "coverage_complete": coverage,
            }],
            "external_review": review,
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, required_digest


def test_missing_required_fields_fail_closed(tmp_path):
    path, _ = _case(tmp_path, provided=["commit"])
    result = check_manifest(path, base_dir=tmp_path)
    assert result["gates"][0]["structural_status"] == "BLOCKED"
    assert result["gates"][0]["status"] == "BLOCKED"


def test_checksum_mismatch_is_invalid(tmp_path):
    path, _ = _case(tmp_path, checksum="0" * 64)
    result = check_manifest(path, base_dir=tmp_path)
    assert result["gates"][0]["status"] == "INVALID"
    assert any("SHA-256 mismatch" in reason for reason in result["gates"][0]["reasons"])


def test_incomplete_coverage_is_blocked(tmp_path):
    path, _ = _case(tmp_path, coverage=False)
    result = check_manifest(path, base_dir=tmp_path)
    assert result["gates"][0]["structural_status"] == "BLOCKED"


def test_unattested_review_cannot_pass_or_change_eligibility(tmp_path):
    path, _ = _case(tmp_path, review={"status": "unresolved", "reviewer_identity": None})
    result = check_manifest(path, base_dir=tmp_path)
    gate = result["gates"][0]
    assert gate["structural_status"] == "PASS"
    assert gate["independent_verification_status"] == "BLOCKED"
    assert gate["status"] == "BLOCKED"
    assert result["baseline_eligible_independent_recipe_roots"] == 0
    assert result["eligibility_decision_changed"] is False


def test_self_asserted_external_review_does_not_authenticate_truth(tmp_path):
    path, _ = _case(tmp_path, review={"status": "verified", "reviewer_identity": "claimed-reviewer"})
    result = check_manifest(path, base_dir=tmp_path)
    assert result["gates"][0]["status"] == "BLOCKED"
