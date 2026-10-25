import hashlib
import json

from qmutbench_lineage_audit import audit_lineage


def _fixture(tmp_path):
    base = tmp_path / "data"
    manifest_dir = base / "ingestion"
    qasm = manifest_dir / "raw" / "qasm_commit" / "Origin_programs" / "a.qasm"
    qasm.parent.mkdir(parents=True)
    qasm.write_bytes(b"OPENQASM 2.0;\n")
    digest = hashlib.sha256(qasm.read_bytes()).hexdigest()
    (manifest_dir / "raw" / "official_all_data_20261015.csv").write_bytes(b"x")
    manifest = {"source_repository": "repo", "source_commit": "commit", "metadata_csv": {"path": "metadata.csv", "sha256": hashlib.sha256(b"x").hexdigest()}, "qasm_artifacts": {"files": [{"path": "Origin_programs/a.qasm", "sha256": digest}]}}
    mpath = manifest_dir / "source_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    npath = tmp_path / "records.jsonl"
    npath.write_text(json.dumps({"original_circuit_id": "origin-a", "qasm_sha256": digest}) + "\n", encoding="utf-8")
    return mpath, npath


def test_hashes_are_checked_and_identity_is_not_promoted_to_root(tmp_path):
    manifest, records = _fixture(tmp_path)
    audit = audit_lineage(manifest, records)
    assert audit["artifact_hashes"][0]["status"] == "verified_local_bytes"
    assert audit["original_circuit_groups"] == ["origin-a"]
    assert audit["eligible_recipe_root_count"] == 0
    assert audit["evaluation_manifest_justified"] is False


def test_missing_and_conflicting_sources_are_explicit(tmp_path):
    manifest, records = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["qasm_artifacts"]["files"].append({"path": "Origin_programs/missing.qasm", "sha256": "0" * 64})
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    audit = audit_lineage(manifest, records)
    assert audit["missing_artifacts"] == ["Origin_programs/missing.qasm"]


def test_same_origin_multiple_mutants_remain_one_group(tmp_path):
    manifest, records = _fixture(tmp_path)
    line = json.loads(records.read_text())
    line["qasm_sha256"] = "f" * 64
    records.write_text(json.dumps(line) + "\n", encoding="utf-8")
    audit = audit_lineage(manifest, records)
    assert audit["original_circuit_group_count_in_bounded_records"] == 1
    assert audit["eligible_recipe_root_count"] == 0
