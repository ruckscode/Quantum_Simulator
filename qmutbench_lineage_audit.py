"""Bounded, deterministic audit of locally staged QMutBench lineage evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_lineage(manifest_path: Path, normalized_path: Path) -> dict[str, Any]:
    """Audit the pinned source bundle and one bounded normalized run; never infer roots from IDs."""
    manifest_path, normalized_path = Path(manifest_path), Path(normalized_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    root = manifest_path.parent
    files = []
    missing, conflicting = [], []
    for entry in manifest.get("qasm_artifacts", {}).get("files", []):
        path = root / "raw" / f"qasm_{manifest['source_commit']}" / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        status = "verified_local_bytes" if actual == entry.get("sha256") else "missing" if actual is None else "hash_conflict"
        files.append({"path": entry["path"], "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "status": status})
        if status == "missing": missing.append(entry["path"])
        if status == "hash_conflict": conflicting.append(entry["path"])
    origins: dict[str, set[str]] = {}
    for record in records:
        origin = record.get("original_circuit_id")
        artifact_hash = record.get("qasm_sha256")
        if origin and artifact_hash:
            origins.setdefault(str(origin), set()).add(str(artifact_hash))
    duplicate_artifacts = sorted(h for hashes in origins.values() for h in hashes if sum(h in s for s in origins.values()) > 1)
    blockers = [
        "Origin_program proves a documented parent grouping only; no custodian-attested independent recipe-root map is local.",
        "QMutBench documentation names MQT Bench as an upstream source, but exact per-origin MQT recipe/ancestor mappings are unresolved.",
        "No independently adjudicated program-behavior labels or reviewer/resolver evidence is present; Killed and simulator observations are not labels.",
        "Only one original group is represented by the bounded local QASM bundle/run, insufficient for family-separated train/validation/test.",
        "The prior-artifact inventory remains incomplete/unreconciled per the existing provenance report.",
    ]
    return {
        "schema": "qmutbench-lineage-audit-v1",
        "source_repository": manifest.get("source_repository"),
        "source_commit": manifest.get("source_commit"),
        "commit_signature_status": manifest.get("commit_signature_status"),
        "metadata": {"path": manifest.get("metadata_csv", {}).get("path"), "expected_sha256": manifest.get("metadata_csv", {}).get("sha256"), "actual_sha256": sha256_file(root / "raw" / "official_all_data_20261015.csv")},
        "identity_source_release": "release-pinned metadata and QASM bundle; local byte hashes verified below",
        "original_circuit_groups": sorted(origins),
        "original_circuit_group_count_in_bounded_records": len(origins),
        "artifact_hashes": files,
        "missing_artifacts": missing,
        "conflicting_artifacts": conflicting,
        "duplicate_artifact_hashes_across_groups": duplicate_artifacts,
        "verified_facts": ["Pinned metadata and locally staged QASM byte hashes can be checked against the local manifest.", "Official Origin_program-to-mutant parent grouping is encoded in metadata and QASM paths."],
        "documented_upstream_facts_not_locally_attested": ["QMutBench documentation describes MQT Bench as an initial original-circuit source.", "The pinned repository commit is reported unsigned in the local source manifest."],
        "unresolved": {"recipe_root_mapping": "missing", "per_origin_mqt_bench_ancestor": "missing", "mutant_ancestry_beyond_parent_group": "documented generation relation; generator/run provenance not attested", "behavioral_label_provenance": "missing independent adjudication", "prior_inventory_reconciliation": "incomplete"},
        "observation_configuration": {"source": "bounded smoke run manifest and observations", "status": "simulator-derived ideal/no-noise computational-basis observations; not independent ground truth"},
        "eligible_recipe_roots": [],
        "eligible_recipe_root_count": 0,
        "evaluation_manifest_justified": False,
        "evaluation_manifest": None,
        "blockers": blockers,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    result = audit_lineage(base / "data/qmutbench_ingestion/source_manifest.json", base / "data/qmutbench_ingestion/runs/smoke_ghz_first12_20261015_v6/normalized_records.jsonl")
    out = base / "reports/qmutbench_lineage_audit_20261015.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
