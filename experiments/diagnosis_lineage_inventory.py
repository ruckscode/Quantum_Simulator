"""Deterministic, conservative inventory and frame builder for diagnosis studies.

The scanner preserves observed provenance; it deliberately never infers a root
from a family label or treats absent lineage/fingerprints as evidence of novelty.
It performs filesystem/manifest analysis only and does not invoke the simulator.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
SAMPLER_VERSION = "sha256-ranked-family-frame-v1"
DEFAULT_SCAN_ROOTS = ("experiments", "tests/fixtures", "reports/experiments", "data", "ml_results", "examples")
MAX_SCAN_FILES = 20000
MAX_SCAN_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 25 * 1024 * 1024
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}
CONTROL_NAMES = {
    "diagnosis_prior_lineage_inventory_20261015.json",
    "diagnosis_family_sampling_frame_20261015.json",
    "autonomous_stabilization_status.json",
    "autonomous_stabilization_journal.md",
    "project_wide_completion_backlog.json",
    "project_wide_completion_report.md",
    "diagnosis_development_validation_plan_20261015.json",
    "project_claims_reconciliation_20261014.md",
}
PROVENANCE_KEYS = {
    "family_id", "circuit_family_id", "lineage_root_id", "circuit_lineage_id",
    "parent_id", "parent_ids", "ancestor_ids", "ancestor_root_ids",
    "mutation_history", "transform_history", "seed", "master_seed",
    "source_sha256", "generator_version", "configuration_sha256",
    "topology_sha256", "circuit_fingerprint", "fingerprint",
}
CIRCUIT_KEYS = {"circuit", "reference_circuit", "submitted_circuit", "instructions", "gates"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _walk_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else key
            yield child_path, value[key]
            yield from _walk_values(value[key], child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_values(item, f"{path}[{index}]")


def _usage(relative: str) -> str:
    text = relative.lower()
    if "finite_shot" in text or "finite-shot" in text or "calibration" in text:
        return "previously_inspected_calibration_or_validation"
    if "reference_diagnosis_generalization_cases" in text or "heldout" in text or "held_out" in text:
        return "frozen_heldout_or_reference_fixture"
    if "controlled" in text or "fault" in text:
        return "controlled_fault_artifact"
    if "full_dataset" in text or "ml_" in text or "dataset" in text:
        return "historical_synthetic_ml_or_dataset"
    if "phase5_benchmark" in text or "qmutbench" in text:
        return "benchmark_or_simulator_output"
    return "other_prior_artifact"


def _extract_artifact(relative: str, raw: bytes) -> dict[str, Any]:
    record: dict[str, Any] = {
        "artifact_id": _sha(relative.encode("utf-8") + b"\0" + raw),
        "path": relative.replace("\\", "/"),
        "artifact_sha256": _sha(raw),
        "artifact_type": Path(relative).suffix.lower().lstrip(".") or "file",
        "use": _usage(relative),
        "raw_provenance": {},
        "circuit_records": [],
        "parse_status": "not_structured",
    }
    try:
        if relative.lower().endswith(".json"):
            value = json.loads(raw.decode("utf-8"))
            record["parse_status"] = "parsed"
        elif relative.lower().endswith(".jsonl"):
            value = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            record["parse_status"] = "parsed"
        else:
            return record
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        record["parse_status"] = f"parse_error:{type(exc).__name__}"
        return record
    flat = list(_walk_values(value))
    record["raw_provenance"] = {
        path: item for path, item in flat
        if path.rsplit(".", 1)[-1].split("[", 1)[0] in PROVENANCE_KEYS
    }
    for path, item in flat:
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if leaf in CIRCUIT_KEYS and isinstance(item, (dict, list)):
            topology = _sha(_canonical(item))
            prov = record["raw_provenance"]
            record["circuit_records"].append({
                "field_path": path,
                "topology_sha256": topology,
                "fingerprint_method": "sha256-canonical-json-exact-structure-v1",
                "lineage_root_values": [v for k, v in prov.items() if k.endswith(("lineage_root_id", "circuit_lineage_id"))],
                "family_values": [v for k, v in prov.items() if k.endswith(("family_id", "circuit_family_id"))],
                "ancestor_values": [v for k, v in prov.items() if "ancestor" in k or "parent" in k],
                "lineage_status": "explicit_root_present" if any(k.endswith(("lineage_root_id", "circuit_lineage_id")) for k in prov) else "unknown",
                "eligibility": "excluded_prior_artifact",
                "eligibility_reason": "prior artifact or descendant; raw lineage retained, never eligible for fresh frame",
            })
    return record


def discover_scan_roots(repository: str | Path) -> tuple[str, ...]:
    """Include retained evidence/output directories, while excluding environments and this run's tempdirs."""
    root = Path(repository).resolve()
    found = list(DEFAULT_SCAN_ROOTS)
    retained = []
    for child in root.iterdir():
        name = child.name
        if not child.is_dir():
            continue
        is_retained_output = name.startswith((".pwb", ".pytest-tmp", ".t", ".tcal", ".qsr"))
        if not is_retained_output or name.startswith(".t-lineage") or name.startswith(".pwb-clean-env"):
            continue
        retained.append(child.relative_to(root).as_posix())
    return tuple(found + sorted(retained))


def build_inventory(repository: str | Path, scan_roots: Iterable[str] | None = None) -> dict[str, Any]:
    root = Path(repository).resolve()
    scan_roots = tuple(dict.fromkeys(scan_roots if scan_roots is not None else discover_scan_roots(root)))
    records: list[dict[str, Any]] = []
    missing_roots: list[str] = []
    skipped_large_files: list[str] = []
    scan_truncated = False
    bytes_scanned = 0
    for scan_root in scan_roots:
        base = (root / scan_root).resolve()
        if not base.is_relative_to(root) or not base.exists():
            missing_roots.append(scan_root)
            continue
        for current, dirs, files in os.walk(base):
            dirs[:] = sorted((d for d in dirs if d not in EXCLUDED_DIRS), key=lambda d: (d.casefold(), d))
            for filename in sorted(files, key=lambda name: (name.casefold(), name)):
                path = Path(current) / filename
                if path.name in CONTROL_NAMES:
                    continue
                if len(records) >= MAX_SCAN_FILES:
                    scan_truncated = True
                    break
                rel = path.relative_to(root).as_posix()
                try:
                    size = path.stat().st_size
                    if size > MAX_FILE_BYTES:
                        skipped_large_files.append(rel)
                        continue
                    if bytes_scanned + size > MAX_SCAN_BYTES:
                        scan_truncated = True
                        break
                    raw = path.read_bytes()
                    bytes_scanned += len(raw)
                    records.append(_extract_artifact(rel, raw))
                except OSError as exc:
                    records.append({"path": rel, "parse_status": f"read_error:{type(exc).__name__}"})
            if scan_truncated:
                break
        if scan_truncated:
            break
    # Exact duplicate content is retained as separate provenance rows and linked.
    first_by_hash: dict[str, str] = {}
    for item in records:
        digest = item.get("artifact_sha256")
        if digest in first_by_hash:
            item["duplicate_of"] = first_by_hash[digest]
        elif digest:
            first_by_hash[digest] = item["path"]
    circuit_rows = [
        {"artifact_path": item["path"], **row}
        for item in records for row in item.get("circuit_records", [])
    ]
    unknown = sum(row["lineage_status"] != "explicit_root_present" for row in circuit_rows)
    root_topologies: dict[str, set[str]] = {}
    family_roots: dict[str, set[str]] = {}
    for row in circuit_rows:
        for root_id in row["lineage_root_values"]:
            if isinstance(root_id, str) and root_id:
                root_topologies.setdefault(root_id, set()).add(row["topology_sha256"])
        for family_id in row["family_values"]:
            if isinstance(family_id, str) and family_id:
                family_roots.setdefault(family_id, set()).update(
                    root_id for root_id in row["lineage_root_values"] if isinstance(root_id, str) and root_id
                )
    conflicts = []
    conflicts.extend({"kind": "lineage_root_maps_to_multiple_fingerprints", "identifier": root_id,
                      "fingerprints": sorted(fingerprints)}
                     for root_id, fingerprints in sorted(root_topologies.items()) if len(fingerprints) > 1)
    conflicts.extend({"kind": "family_id_maps_to_multiple_roots", "identifier": family_id,
                      "lineage_roots": sorted(roots)}
                     for family_id, roots in sorted(family_roots.items()) if len(roots) > 1)
    root_index: dict[str, dict[str, Any]] = {}
    for artifact in records:
        provenance = artifact.get("raw_provenance", {})
        roots = [value for key, value in provenance.items()
                 if key.endswith(("lineage_root_id", "circuit_lineage_id")) and isinstance(value, str) and value]
        family_ids = [value for key, value in provenance.items()
                      if key.endswith(("family_id", "circuit_family_id")) and isinstance(value, str) and value]
        for root_id in roots:
            item = root_index.setdefault(root_id, {
                "lineage_root_id": root_id, "family_ids": set(), "topology_sha256s": set(),
                "artifact_paths": set(), "ancestor_values": [], "lineage_status": "observed_unadjudicated",
                "eligibility": "excluded_prior_artifact",
            })
            item["family_ids"].update(family_ids)
            item["artifact_paths"].add(artifact["path"])
            item["ancestor_values"].extend(value for key, value in provenance.items()
                                            if "ancestor" in key or "parent" in key)
    for row in circuit_rows:
        for root_id in row["lineage_root_values"]:
            if not isinstance(root_id, str) or not root_id:
                continue
            item = root_index.setdefault(root_id, {
                "lineage_root_id": root_id, "family_ids": set(), "topology_sha256s": set(),
                "artifact_paths": set(), "ancestor_values": [],
                "lineage_status": "observed_unadjudicated", "eligibility": "excluded_prior_artifact",
            })
            item["family_ids"].update(v for v in row["family_values"] if isinstance(v, str) and v)
            item["topology_sha256s"].add(row["topology_sha256"])
            item["artifact_paths"].add(row["artifact_path"])
            item["ancestor_values"].extend(row["ancestor_values"])
    families = [{
        **item,
        "family_ids": sorted(item["family_ids"]),
        "topology_sha256s": sorted(item["topology_sha256s"]),
        "topology_sha256": next(iter(item["topology_sha256s"])) if len(item["topology_sha256s"]) == 1 else None,
        "artifact_paths": sorted(item["artifact_paths"]),
        "ancestor_values": item["ancestor_values"],
        "eligibility_reason": "observed in a prior artifact; root/descendant relationship requires reconciliation and is not eligible for fresh sampling",
    } for _, item in sorted(root_index.items())]
    source_lineage_rules = []
    generator_path = root / "experiments" / "full_dataset.py"
    if generator_path.is_file():
        try:
            generator_bytes = generator_path.read_bytes()
            expression = b'design_recipe_{index % 6 + 1:02d}'
            if expression in generator_bytes:
                declared = [f"design_recipe_{index:02d}" for index in range(1, 7)]
                source_lineage_rules.append({
                    "source_path": "experiments/full_dataset.py",
                    "source_sha256": _sha(generator_bytes),
                    "raw_expression": expression.decode("ascii"),
                    "declared_possible_roots": declared,
                    "status": "source_formula_domain; does not prove each root has a saved descendant",
                })
                existing = {item["lineage_root_id"] for item in families}
                for root_id in declared:
                    if root_id not in existing:
                        families.append({
                            "lineage_root_id": root_id, "family_ids": [], "topology_sha256s": [],
                            "topology_sha256": None, "artifact_paths": ["experiments/full_dataset.py"],
                            "ancestor_values": [], "lineage_status": "source_declared_not_observed_in_saved_rows",
                            "eligibility": "excluded_prior_artifact",
                            "eligibility_reason": "source declares this possible prior recipe root; no matching saved row was reconciled",
                        })
                families.sort(key=lambda item: item["lineage_root_id"])
        except OSError:
            source_lineage_rules.append({"source_path": "experiments/full_dataset.py", "status": "source_read_error"})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inventory_version": "conservative-prior-artifact-inventory-v1",
        "scan_roots": sorted(set(scan_roots)),
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "missing_scan_roots": missing_roots,
        "scan_limits": {"max_files": MAX_SCAN_FILES, "max_total_bytes": MAX_SCAN_BYTES,
                        "max_single_file_bytes": MAX_FILE_BYTES},
        "scan_truncated": scan_truncated,
        "bytes_scanned": bytes_scanned,
        "skipped_large_file_paths": skipped_large_files,
        "artifacts": records,
        "families": families,
        "source_lineage_rules": source_lineage_rules,
        "circuit_records": circuit_rows,
        "summary": {
            "artifact_count": len(records),
            "duplicate_content_count": sum("duplicate_of" in item for item in records),
            "parsed_artifact_count": sum(item.get("parse_status") == "parsed" for item in records),
            "circuit_record_count": len(circuit_rows),
            "observed_lineage_root_count": len(root_index),
            "source_declared_possible_root_count": sum(
                len(rule["declared_possible_roots"]) for rule in source_lineage_rules
            ),
            "source_lineage_rule_count": len(source_lineage_rules),
            "family_only_without_root_record_count": sum(
                bool(item.get("raw_provenance")) and any(
                    key.endswith(("family_id", "circuit_family_id")) for key in item["raw_provenance"]
                ) and not any(key.endswith(("lineage_root_id", "circuit_lineage_id"))
                              for key in item["raw_provenance"])
                for item in records
            ),
            "unknown_lineage_circuit_count": unknown,
            "excluded_prior_circuit_count": len(circuit_rows),
            "conflict_count": len(conflicts),
        },
        "conflicts": conflicts,
        "complete": False,
        "completeness_reason": (
            "A repository scan cannot certify that all previously inspected external, temporary, or unindexed "
            "templates are represented. Every discovered circuit is prior and excluded; records without explicit "
            "roots/fingerprints remain unresolved. Completeness requires independent reconciliation."
            + (" Scan stopped at a configured resource cap." if scan_truncated else "")
            + (f" {len(skipped_large_files)} oversized files were excluded from content parsing." if skipped_large_files else "")
        ),
    }
    payload["inventory_sha256"] = _sha(_canonical(payload))
    return payload


def build_sampling_frame(inventory: dict[str, Any], *, selection_seed: int = 20261015,
                         target_development: int = 8, target_validation: int = 8) -> dict[str, Any]:
    # No candidate root is accepted from a prior-artifact scan. Missing lineage
    # cannot become a fresh family by exclusion, and completeness is not self-certified.
    frame: dict[str, Any] = {
        "schema_version": 1,
        "sampler_version": SAMPLER_VERSION,
        "inventory_sha256": inventory.get("inventory_sha256"),
        "selection_seed": selection_seed,
        "algorithm": "stable-sort-by-sha256(seed || explicit-root-id); prior roots and unknown lineage excluded",
        "target": {"development": target_development, "validation": target_validation},
        "candidates": [],
        "selected": {"development": [], "validation": []},
        "excluded_counts": {
            "prior_artifact_circuit_records": inventory.get("summary", {}).get("circuit_record_count", 0),
            "unknown_or_unreconciled_lineage": inventory.get("summary", {}).get("unknown_lineage_circuit_count", 0),
            "eligible_roots": 0,
        },
        "shortfall": {"development": target_development, "validation": target_validation},
        "status": "blocked_incomplete_prior_inventory",
        "blocker": "No root is eligible until the prior inventory is independently certified complete and all discovered ancestry/fingerprints are reconciled.",
    }
    frame["frame_sha256"] = _sha(_canonical(frame))
    return frame


def verify_hash(document: dict[str, Any], field: str) -> bool:
    claimed = document.get(field)
    payload = dict(document)
    payload.pop(field, None)
    return isinstance(claimed, str) and claimed == _sha(_canonical(payload))
