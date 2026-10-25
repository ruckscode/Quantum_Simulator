"""Bounded record-level audit of recoverable metadata in historical ML reports.

This tool reports provenance found in saved artifacts; it never reconstructs
the generator version from current source and never makes rows fit-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiments" / "full_dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "experiments" / "ml_historical_record_recovery_20261015_v2.json"
MAX_REPORTS = 1000
MAX_SINGLE_REPORT_BYTES = 1024 * 1024
MAX_TOTAL_REPORT_BYTES = 64 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_dict(*values: Any) -> dict[str, Any]:
    return next((value for value in values if isinstance(value, dict)), {})


def build_record_recovery_table(
    report_dir: Path = DEFAULT_REPORT_DIR,
    *,
    max_reports: int = MAX_REPORTS,
    max_total_bytes: int = MAX_TOTAL_REPORT_BYTES,
) -> dict[str, Any]:
    """Recover only directly recorded row fields under explicit read caps."""
    report_dir = Path(report_dir)
    manifest_path = report_dir / "dataset_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest_size = manifest_path.stat().st_size
        if manifest_size > MAX_SINGLE_REPORT_BYTES:
            raise ValueError("dataset manifest exceeds the bounded read size")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("dataset_manifest.json is missing")
    files = sorted(report_dir.glob("run-*.json"), key=lambda path: path.name)
    if len(files) > max_reports:
        raise ValueError(f"report count {len(files)} exceeds cap {max_reports}")
    total_bytes = 0
    records = []
    seen_run_ids: set[str] = set()
    family_index = {item.get("circuit_family_id"): item for item in manifest.get("families", []) if isinstance(item, dict)}
    for path in files:
        size = path.stat().st_size
        if size > MAX_SINGLE_REPORT_BYTES:
            raise ValueError(f"report exceeds bounded read size: {path.name}")
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ValueError(f"report bytes {total_bytes} exceed cap {max_total_bytes}")
        report = json.loads(path.read_text(encoding="utf-8"))
        config = _first_dict(report.get("configuration"))
        execution = _first_dict(report.get("execution_metadata"))
        metadata = _first_dict(report.get("experiment_metadata"))
        reproducibility = _first_dict(report.get("reproducibility"))
        run_id = config.get("run_id") or metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            run_id = path.stem
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate stable run/artifact ID: {run_id}")
        seen_run_ids.add(run_id)
        family_id = config.get("circuit_family_id") or metadata.get("circuit_family_id")
        family_manifest = family_index.get(family_id, {})
        label = config.get("ground_truth_label") or metadata.get("ground_truth_label")
        label_source = (
            "controlled simulator scenario-construction metadata; synthetic, not independently adjudicated"
            if execution.get("execution_kind") == "SYNTHETIC_VALIDATION"
            else "unknown or non-synthetic origin; requires separate review"
        )
        source_hashes = _first_dict(
            report.get("source_sha256"), execution.get("source_sha256"), reproducibility.get("source_sha256")
        )
        observed_fields = {
            "stable_artifact_id": run_id,
            "run_id": run_id,
            "circuit_id": config.get("circuit_id") or metadata.get("circuit_id"),
            "circuit_family_id": family_id,
            "experimental_group_id": config.get("experimental_group_id") or metadata.get("experimental_group_id"),
            "repetition_index": config.get("repetition_index") or metadata.get("repetition_index"),
            "random_seed": execution.get("random_seed", config.get("random_seed")),
            "shots": execution.get("shots", config.get("shots")),
            "python_version": execution.get("python_version"),
            "numpy_version": execution.get("numpy_version"),
            "simulator_version_string": reproducibility.get("simulator_version"),
            "label": label,
            "label_origin": label_source,
            "circuit_configuration": config.get("circuit"),
            "submitted_circuit_configuration": config.get("observed_circuit"),
            "hardware_configuration": config.get("hardware_parameters"),
            "noise_configuration": config.get("noise_parameters"),
            "saved_reproducibility_fingerprint": reproducibility.get("fingerprint"),
        }
        missing = {
            "per_run_source_sha256": not bool(source_hashes),
            "exact_generation_provenance": not isinstance(manifest.get("generation_provenance"), dict),
            "circuit_lineage_or_recipe_root": not isinstance(config.get("circuit_lineage_id") or metadata.get("circuit_lineage_id"), str),
            "independent_label_adjudication": True,
            "real_hardware_observation": execution.get("execution_target") != "REAL_HARDWARE",
        }
        field_status = {
            "stable_artifact_id": "RECORDED_IN_SAVED_REPORT",
            "run_id": "RECORDED_IN_SAVED_REPORT",
            "circuit_id": "RECORDED_IN_SAVED_REPORT",
            "circuit_family_id": "RECORDED_IN_SAVED_REPORT; DOES_NOT_IDENTIFY_RECIPE_ROOT",
            "experimental_group_id": "RECORDED_IN_SAVED_REPORT",
            "repetition_index": "RECORDED_IN_SAVED_REPORT",
            "random_seed": "RECORDED_IN_SAVED_REPORT",
            "shots": "RECORDED_IN_SAVED_REPORT",
            "python_version": "RECORDED_IN_SAVED_REPORT; SOURCE_BUILD_NOT_IDENTIFIED",
            "numpy_version": "RECORDED_IN_SAVED_REPORT",
            "simulator_version_string": "RECORDED_BUT_NONUNIQUE_AND_NOT_A_SOURCE_REVISION",
            "circuit_configuration": "RECORDED_IN_SAVED_REPORT",
            "submitted_circuit_configuration": "RECORDED_IN_SAVED_REPORT",
            "hardware_configuration": "RECORDED_IN_SAVED_REPORT; SIMULATOR_CONFIGURATION_ONLY",
            "noise_configuration": "RECORDED_IN_SAVED_REPORT",
            "saved_reproducibility_fingerprint": "RECORDED_IN_SAVED_REPORT; NOT_A_GENERATOR_SOURCE_MAP",
            "label": "RECORDED_SYNTHETIC_SCENARIO_LABEL; NOT_INDEPENDENT_TRUTH",
            "label_origin": "INFERRED_FROM_RECORDED_EXECUTION_AND_VALIDATION_METADATA",
            "report_sha256_current_artifact_integrity_only": "RECOMPUTED_FROM_CURRENT_SAVED_BYTES",
            "configuration_sha256_recomputed_from_saved_json": "RECOMPUTED_FROM_CURRENT_SAVED_JSON; NOT_HISTORICAL_SOURCE_PROVENANCE",
            "circuit_pair_sha256_recomputed_from_saved_json": "RECOMPUTED_FROM_CURRENT_SAVED_JSON; NOT_HISTORICAL_SOURCE_PROVENANCE",
            "per_run_source_sha256": "MISSING" if not source_hashes else "RECORDED_IN_SAVED_REPORT",
            "generation_source_hash_map": "MISSING" if not isinstance(manifest.get("generation_provenance"), dict) else "RECORDED_IN_MANIFEST",
            "recipe_root_or_ancestry": "MISSING" if missing["circuit_lineage_or_recipe_root"] else "RECORDED_IN_SAVED_REPORT",
            "independent_label_adjudication": "MISSING",
            "real_hardware_observation": "ABSENT" if missing["real_hardware_observation"] else "RECORDED",
        }
        records.append({
            **observed_fields,
            "available_field_status": field_status,
            "report_path": path.as_posix(),
            "report_sha256_current_artifact_integrity_only": _sha256_file(path),
            "configuration_sha256_recomputed_from_saved_json": _canonical_sha256(config),
            "circuit_pair_sha256_recomputed_from_saved_json": _canonical_sha256({
                "reference": config.get("circuit"), "submitted": config.get("observed_circuit")
            }),
            "family_manifest_entry_present": bool(family_manifest),
            "family_recipe_root": None,
            "source_sha256_recorded": source_hashes or None,
            "missing_or_unproven": missing,
            "eligibility": {
                "primary_training": "INELIGIBLE",
                "development_or_model_selection": "INELIGIBLE",
                "calibration": "INELIGIBLE",
                "held_out_evaluation": "INELIGIBLE",
                "software_engineering_fixture": "USABLE_WITH_SYNTHETIC_LABEL_AND_PROVENANCE_CAVEAT",
            },
            "eligibility_reason": "Missing generator source identity and recipe ancestry; synthetic scenario labels are not independent truth. Recomputed artifact/configuration hashes establish only current saved-byte integrity.",
            "evidence": [
                path.as_posix(), manifest_path.as_posix(),
                "reports/experiments/ml_data_foundation_preflight_20261015.json",
                "reports/experiments/ml_source_compatibility_audit_20261015.json",
            ],
        })
    if manifest.get("records_written") is not None and int(manifest["records_written"]) != len(records):
        raise ValueError("saved dataset manifest record count disagrees with bounded report scan")
    summary = {
        "record_count": len(records),
        "unique_run_ids": len(seen_run_ids),
        "unique_circuit_families": len({row["circuit_family_id"] for row in records if row["circuit_family_id"]}),
        "label_counts": dict(Counter(str(row["label"]) for row in records)),
        "missing_run_source_hash_rows": sum(not row["source_sha256_recorded"] for row in records),
        "missing_lineage_rows": sum(row["missing_or_unproven"]["circuit_lineage_or_recipe_root"] for row in records),
        "independent_adjudication_rows": 0,
        "real_hardware_rows": sum(row["missing_or_unproven"]["real_hardware_observation"] is False for row in records),
        "total_report_bytes_read": total_bytes,
        "caps": {"max_reports": max_reports, "max_single_report_bytes": MAX_SINGLE_REPORT_BYTES, "max_total_report_bytes": max_total_bytes},
        "dataset_manifest_has_generation_provenance": isinstance(manifest.get("generation_provenance"), dict),
        "eligibility": "No historical record is eligible for ML training, development/model selection, calibration, or held-out generalization claims.",
    }
    return {
        "schema_version": 2,
        "audit": "historical_ml_record_provenance_recovery",
        "audit_date": "2026-10-15",
        "method": "Bounded direct-field inspection; no provenance inferred from current source; current hashes are for saved-artifact integrity only.",
        "field_status_definitions": {
            "RECORDED_IN_SAVED_REPORT": "The value is present in the identified JSON report; this verifies recorded content, not that the record is accurate or identifies its generating source code.",
            "INFERRED_FROM_RECORDED_EXECUTION_AND_VALIDATION_METADATA": "The synthetic label-origin disposition follows saved validation/execution metadata and is not independent adjudication.",
            "RECOMPUTED_FROM_CURRENT_SAVED_BYTES": "A current integrity digest; not a source/generator digest captured at the time of execution.",
            "MISSING": "No value is present in the inspected row/manifest and none is reconstructed from present-day code.",
        },
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "summary": summary,
        "records": records,
    }


def write_report(report: dict[str, Any], destination: Path = DEFAULT_OUTPUT) -> None:
    """Write one new evidence artifact without replacing an existing file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_record_recovery_table(args.report_dir)
    write_report(report, args.output)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
