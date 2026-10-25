"""Create and validate a tiny, simulator-only ML data-lineage pilot.

The pilot exercises provenance, feature/label separation, and lineage-safe
partitioning. Its synthetic scenario labels are never eligible as independent
ground truth, and the module has no model-fitting path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from experiments.full_dataset import build_family_configs
from experiments.run_experiment import _source_fingerprints, run_experiment
from ml_evaluation import _features


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ml-data-pilot-v1"
PREDICTOR_FIELDS = (
    "total_variation_distance", "js_divergence", "hellinger_distance",
    "observed_entropy_bits", "observed_max_probability",
)
FORBIDDEN_PREDICTOR_FIELDS = frozenset({
    "ground_truth_label", "scenario_label", "program_fault_id", "hardware_condition_id",
    "program_fault_description", "program_fault_operation_index", "program_fault_qubits",
    "hardware_affected_component", "diagnosis", "program_evidence", "hardware_evidence",
    "gate_mismatch_fraction", "calibration_readout_error_max", "calibration_t1_min",
    "calibration_t2_min", "configuration", "circuit", "observed_counts",
})
PARTITIONS = ("development", "calibration", "untouched_test")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for row in rows)).encode("utf-8")


def _family_partition(lineage_id: str, partitions: dict[str, str]) -> str:
    if lineage_id not in partitions:
        raise ValueError(f"no predeclared partition for lineage {lineage_id!r}")
    value = partitions[lineage_id]
    if value not in PARTITIONS:
        raise ValueError(f"invalid partition {value!r} for lineage {lineage_id!r}")
    return value


def _current_runtime() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": " ".join(platform.python_build()),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _git_provenance() -> dict[str, Any]:
    """Record the checkout identity and a digest of tracked dirty paths only."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.encode("utf-8")
    except (OSError, subprocess.SubprocessError):
        return {"head_commit": None, "tracked_worktree_status_sha256": None, "tracked_worktree_clean": None}
    return {
        "head_commit": head or None,
        "tracked_worktree_status_sha256": _sha256_bytes(status),
        "tracked_worktree_clean": not bool(status),
    }


def _pilot_source_fingerprints() -> dict[str, str]:
    """Hash generation, execution, extraction, and pilot-validation code."""
    sources = dict(_source_fingerprints())
    for relative in ("ml_evaluation.py", "experiments/ml_data_pilot.py"):
        sources[relative] = _sha256_bytes((ROOT / relative).read_bytes())
    return dict(sorted(sources.items()))


def generate_pilot(
    output_dir: Path,
    *,
    family_indices: tuple[int, ...] = (0, 1, 2),
    shots: int = 32,
) -> dict[str, Any]:
    """Write nine small local-simulator cases without overwriting any artifact."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"pilot output directory is not empty; choose a new path: {output_dir}")
    if len(family_indices) != 3 or len(set(family_indices)) != 3:
        raise ValueError("pilot requires exactly three distinct families for its three partition smoke checks")
    if any(not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 30 for index in family_indices):
        raise ValueError("family_indices must contain three integers in [0, 29]")
    if shots != 32:
        raise ValueError("pilot shot count is fixed at 32 to keep this validation bounded")

    family_to_lineage = {f"circuit_family_{index + 1:02d}": f"design_recipe_{index % 6 + 1:02d}" for index in family_indices}
    lineages = list(dict.fromkeys(family_to_lineage.values()))
    if len(lineages) != 3:
        raise ValueError("pilot families must come from three distinct recipe lineages")
    partition_map = dict(zip(lineages, PARTITIONS))
    source_hashes = _pilot_source_fingerprints()
    configs = []
    for index in family_indices:
        configs.extend(config for config in build_family_configs(index) if config.repetition_index == 0)
    if len(configs) != 9:
        raise ValueError("pilot design must contain three scenario conditions per family")

    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    config_records = []
    run_artifacts: list[dict[str, Any]] = []
    for config in configs:
        config.shots = shots
        config_records.append(config.to_dict())
        path = run_dir / f"{config.run_id}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite pilot run: {path}")
        result = run_experiment(config, output_path=path).to_dict()
        if result["status"] != "COMPLETED":
            raise RuntimeError(f"pilot simulation failed for {config.experiment_name}: {result.get('errors')}")
        metadata = result["experiment_metadata"]
        raw_features = _features(result)
        features = {key: raw_features[key] for key in PREDICTOR_FIELDS}
        observation_rows.append({
            "run_id": metadata["run_id"],
            "circuit_id": metadata["circuit_id"],
            "circuit_family_id": metadata["circuit_family_id"],
            "circuit_lineage_id": metadata["circuit_lineage_id"],
            "partition": _family_partition(metadata["circuit_lineage_id"], partition_map),
            "shots": result["execution_metadata"]["shots"],
            "random_seed": result["execution_metadata"]["random_seed"],
            "observed_counts": result["observed_counts"],
            "predictors": features,
            "source_sha256": source_hashes,
            "run_source_sha256": result["reproducibility"]["source_sha256"],
            "deterministic_run_fingerprint": result["reproducibility"]["fingerprint"],
            "runtime": {
                "python_version": result["execution_metadata"]["python_version"],
                "numpy_version": result["execution_metadata"]["numpy_version"],
            },
        })
        label_rows.append({
            "run_id": metadata["run_id"],
            "synthetic_scenario_label": metadata["ground_truth_label"],
            "label_origin": "SIMULATOR_SCENARIO_CONSTRUCTION",
            "independent_ground_truth_label": None,
            "adjudication_status": "NOT_ADJUDICATED",
            "attribution_status": "UNRESOLVED_PENDING_INDEPENDENT_ADJUDICATION",
            "fault_details": {
                key: metadata.get(key) for key in (
                    "program_fault_id", "hardware_condition_id", "program_fault_description",
                    "program_fault_operation_index", "program_fault_qubits", "hardware_affected_component",
                )
            },
        })
        run_artifacts.append({
            "file": f"runs/{path.name}",
            "sha256": _sha256_bytes(path.read_bytes()),
            "deterministic_fingerprint": result["reproducibility"]["fingerprint"],
            "run_id": metadata["run_id"],
            "random_seed": result["execution_metadata"]["random_seed"],
            "circuit_lineage_id": metadata["circuit_lineage_id"],
        })

    config_sha256 = _canonical_digest(config_records)
    artifacts = {
        "observations.jsonl": _jsonl_bytes(observation_rows),
        "synthetic_labels.jsonl": _jsonl_bytes(label_rows),
    }
    artifact_hashes = {}
    for name, content in artifacts.items():
        destination = output_dir / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite pilot artifact: {destination}")
        destination.write_bytes(content)
        artifact_hashes[name] = _sha256_bytes(content)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "bounded_local_simulator_provenance_pilot",
        "purpose": "provenance_and_leakage_control_validation_only",
        "execution_target": "LOCAL_SIMULATOR",
        "ground_truth_status": "SYNTHETIC_ONLY_NOT_INDEPENDENT",
        "independent_evaluation_ready": False,
        "model_fitting_allowed": False,
        "reason_not_fit_eligible": "No independent adjudication or independent family sampling frame exists.",
        "design": {
            "family_indices": list(family_indices),
            "families": sorted(family_to_lineage),
            "lineage_to_partition": partition_map,
            "partition_unit": "circuit_lineage_id",
            "partition_names": list(PARTITIONS),
            "shots_per_run": shots,
            "design_seed_base": 6_104_271,
            "execution_seed_base": 930_000,
            "repetitions_per_condition": 1,
            "conditions": ["NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"],
            "random_seed_by_run": {config.run_id: config.random_seed for config in configs},
            "configuration_sha256": config_sha256,
            "configuration_records": config_records,
        },
        "features": {
            "predictor_fields": list(PREDICTOR_FIELDS),
            "label_fields_are_separate_artifact": True,
            "excluded_fields": sorted(FORBIDDEN_PREDICTOR_FIELDS),
            "gate_mismatch_fraction_disposition": "EXCLUDED_TARGET_LEAKAGE_SHORTCUT",
            "calibration_feature_disposition": "EXCLUDED_NOT_OBSERVATION_ONLY",
        },
        "source_provenance": {
            "source_sha256": source_hashes,
            "generator_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "config_sha256": config_sha256,
            "git": _git_provenance(),
        },
        "runtime": _current_runtime(),
        "artifacts": {
            name: {"sha256": artifact_hashes[name], "rows": len(observation_rows)}
            for name in artifacts
        },
        "internal_run_artifacts": {
            "disposition": "DIAGNOSTIC_AND_SCENARIO_METADATA; NEVER A MODEL_INPUT",
            "records": run_artifacts,
        },
        "counts": {
            "families": len(family_to_lineage), "lineages": len(partition_map),
            "runs": len(observation_rows), "independently_adjudicated": 0,
            "unresolved_attribution": len(label_rows),
        },
        "limitations": [
            "Synthetic construction labels are not independent ground truth.",
            "Three lineages are a pipeline smoke fixture, not a statistical evaluation sample.",
            "No model fitting, calibration, acceptance scoring, or accuracy claim is permitted.",
        ],
    }
    manifest_path = output_dir / "pilot_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite pilot manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_pilot(output_dir)
    if not validation["provenance_valid"] or validation["blockers"]:
        raise ValueError(f"generated pilot did not pass its structural validation: {validation}")
    return validation


def validate_pilot(
    output_dir: Path,
    *,
    current_sources: dict[str, str] | None = None,
    current_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify byte/source/runtime/schema/partition identity and report fit blockers."""
    output_dir = Path(output_dir)
    blockers: list[str] = []
    try:
        manifest = json.loads((output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"provenance_valid": False, "fit_eligible": False, "blockers": [f"pilot manifest unavailable or invalid: {exc}"]}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        blockers.append(f"incompatible pilot schema: expected {SCHEMA_VERSION!r}")
    declared_sources = manifest.get("source_provenance", {}).get("source_sha256")
    active_sources = current_sources if current_sources is not None else _pilot_source_fingerprints()
    if not isinstance(declared_sources, dict) or declared_sources != active_sources:
        blockers.append("pilot source fingerprints are missing, stale, or incompatible with the current tree")
    if manifest.get("source_provenance", {}).get("generator_sha256") != _sha256_bytes(Path(__file__).read_bytes()):
        blockers.append("pilot generator source hash is missing or stale")
    if manifest.get("source_provenance", {}).get("git") != _git_provenance():
        blockers.append("pilot Git checkout identity or tracked worktree status is missing or stale")
    active_runtime = current_runtime if current_runtime is not None else _current_runtime()
    if manifest.get("runtime") != active_runtime:
        blockers.append("pilot runtime identity is missing, stale, or incompatible with the current runtime")

    loaded: dict[str, list[dict[str, Any]]] = {}
    artifact_meta = manifest.get("artifacts")
    for name in ("observations.jsonl", "synthetic_labels.jsonl"):
        path = output_dir / name
        try:
            content = path.read_bytes()
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            blockers.append(f"required artifact {name} is unavailable or invalid: {exc}")
            rows = []
            content = b""
        loaded[name] = rows
        expected = artifact_meta.get(name, {}) if isinstance(artifact_meta, dict) else {}
        if _sha256_bytes(content) != expected.get("sha256"):
            blockers.append(f"artifact digest mismatch for {name}")

    internal_manifest = manifest.get("internal_run_artifacts", {})
    internal_records = internal_manifest.get("records", [])
    if internal_manifest.get("disposition") != "DIAGNOSTIC_AND_SCENARIO_METADATA; NEVER A MODEL_INPUT":
        blockers.append("raw execution reports lack an explicit internal-only disposition")
    internal_by_run = {}
    for item in internal_records:
        relative = item.get("file", "")
        if not relative.startswith("runs/") or ".." in Path(relative).parts:
            blockers.append("invalid internal run artifact path")
            continue
        try:
            raw = (output_dir / relative).read_bytes()
            report = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            blockers.append(f"internal run artifact unavailable or invalid: {relative}: {exc}")
            continue
        if _sha256_bytes(raw) != item.get("sha256"):
            blockers.append(f"internal run artifact digest mismatch: {relative}")
        if report.get("reproducibility", {}).get("fingerprint") != item.get("deterministic_fingerprint"):
            blockers.append(f"internal run deterministic fingerprint mismatch: {relative}")
        metadata = report.get("experiment_metadata", {})
        if metadata.get("run_id") != item.get("run_id") or metadata.get("circuit_lineage_id") != item.get("circuit_lineage_id"):
            blockers.append(f"internal run identity/lineage mismatch: {relative}")
        internal_by_run[item.get("run_id")] = item

    observations, labels = loaded["observations.jsonl"], loaded["synthetic_labels.jsonl"]
    if len(internal_records) != len(observations):
        blockers.append("internal run artifact inventory does not cover every observation")
    observation_ids = [row.get("run_id") for row in observations]
    label_ids = [row.get("run_id") for row in labels]
    if not observations or len(set(observation_ids)) != len(observation_ids):
        blockers.append("observation rows are missing or have duplicate run IDs")
    if len(observation_ids) != len(label_ids) or set(observation_ids) != set(label_ids) or len(set(label_ids)) != len(label_ids):
        blockers.append("synthetic labels are missing, duplicated, or do not align one-to-one with observations")
    for row in observations:
        if set(row.get("predictors", {})) != set(PREDICTOR_FIELDS):
            blockers.append(f"incompatible predictor schema for run {row.get('run_id')}")
        if FORBIDDEN_PREDICTOR_FIELDS.intersection(row.get("predictors", {})):
            blockers.append(f"leakage field in predictors for run {row.get('run_id')}")
        if row.get("source_sha256") != declared_sources:
            blockers.append(f"per-run source provenance mismatch for run {row.get('run_id')}")
        report_sources = row.get("run_source_sha256")
        expected_report_sources = {
            key: value for key, value in declared_sources.items() if key in _source_fingerprints()
        } if isinstance(declared_sources, dict) else {}
        if report_sources != expected_report_sources:
            blockers.append(f"execution source provenance mismatch for run {row.get('run_id')}")
        if row.get("runtime") != {k: active_runtime.get(k) for k in ("python_version", "numpy_version")}:
            blockers.append(f"per-run runtime mismatch for run {row.get('run_id')}")
        internal = internal_by_run.get(row.get("run_id"))
        if not internal or internal.get("deterministic_fingerprint") != row.get("deterministic_run_fingerprint"):
            blockers.append(f"observation row lacks its matching reproducible internal run for {row.get('run_id')}")
    for row in labels:
        if row.get("synthetic_scenario_label") not in {"NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY", "AMBIGUOUS", "UNLABELED", None}:
            blockers.append(f"unsupported or missing synthetic label for run {row.get('run_id')}")
        if row.get("attribution_status") not in {"UNRESOLVED_PENDING_INDEPENDENT_ADJUDICATION", "OBSERVATIONALLY_EQUIVALENT", "AMBIGUOUS", "RESOLVED_BY_INDEPENDENT_ADJUDICATION"}:
            blockers.append(f"missing or incompatible attribution disposition for run {row.get('run_id')}")
        if row.get("independent_ground_truth_label") is not None and row.get("adjudication_status") != "INDEPENDENTLY_ADJUDICATED":
            blockers.append("independent label is present without independent adjudication status")

    design = manifest.get("design", {})
    if _canonical_digest(design.get("configuration_records", [])) != design.get("configuration_sha256"):
        blockers.append("pilot configuration records do not match their declared digest")
    if manifest.get("source_provenance", {}).get("config_sha256") != design.get("configuration_sha256"):
        blockers.append("configuration provenance digest is inconsistent")
    partition_map = design.get("lineage_to_partition", {})
    if not partition_map:
        blockers.append("lineage partition map is missing")
    observed_partition: dict[str, set[str]] = {}
    family_lineage: dict[str, str] = {}
    for row in observations:
        lineage = row.get("circuit_lineage_id")
        family = row.get("circuit_family_id")
        try:
            expected_partition = _family_partition(lineage, partition_map)
        except ValueError as exc:
            blockers.append(str(exc))
            continue
        observed_partition.setdefault(lineage, set()).add(row.get("partition"))
        if row.get("partition") != expected_partition:
            blockers.append(f"partition differs from predeclared lineage assignment for {lineage}")
        if family in family_lineage and family_lineage[family] != lineage:
            blockers.append(f"family {family} has conflicting lineage identifiers")
        family_lineage[family] = lineage
        counts = row.get("observed_counts")
        if not isinstance(counts, dict) or any(not isinstance(value, int) or value < 0 for value in counts.values()):
            blockers.append(f"invalid or missing observed counts for run {row.get('run_id')}")
    if any(len(partitions) != 1 for partitions in observed_partition.values()):
        blockers.append("a circuit lineage crosses partitions")

    # Screen identical and near-identical reference circuit families; any pair
    # crossing partitions is merged as one dependency for this pilot protocol.
    family_circuits: dict[str, str] = {}
    for family, lineage in family_lineage.items():
        matching = [row for row in observations if row.get("circuit_family_id") == family]
        config = next((item for item in design.get("configuration_records", []) if item.get("circuit_family_id") == family), None)
        if config:
            family_circuits[family] = json.dumps(config.get("circuit"), sort_keys=True, separators=(",", ":"))
    families = sorted(family_circuits)
    near_duplicate_pairs = []
    for left_index, left in enumerate(families):
        for right in families[left_index + 1:]:
            similarity = SequenceMatcher(None, family_circuits[left], family_circuits[right]).ratio()
            if similarity >= 0.8:
                near_duplicate_pairs.append({"left": left, "right": right, "similarity": similarity})
                left_partition = partition_map.get(family_lineage[left])
                right_partition = partition_map.get(family_lineage[right])
                if left_partition != right_partition:
                    blockers.append(f"near-duplicate families cross partitions: {left}, {right}")

    provenance_errors = [item for item in blockers if any(token in item for token in ("source", "runtime", "digest", "schema", "artifact", "provenance", "configuration", "Git"))]
    fit_blockers = list(blockers)
    if any(row.get("independent_ground_truth_label") is None or row.get("adjudication_status") != "INDEPENDENTLY_ADJUDICATED" for row in labels):
        fit_blockers.append("independent ground truth or adjudication evidence is unavailable")
    if manifest.get("ground_truth_status") != "INDEPENDENTLY_ADJUDICATED":
        fit_blockers.append("pilot labels are synthetic and cannot support model fitting or independent evaluation")
    fit_eligible = not fit_blockers and manifest.get("model_fitting_allowed") is True
    return {
        "provenance_valid": not provenance_errors,
        "structurally_valid": not blockers,
        "fit_eligible": fit_eligible,
        "blockers": sorted(set(blockers)),
        "fit_blockers": sorted(set(fit_blockers)),
        "counts": {"observations": len(observations), "labels": len(labels), "families": len(family_lineage), "lineages": len(observed_partition), "near_duplicate_family_pairs": len(near_duplicate_pairs)},
        "near_duplicate_pairs": near_duplicate_pairs,
        "checked_source_sha256": active_sources,
        "checked_runtime": active_runtime,
    }


def require_fit_eligible(output_dir: Path) -> None:
    """Fail before model fitting if provenance, schema, or independent truth is absent."""
    result = validate_pilot(output_dir)
    if not result["fit_eligible"]:
        raise ValueError("pilot is not eligible for model fitting: " + "; ".join(result["fit_blockers"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = generate_pilot(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
