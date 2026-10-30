"""Audit the full execution-derived dataset. No model fitting occurs here."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "experiments" / "full_dataset"
DEFAULT_EVALUATION_OUTPUT_DIR = ROOT / "ml_results" / "grouped_cv_audit_v2"
DATASET_FINGERPRINT_SCHEME = "source_json_bytes_sha256_v1"
DATASET_FINGERPRINT_DEFINITION = (
    "SHA-256 over sorted POSIX-relative paths and raw bytes of every .json file "
    "under the selected source directory; the dataset_manifest.json is included; "
    "configured generated-output subdirectories are excluded."
)
LABELS = ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
REQUIRED_METADATA = (
    "ground_truth_label", "circuit_id", "circuit_family_id", "circuit_lineage_id", "program_fault_id",
    "hardware_condition_id", "experimental_group_id", "run_id", "repetition_index",
    "program_fault_description", "program_fault_operation_index", "program_fault_qubits",
    "hardware_affected_component",
)
EXPLICITLY_EXCLUDED = (
    "ground_truth_label", "experiment_name", "source", "fault_type", "circuit_family_id", "circuit_lineage_id",
    "circuit_id", "program_fault_id", "hardware_condition_id", "experimental_group_id",
    "run_id", "repetition_index", "program_fault_description", "program_fault_operation_index",
    "program_fault_qubits", "hardware_affected_component", "diagnosis result/category/evidence",
    "any feature generated directly from the label",
)
OBSERVABLE_ONLY = (
    "total_variation_distance", "js_divergence", "hellinger_distance",
    "observed_entropy_bits", "observed_max_probability", "calibration_readout_error_max",
    "calibration_t1_min", "calibration_t2_min",
)
WITH_PROGRAM_COMPARISON = OBSERVABLE_ONLY + ("gate_mismatch_fraction",)


def dataset_fingerprint(
    report_dir: Path = REPORT_DIR,
    *,
    excluded_dirs: tuple[Path, ...] | list[Path] | None = None,
) -> str:
    """Fingerprint source JSON files, independent of traversal order and outputs.

    Each input contributes its POSIX-relative path and exact file bytes. The
    default evaluation output tree is excluded, and callers may exclude any
    additional generated-output directories.
    """
    source_root = Path(report_dir).resolve()
    excluded = [Path(path).resolve() for path in (excluded_dirs or ())]
    excluded.append(DEFAULT_EVALUATION_OUTPUT_DIR.resolve())
    files = []
    for path in source_root.rglob("*.json"):
        resolved = path.resolve()
        if any(resolved == directory or directory in resolved.parents for directory in excluded):
            continue
        files.append((path.relative_to(source_root).as_posix(), path))
    digest = hashlib.sha256()
    for relative_name, path in sorted(files, key=lambda item: item[0]):
        name_bytes = relative_name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _fingerprint(report: dict[str, Any]) -> str:
    reproducibility = report.get("reproducibility")
    value = reproducibility.get("fingerprint") if isinstance(reproducibility, Mapping) else None
    if value:
        return str(value)
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()


def _operation_signatures(circuit: list[Any]) -> list[str]:
    signatures = []
    for operation in circuit:
        if isinstance(operation, dict):
            normalized = {
                "name": str(operation.get("name", operation.get("op", operation.get("gate", "")))).lower(),
                "qubits": operation.get("qubits", []),
                "parameters": operation.get("parameters", []),
            }
        elif isinstance(operation, (tuple, list)) and operation:
            normalized = {"name": str(operation[0]).lower(), "operands": operation[1:]}
        else:
            continue
        signatures.append(json.dumps(normalized, sort_keys=True, separators=(",", ":")))
    return signatures


def _features(report: dict[str, Any]) -> dict[str, float]:
    configuration = report.get("configuration", {})
    anomaly = report.get("anomaly") or {}
    counts = report.get("observed_counts") or {}
    total = sum(counts.values())
    probabilities = [value / total for value in counts.values()] if total else []
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    hardware = configuration.get("hardware_parameters") or {}
    qubits = list((hardware.get("qubits") or {}).values())
    readout = [float(q["readout_error"]) for q in qubits if q.get("readout_error") is not None]
    t1 = [float(q["t1"]) for q in qubits if q.get("t1") is not None]
    t2 = [float(q["t2"]) for q in qubits if q.get("t2") is not None]
    reference = _operation_signatures(report.get("circuit", {}).get("ideal", []))
    observed = _operation_signatures(report.get("circuit", {}).get("observed", []))
    width = max(len(reference), len(observed), 1)
    mismatch_count = sum(left != right for left, right in itertools.zip_longest(reference, observed))
    return {
        "total_variation_distance": float(anomaly.get("total_variation_distance", np.nan)),
        "js_divergence": float(anomaly.get("js_divergence", np.nan)),
        "hellinger_distance": float(anomaly.get("hellinger_distance", np.nan)),
        "observed_entropy_bits": entropy if total else np.nan,
        "observed_max_probability": max(probabilities, default=np.nan),
        "calibration_readout_error_max": max(readout, default=np.nan),
        "calibration_t1_min": min(t1, default=np.nan),
        "calibration_t2_min": min(t2, default=np.nan),
        "gate_mismatch_fraction": mismatch_count / width,
    }


def load_dataset(report_dir: Path = REPORT_DIR) -> list[dict[str, Any]]:
    """Load reports with explicit ground truth from the selected dataset directory."""
    report_root = Path(report_dir).resolve()
    records = []
    seen = set()
    for path in sorted(report_root.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # The disposition audit records malformed/unreadable source files;
            # they cannot become feature rows.
            continue
        if not isinstance(report, Mapping):
            continue
        metadata = report.get("experiment_metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        if report.get("status") != "COMPLETED" or metadata.get("ground_truth_label") not in LABELS:
            continue
        reproducibility = report.get("reproducibility")
        source_sha256 = reproducibility.get("source_sha256") if isinstance(reproducibility, Mapping) else None
        fingerprint = _fingerprint(report)
        duplicate_fingerprint = fingerprint in seen
        seen.add(fingerprint)
        try:
            report_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            report_path = path.as_posix()
        records.append({
            **{field: metadata.get(field) for field in REQUIRED_METADATA},
            "experiment_name": metadata.get("experiment_name"),
            "validation_label": metadata.get("validation_label"),
            "execution_kind": report.get("execution_metadata", {}).get("execution_kind"),
            "observation_source": report.get("execution_metadata", {}).get("observation_source"),
            "hardware_model_applied": report.get("execution_metadata", {}).get("hardware_model_applied"),
            "report_fingerprint": fingerprint,
            "_source_sha256": source_sha256,
            "duplicate_fingerprint": duplicate_fingerprint,
            "report_path": report_path,
            "_metadata_fields_present": sorted(metadata),
            **_features(report),
        })
    return records


def load_report_dispositions(report_dir: Path = REPORT_DIR) -> list[dict[str, Any]]:
    """Return auditable status rows for source reports, including excluded cases.

    This is intentionally separate from :func:`load_dataset`, whose model input
    rows remain limited to completed reports with explicit supported labels.
    """
    dispositions = []
    for path in sorted(report_dir.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report = None
            parse_error = f"{type(exc).__name__}: {exc}"
        else:
            parse_error = None
        try:
            report_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            report_path = path.resolve().as_posix()
        if not isinstance(report, Mapping):
            dispositions.append({
                "run_id": path.stem,
                "circuit_family_id": None,
                "ground_truth_label": None,
                "status": "PARSING_FAILURE",
                "disposition": "PARSING_FAILURE",
                "evaluation_eligibility": "INELIGIBLE",
                "ineligibility_reason": parse_error or "report must be a JSON object",
                "errors": [parse_error or "report must be a JSON object"],
                "report_path": report_path,
            })
            continue
        status = report.get("status")
        if status is None:
            continue
        metadata = report.get("experiment_metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        errors = report.get("errors", [])
        error_text = " ".join(str(value) for value in errors).lower()
        unsupported = status == "UNSUPPORTED" or "unsupported operation" in error_text
        disposition = "COMPLETED" if status == "COMPLETED" else (
            "UNSUPPORTED" if unsupported else "FAILED" if status == "FAILED" else "OTHER"
        )
        label = metadata.get("ground_truth_label")
        eligible = disposition == "COMPLETED" and label in LABELS
        dispositions.append({
            "run_id": metadata.get("run_id") or report.get("run_id") or path.stem,
            "circuit_family_id": metadata.get("circuit_family_id"),
            "circuit_lineage_id": metadata.get("circuit_lineage_id"),
            "ground_truth_label": label,
            "status": status,
            "disposition": disposition,
            "evaluation_eligibility": "ELIGIBLE" if eligible else "INELIGIBLE",
            "ineligibility_reason": None if eligible else (
                "execution did not complete" if disposition != "COMPLETED"
                else "missing or unsupported ground-truth label"
            ),
            "errors": errors,
            "report_path": report_path,
        })
    return dispositions


def _split_feasibility(records: list[dict[str, Any]]) -> tuple[bool, str]:
    missing_lineage = sum(not row.get("circuit_lineage_id") for row in records)
    if missing_lineage:
        return False, f"safe split blocked: {missing_lineage} rows lack circuit_lineage_id; family-only grouping may leak related design recipes"
    families = sorted({row["circuit_lineage_id"] for row in records if row["circuit_lineage_id"]})
    if len(families) < 2:
        return False, "at least two circuit-lineage groups are required"
    labels = np.asarray([row["ground_truth_label"] for row in records])
    groups = np.asarray([row["circuit_lineage_id"] for row in records])
    try:
        from sklearn.model_selection import GroupShuffleSplit

        for seed in range(200):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train, test = next(splitter.split(np.zeros((len(records), 1)), labels, groups))
            if set(labels[train]) == set(LABELS) and set(labels[test]) == set(LABELS):
                held_out = sorted(str(value) for value in set(groups[test]))
                return True, f"GroupShuffleSplit feasible; held-out circuit lineages: {held_out}"
        return False, "GroupShuffleSplit found no split containing all classes on both sides"
    except Exception as exc:
        for count in range(1, len(families)):
            for held_out in itertools.combinations(families, count):
                test_rows = [row for row in records if row["circuit_lineage_id"] in held_out]
                train_rows = [row for row in records if row["circuit_lineage_id"] not in held_out]
                if {row["ground_truth_label"] for row in train_rows} == set(LABELS) and {row["ground_truth_label"] for row in test_rows} == set(LABELS):
                    return True, f"exact lineage-subset check feasible; sklearn unavailable ({type(exc).__name__})"
        return False, f"no family-held-out split contains all classes; sklearn unavailable ({type(exc).__name__})"


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _print_stats(records: list[dict[str, Any]], features: tuple[str, ...], title: str) -> None:
    print(title)
    print("feature | missing | dtype | count | mean | std | min | median | max")
    for feature in features:
        values = [float(row[feature]) for row in records if not _missing(row[feature])]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        if values:
            print(f"{feature} | {len(records) - len(values)} | float | {len(values)} | {np.mean(values):.6g} | {std:.6g} | {min(values):.6g} | {np.median(values):.6g} | {max(values):.6g}")
        else:
            print(f"{feature} | {len(records)} | unknown | 0 | n/a | n/a | n/a | n/a | n/a")


def _feature_findings(records: list[dict[str, Any]], features: tuple[str, ...], set_name: str) -> None:
    print(f"\n{set_name}: feature distribution by ground-truth class")
    all_by_feature: dict[str, list[float]] = {}
    associations = []
    constants = []
    for feature in features:
        by_class = {
            label: [float(row[feature]) for row in records if row["ground_truth_label"] == label and not _missing(row[feature])]
            for label in LABELS
        }
        values = [value for class_values in by_class.values() for value in class_values]
        all_by_feature[feature] = values
        overall = float(np.mean(values)) if values else 0.0
        between = sum(len(class_values) * (float(np.mean(class_values)) - overall) ** 2 for class_values in by_class.values() if class_values)
        total = sum((value - overall) ** 2 for value in values)
        eta_squared = between / total if total > 0 else 0.0
        associations.append((eta_squared, feature))
        if values and max(values) - min(values) <= 1e-12:
            constants.append(feature)
        class_summary = []
        for label in LABELS:
            class_values = by_class[label]
            if class_values:
                sd = float(np.std(class_values, ddof=1)) if len(class_values) > 1 else float("nan")
                class_summary.append(f"{label}:n={len(class_values)},mean={np.mean(class_values):.4g},sd={sd:.4g}")
            else:
                class_summary.append(f"{label}:n=0")
        print(f"{feature}: " + " | ".join(class_summary) + f" | eta_squared={eta_squared:.4f}")

    high = [(eta, name) for eta, name in associations if eta >= 0.80]
    print("high class association (descriptive eta-squared >= 0.80): " + (", ".join(f"{name}={eta:.3f}" for eta, name in sorted(high, reverse=True)) or "none"))
    print("constant features: " + (", ".join(constants) if constants else "none"))
    exact_pairs, near_pairs = [], []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1:]:
            xs, ys = [], []
            for row in records:
                if not _missing(row[left]) and not _missing(row[right]):
                    xs.append(float(row[left]))
                    ys.append(float(row[right]))
            if len(xs) < 2:
                continue
            if np.allclose(xs, ys, rtol=0, atol=1e-12):
                exact_pairs.append((left, right))
            if np.std(xs) > 0 and np.std(ys) > 0:
                correlation = float(np.corrcoef(xs, ys)[0, 1])
                if abs(correlation) >= 0.995:
                    near_pairs.append((left, right, round(correlation, 6)))
    print("exact duplicate feature pairs: " + (str(exact_pairs) if exact_pairs else "none"))
    print("near-duplicate pairs (|Pearson r| >= 0.995): " + (str(near_pairs) if near_pairs else "none"))


def main() -> int:
    records = load_dataset()
    if not records:
        print(f"No completed reports with explicit ground_truth_label found under {REPORT_DIR}")
        print("ML training is disabled; this script performs data and split audits only.")
        return 0

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["experimental_group_id"]:
            groups[record["experimental_group_id"]].append(record)
    runs = [record["run_id"] for record in records]
    class_records = Counter(row["ground_truth_label"] for row in records)
    class_groups = {label: len({row["experimental_group_id"] for row in records if row["ground_truth_label"] == label}) for label in LABELS}
    rep_counts = Counter(len(rows) for rows in groups.values())
    repeated_groups = sum(len(rows) > 1 for rows in groups.values())
    duplicate_run_ids = len(runs) - len(set(runs))
    rep_assignments = Counter((row["experimental_group_id"], row["repetition_index"]) for row in records)
    duplicate_group_rep_count = sum(count - 1 for count in rep_assignments.values() if count > 1)

    print("FULL DATASET AUDIT (NO MODEL TRAINING)")
    print(f"total records: {len(records)}")
    print(f"independent circuit families: {len({row['circuit_family_id'] for row in records if row['circuit_family_id']})}")
    print(f"unique circuit identities: {len({row['circuit_id'] for row in records if row['circuit_id']})}")
    print(f"unique experimental groups: {len(groups)}")
    print(f"unique run IDs: {len(set(runs))}; duplicate run IDs: {duplicate_run_ids}")
    print(f"experimental groups with repeated records: {repeated_groups}")
    print(f"repetitions per group distribution: {dict(sorted(rep_counts.items()))}")
    print(f"duplicate (group, repetition_index) assignments: {duplicate_group_rep_count}")
    print("records / independent experimental groups per class:")
    for label in LABELS:
        print(f"{label}: {class_records[label]} / {class_groups[label]}")

    print("missing metadata (applicability-aware):")
    applicable_fields = {
        "program_fault_id", "program_fault_description", "program_fault_operation_index", "program_fault_qubits",
    }
    for field in REQUIRED_METADATA:
        applicable = [row for row in records if (
            field not in applicable_fields | {"hardware_condition_id", "hardware_affected_component"}
            or (field in applicable_fields and row["ground_truth_label"] == "PROGRAM_FAULT")
            or (field in {"hardware_condition_id", "hardware_affected_component"} and row["ground_truth_label"] == "HARDWARE_ANOMALY")
        )]
        missing_count = sum(row.get(field) is None for row in applicable)
        print(f"{field}: {missing_count} missing; {len(records) - len(applicable)} null by design")
    print(f"duplicate fingerprints: {sum(row['duplicate_fingerprint'] for row in records)}")
    anomaly_rows = [row for row in records if row["ground_truth_label"] == "HARDWARE_ANOMALY"]
    print(f"unique hardware condition IDs: {len({row['hardware_condition_id'] for row in anomaly_rows if row['hardware_condition_id']})}")
    hardware_applied_ok = all(row["hardware_model_applied"] is True for row in records)
    print(f"configured hardware model reached execution: {'PASS' if hardware_applied_ok else 'FAIL'} ({sum(row['hardware_model_applied'] is True for row in records)}/{len(records)} runs)")

    family_classes = {
        family: {row["ground_truth_label"] for row in records if row["circuit_family_id"] == family}
        for family in sorted({row["circuit_family_id"] for row in records if row["circuit_family_id"]})
    }
    all_families_complete = all(classes == set(LABELS) for classes in family_classes.values())
    complete_family_count = sum(classes == set(LABELS) for classes in family_classes.values())
    print(f"every family contains all three classes: {'PASS' if all_families_complete else 'FAIL'} ({complete_family_count}/{len(family_classes)})")

    consistency_ok = all(len({row["ground_truth_label"] for row in rows}) == 1 for rows in groups.values())
    condition_metadata_ok = all(
        (row["ground_truth_label"] == "NORMAL" and row["program_fault_id"] is None and row["hardware_condition_id"] is None)
        or (row["ground_truth_label"] == "PROGRAM_FAULT" and bool(row["program_fault_id"]) and bool(row["program_fault_description"]) and row["hardware_condition_id"] is None)
        or (row["ground_truth_label"] == "HARDWARE_ANOMALY" and row["program_fault_id"] is None and bool(row["hardware_condition_id"]) and bool(row["hardware_affected_component"]))
        for row in records
    )
    print(f"ground-truth consistency per group: {'PASS' if consistency_ok else 'FAIL'}")
    print(f"condition ID/description applicability: {'PASS' if condition_metadata_ok else 'FAIL'}")

    print("\nLEAKAGE AUDIT")
    print("excluded from features: " + ", ".join(EXPLICITLY_EXCLUDED))
    print("labels are read only from explicit configured ground_truth_label; no output threshold or diagnosis result is used")
    print("OBSERVABLE_ONLY uses execution distributions and configured calibration observations; calibration profiles are condition-associated by design")
    print("WITH_PROGRAM_COMPARISON adds circuit-operation comparison; gate_mismatch_fraction directly reveals an injected program change and must be treated as an explicit ablation/confound")
    print("no metadata ID or label-derived field is included in either feature set")

    feature_missing = {}
    for features, title in ((OBSERVABLE_ONLY, "OBSERVABLE_ONLY"), (WITH_PROGRAM_COMPARISON, "WITH_PROGRAM_COMPARISON")):
        _print_stats(records, features, f"\n{title}: overall feature quality/statistics")
        _feature_findings(records, features, title)
        feature_missing[title] = {feature: sum(_missing(row[feature]) for row in records) for feature in features}
    print("\nfeature missing counts:")
    for set_name, missingness in feature_missing.items():
        print(f"{set_name}: {missingness}")

    feasible, explanation = _split_feasibility(records)
    print("\nGROUP-AWARE EVALUATION READINESS")
    print("holdout unit: circuit_lineage_id (all related recipe siblings, conditions, and repetitions remain together)")
    print(f"independent experimental groups per class: {class_groups}")
    print(f"GroupShuffleSplit all-class train/test feasibility: {'YES' if feasible else 'NO'}")
    print(explanation)
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
        labels = np.asarray([row["ground_truth_label"] for row in records])
        family_ids = np.asarray([row["circuit_lineage_id"] for row in records])
        valid_folds = sum(
            set(labels[train]) == set(LABELS) and set(labels[test]) == set(LABELS)
            for train, test in folds.split(np.zeros((len(records), 1)), labels, family_ids)
        )
        print(f"StratifiedGroupKFold available: YES; 5 folds with all classes on both sides: {valid_folds}/5")
    except Exception as exc:
        print(f"StratifiedGroupKFold available: NO ({type(exc).__name__}: {exc})")
    print("No model was trained; final predictive performance is not reported.")
    audit_pass = len(records) == 270 and len(family_classes) == 30 and len({row["circuit_id"] for row in records}) == 30 and all_families_complete and consistency_ok and condition_metadata_ok and hardware_applied_ok and duplicate_run_ids == 0 and duplicate_group_rep_count == 0 and feasible
    print(f"dataset audit readiness: {'PASS' if audit_pass else 'FAIL'}")
    return 0 if audit_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
