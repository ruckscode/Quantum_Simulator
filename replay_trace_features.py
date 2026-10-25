"""Generate replay-derived trace features without training a model.

The output is explicitly a fresh deterministic replay of stored report inputs;
it is not an original-run trace and does not use diagnosis output as a feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "reports" / "experiments" / "full_dataset"
OUTPUT_DIR = ROOT / "reports" / "experiments" / "replay_derived_trace_features"
DATASET_NAME = "REPLAY_DERIVED_TRACE_FEATURES"
TRACE_ORIGIN = "REPLAY_DERIVED"
TRACE_GENERATION_VERSION = "stage2_operation_trace_seeded_v1"
# Fixed before processing report labels: this is the simulator's existing
# distribution-comparison default threshold, not a dataset-optimized cutoff.
HIGH_DEVIATION_THRESHOLD = 0.05
EXPECTED_CLASSES = ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
FEATURES = (
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
)
PROBABILITY_DIFFERENCE_ROUNDOFF_TOLERANCE = 1e-12
FEATURE_SOURCE_FIELDS = (
    "replayed_trace[*].probability_difference",
    "replayed_trace[*].probability_difference",
    "replayed_trace[*].probability_difference",
    "replayed_trace[*].probability_difference",
    "replayed_trace[*].probability_difference",
)
FORBIDDEN_FEATURE_SOURCES = {
    "ground_truth_label",
    "diagnosis",
    "diagnosis.category",
    "diagnosis.program_evidence_candidates",
    "diagnosis.hardware_evidence_candidates",
    "diagnosis.trace_deviation_candidates",
    "gate_error",
    "readout_error",
    "t1",
    "t2",
    "gate_failed",
    "step_position",
}
METADATA_COLUMNS = (
    "dataset_name",
    "trace_origin",
    "run_id",
    "circuit_family_id",
    "ground_truth_label",
    "repetition_index",
    "random_seed",
    "trace_generation_version",
    "high_deviation_threshold",
    "source_report_run_id",
    "source_report_path",
    "source_report_sha256",
)
IMPLEMENTATION_FILES = {
    "feature_generator": Path(__file__).resolve(),
    "quantum_simulator": Path(inspect.getsourcefile(QuantumSimulator)).resolve(),
}


def _hardware_model(num_qubits: int, values: dict[str, Any] | None) -> HardwareModel | None:
    if values is None:
        return None
    model = HardwareModel(
        num_qubits,
        coupling_map=values.get("coupling_map", []),
        backend_name="replay-derived",
        calibration_timestamp="replay-derived",
        calibration_timestamp_provenance="synthetic",
    )
    for qubit, props in (values.get("qubits") or {}).items():
        model.set_qubit_properties(
            int(qubit),
            **{key: props[key] for key in ("t1", "t2", "readout_error", "status", "available") if key in props},
        )
    gate_values = values.get("gates") or []
    if isinstance(gate_values, dict):
        gate_entries = []
        for name, entries in gate_values.items():
            if isinstance(entries, dict):
                entries = [entries]
            gate_entries.extend({"gate": name, **entry} for entry in entries)
    else:
        gate_entries = gate_values
    for entry in gate_entries:
        model.add_gate_calibration(
            str(entry["gate"]),
            qubits=entry.get("qubits", []),
            **{key: entry[key] for key in ("error", "duration", "supported") if key in entry},
        )
    for edge in values.get("edges") or []:
        left, right = edge.get("qubits", edge.get("pair"))
        model.add_edge(
            int(left), int(right),
            **{key: edge[key] for key in ("direction", "error", "duration", "supported") if key in edge},
        )
    return model


def _extract_features(trace: list[dict[str, Any]]) -> dict[str, float]:
    if not trace:
        raise ValueError("a replayed trace must contain at least one operation row")
    differences = []
    for index, row in enumerate(trace):
        try:
            raw_value = row["probability_difference"]
            if isinstance(raw_value, bool):
                raise TypeError("boolean is not a probability difference")
            value = float(raw_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"trace row {index} must contain a numeric probability_difference"
            ) from exc
        if not math.isfinite(value) or not (
            -PROBABILITY_DIFFERENCE_ROUNDOFF_TOLERANCE
            <= value
            <= 1.0 + PROBABILITY_DIFFERENCE_ROUNDOFF_TOLERANCE
        ):
            raise ValueError(
                f"trace row {index} probability_difference must be finite and between 0 and 1"
            )
        # Total variation is bounded by 1; tolerate and clamp floating-point
        # roundoff from summing normalized state probabilities.
        differences.append(min(1.0, max(0.0, value)))
    total = sum(differences)
    return {
        "trace_max_probability_difference": max(differences),
        "trace_mean_probability_difference": sum(differences) / len(differences),
        "trace_median_probability_difference": float(sorted(differences)[len(differences) // 2])
        if len(differences) % 2
        else (sorted(differences)[len(differences) // 2 - 1] + sorted(differences)[len(differences) // 2]) / 2,
        "trace_deviation_concentration": max(differences) / total if total > 0 else 0.0,
        "trace_high_deviation_fraction": sum(value > HIGH_DEVIATION_THRESHOLD for value in differences) / len(differences),
    }


def _replay_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    configuration = report["configuration"]
    metadata = report["experiment_metadata"]
    ideal = report["circuit"]["ideal"]
    observed = report["circuit"]["observed"]
    num_qubits = int(configuration["num_qubits"])
    seed = int(configuration["random_seed"])
    hardware = _hardware_model(num_qubits, configuration.get("hardware_parameters"))

    _, _, _, trace = QuantumSimulator(num_qubits, seed=seed).run_circuit_traced(
        ideal,
        observed,
        hardware_model=hardware,
        seed=seed,
    )
    features = _extract_features(trace)
    report_bytes = path.read_bytes()
    relative_path = path.resolve().relative_to(ROOT).as_posix()
    run_id = metadata["run_id"]
    return {
        "dataset_name": DATASET_NAME,
        "trace_origin": TRACE_ORIGIN,
        "run_id": run_id,
        "circuit_family_id": metadata["circuit_family_id"],
        # Target column only. It is excluded from FEATURES and feature sources.
        "ground_truth_label": metadata["ground_truth_label"],
        "repetition_index": int(metadata["repetition_index"]),
        "random_seed": seed,
        "trace_generation_version": TRACE_GENERATION_VERSION,
        "high_deviation_threshold": HIGH_DEVIATION_THRESHOLD,
        "source_report_run_id": run_id,
        "source_report_path": relative_path,
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        **features,
    }


def audit_records(records: list[dict[str, Any]], successful_replays: int, replay_failures: list[dict[str, str]]) -> dict[str, Any]:
    classes = Counter(row["ground_truth_label"] for row in records)
    families = {row["circuit_family_id"] for row in records}
    repetitions = Counter((row["circuit_family_id"], row["ground_truth_label"]) for row in records)
    expected_groups = {(family, label) for family in families for label in EXPECTED_CLASSES}
    class_repetition_ok = all(repetitions[group] == 3 for group in expected_groups)
    missing = {
        feature: sum(
            feature not in row or row[feature] is None
            or (isinstance(row[feature], float) and not math.isfinite(row[feature]))
            for row in records
        )
        for feature in FEATURES
    }
    feature_source_set = set(FEATURE_SOURCE_FIELDS)
    ground_truth_not_feature = "ground_truth_label" not in FEATURES
    diagnosis_not_feature = not any("diagnosis" in source for source in feature_source_set)
    return {
        "dataset_name": DATASET_NAME,
        "trace_origin": TRACE_ORIGIN,
        "description": "Fresh seeded replays of stored reports; these are not original-run traces.",
        "trace_generation_version": TRACE_GENERATION_VERSION,
        "high_deviation_threshold": HIGH_DEVIATION_THRESHOLD,
        "high_deviation_rule": "probability_difference > 0.05; fixed from the simulator's default statistical threshold before label inspection",
        "source_report_count": successful_replays + len(replay_failures),
        "successful_replays": successful_replays,
        "replay_failure_count": len(replay_failures),
        "replay_failures": replay_failures,
        "class_counts": {label: classes[label] for label in EXPECTED_CLASSES},
        "class_count_check": all(classes[label] == 90 for label in EXPECTED_CLASSES),
        "circuit_family_count": len(families),
        "circuit_family_check": len(families) == 30,
        "repetitions_per_family_class": 3,
        "repetition_group_count": len(repetitions),
        "repetitions_check": len(expected_groups) == 90 and class_repetition_ok,
        "feature_names": list(FEATURES),
        "feature_source_fields": list(FEATURE_SOURCE_FIELDS),
        "feature_missing_counts": missing,
        "no_missing_features_check": all(count == 0 for count in missing.values()),
        "ground_truth_label_is_target_only": True,
        "ground_truth_not_in_features_check": ground_truth_not_feature,
        "diagnosis_output_used_as_feature": False,
        "diagnosis_not_in_features_check": diagnosis_not_feature,
        "forbidden_feature_sources": sorted(FORBIDDEN_FEATURE_SOURCES),
        "audit_pass": (
            successful_replays == 270
            and not replay_failures
            and len(records) == 270
            and all(classes[label] == 90 for label in EXPECTED_CLASSES)
            and len(families) == 30
            and len(expected_groups) == 90
            and class_repetition_ok
            and all(count == 0 for count in missing.values())
            and ground_truth_not_feature
            and diagnosis_not_feature
        ),
    }


def generate_dataset(source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    reports = []
    for path in sorted(source_dir.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("status") == "COMPLETED" and report.get("experiment_metadata", {}).get("ground_truth_label") in EXPECTED_CLASSES:
            reports.append((path, report))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path, report in reports:
        try:
            records.append(_replay_report(path, report))
        except Exception as exc:
            failures.append({"source_report": path.resolve().relative_to(ROOT).as_posix(), "error": f"{type(exc).__name__}: {exc}"})

    audit = audit_records(records, len(records), failures)
    audit["implementation_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in IMPLEMENTATION_FILES.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = {
        "dataset_name": DATASET_NAME,
        "trace_origin": TRACE_ORIGIN,
        "description": "Replay-derived features from fresh seeded traces; not the original execution trace.",
        "target_field": "ground_truth_label",
        "input_feature_names": list(FEATURES),
        "feature_source_fields": list(FEATURE_SOURCE_FIELDS),
        "trace_generation_version": TRACE_GENERATION_VERSION,
        "high_deviation_threshold": HIGH_DEVIATION_THRESHOLD,
        "records": records,
    }
    (output_dir / "replay_trace_features.json").write_text(
        json.dumps(dataset_json, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    csv_columns = list(METADATA_COLUMNS) + list(FEATURES)
    with (output_dir / "replay_trace_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(records)
    audit["replay_feature_csv_sha256"] = hashlib.sha256(
        (output_dir / "replay_trace_features.csv").read_bytes()
    ).hexdigest()
    (output_dir / "audit_report.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    audit = generate_dataset(args.source_dir, args.output_dir)
    print(json.dumps(audit, indent=2))
    return 0 if audit["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

