"""Contract-specific Phase 5 configuration, preflight, execution and scoring."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

from experiment_config import ExperimentConfig
from quantum_simulator.diagnosis_engine import diagnose_execution
from quantum_simulator.ml_diagnosis import CLASSES, MIN_CONFIDENCE
from quantum_simulator.observable_model_adapter import PortableLogisticRegressionAdapter

from . import phase5_benchmark_infrastructure as p5

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "reports" / "experiments" / "phase5_benchmark"
SPECIFICITY_CASES = p5._SPECIFICITY_CASES
F3_GATE_DIAGNOSTIC_CASES = p5._F3_GATE_DIAGNOSTIC_CASES
SENSITIVITY_CASES = p5._SENSITIVITY_CASES


@dataclass(frozen=True)
class BenchmarkConfiguration:
    case: p5.Case
    seed: int
    shots: int
    config: ExperimentConfig

    @property
    def key(self) -> tuple[str, int, int]:
        return self.case.case_id, self.seed, self.shots


def _profile(case: p5.Case) -> dict[str, Any]:
    readout_p = .15 if case.suffix == "C15" else {
        "R02": .02, "R06": .06, "R15": .15, "R30": .30,
    }.get(case.suffix, 0.0)
    target_qubit = p5._R_TARGET[case.family_id]
    qubits = {
        q: {
            "readout_error": readout_p if (case.suffix.startswith("R") or case.suffix == "C15") and q == target_qubit else 0.0,
            "t1": 1e6, "t2": 1e6, "status": "available", "available": True,
        }
        for q in range(4)
    }
    keys = set()
    for circuit in (case.expected_circuit, case.observed_circuit):
        for name, *operands in circuit:
            keys.add((name, tuple(operands)))
    gate_target = p5._G_TARGET[case.family_id]
    gate_p = {"G02": .02, "G06": .06, "G15": .15, "G30": .30}.get(case.suffix, 0.0)
    gates = [
        {
            "gate": name,
            "qubits": list(operands),
            "error": gate_p if case.suffix.startswith("G") and (name, operands) == (gate_target[0], tuple(gate_target[1:])) else 0.0,
            "duration": 0.0,
            "supported": True,
        }
        for name, operands in sorted(keys)
    ]
    return {"qubits": qubits, "gates": gates, "coupling_map": [], "edges": []}


def build_phase5_configurations() -> tuple[BenchmarkConfiguration, ...]:
    """Build and fail-closed validate all 60 × 5 × 3 frozen configurations."""
    cases = p5.build_ledger()
    p5.validate_ledger(cases)
    p5.validate_family_split(p5.FAMILIES)
    configurations = []
    for case in cases:
        profile = _profile(case)
        p5.validate_calibration(profile, case.expected_circuit, case.observed_circuit, case=case)
        for seed in p5.SEEDS:
            for shots in p5.SHOT_COUNTS:
                # C15 is multi-label and must not be represented as one of the
                # learner's three single-cause training classes.
                label = case.truth if isinstance(case.truth, str) else None
                config = ExperimentConfig(
                    experiment_name=f"phase5_{case.case_id}_seed{seed}_shots{shots}",
                    num_qubits=4,
                    shots=shots,
                    circuit=[list(op) for op in case.expected_circuit],
                    observed_circuit=[list(op) for op in case.observed_circuit],
                    random_seed=seed,
                    anomaly_threshold=.05,
                    hardware_parameters=profile,
                    circuit_family_id=case.family_id,
                    ground_truth_label=label,
                    program_fault_id=case.suffix if label == "PROGRAM_FAULT" else None,
                    program_fault_description=case.suffix if label == "PROGRAM_FAULT" else None,
                    hardware_condition_id=case.suffix if label == "HARDWARE_ANOMALY" else None,
                    hardware_affected_component=("READOUT_QUBIT" if case.suffix.startswith("R") else "GATE_TUPLE") if label == "HARDWARE_ANOMALY" else None,
                )
                configurations.append(BenchmarkConfiguration(case, seed, shots, config))
    keys = [item.key for item in configurations]
    expected = {
        (case.case_id, seed, shots)
        for case in cases for seed in p5.SEEDS for shots in p5.SHOT_COUNTS
    }
    if len(keys) != 900 or len(set(keys)) != 900 or set(keys) != expected:
        raise ValueError("Phase 5 configuration grid must contain each of the exact 900 rows once")
    return tuple(configurations)


def preflight_phase5_benchmark(
    *, result_dir: str | Path = DEFAULT_RESULT_DIR,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only dry-run: validate grid, every profile, output path and model."""
    configs = build_phase5_configurations()
    reproducibility_manifest = build_phase5_reproducibility_manifest(configs)
    destination = Path(result_dir)
    if destination.exists():
        raise FileExistsError(f"refusing existing Phase 5 benchmark result directory: {destination}")
    model = PortableLogisticRegressionAdapter(artifact_path) if artifact_path else PortableLogisticRegressionAdapter()
    if tuple(model.feature_names) != p5.FEATURE_NAMES or model.probabilities_calibrated is not False or model.confidence_cutoff != .60:
        raise ValueError("Phase 5 model/schema/confidence preflight failed")
    families = {item.case.family_id for item in configs}
    if families != set(p5.FAMILIES):
        raise ValueError("benchmark configurations do not cover all frozen families")
    return {
        "status": "PREFLIGHT_READY",
        "configuration_count": len(configs),
        "case_count": len(p5.build_ledger()),
        "seed_count": len(p5.SEEDS),
        "shot_counts": list(p5.SHOT_COUNTS),
        "profile_count_validated": len(p5.build_ledger()),
        "shot_circuit_executions_if_later_authorized": 3_225_600,
        "deterministic_config_reference_artifacts_verified": len(reproducibility_manifest),
        "reproducibility_check_shot_circuit_executions": 0,
        "stochastic_sampling_repeated": False,
        "partitions": {name: sorted(values) for name, values in p5.PARTITIONS.items()},
        "artifact": str(model.artifact_path),
        "feature_order": list(model.feature_names),
        "class_order": list(CLASSES),
        "confidence_cutoff": model.confidence_cutoff,
        "probabilities_calibrated": model.probabilities_calibrated,
        "sensitivity_templates": len(SENSITIVITY_CASES),
        "sensitivity_per_shot_denominator": len(SENSITIVITY_CASES) * len(p5.SEEDS),
        "specificity_templates": len(SPECIFICITY_CASES),
        "specificity_per_shot_denominator": len(SPECIFICITY_CASES) * len(p5.SEEDS),
        "specificity_membership": "owner decision 2026-10-13: N0 plus masked D− controls; F3 G cases separate",
        "reserved_family_f3_gate_diagnostic_templates": len(F3_GATE_DIAGNOSTIC_CASES),
        "result_dir": str(destination),
        "cases_executed": False,
    }


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_ABSTENTION_DISPOSITIONS = {
    "ABSTAIN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS / INSUFFICIENT_EVIDENCE",
}


def adjudicated_class_from_disposition(disposition: str) -> str:
    """Map the final evidence disposition to the contract's scoring classes."""
    if not isinstance(disposition, str):
        raise ValueError("diagnosis disposition must be a string")
    value = disposition.strip().upper()
    if value == "NO_ANOMALY":
        return "NORMAL"
    if value in ("PROGRAM_FAULT", "HARDWARE_ANOMALY"):
        return value
    if value in _ABSTENTION_DISPOSITIONS:
        return "ABSTAIN"
    raise ValueError(f"unsupported Phase 5 diagnosis disposition: {disposition!r}")


def _is_c15_ambiguous(disposition: str) -> bool:
    # The C15 acceptance target names the categorical AMBIGUOUS result. A
    # generic insufficient-evidence abstention is not successful C15 handling.
    return isinstance(disposition, str) and disposition.strip().upper() == "AMBIGUOUS"


def build_phase5_reproducibility_manifest(
    configurations: Sequence[BenchmarkConfiguration] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compare regenerated deterministic config/reference artifacts, without sampling."""
    first = tuple(configurations) if configurations is not None else build_phase5_configurations()
    regenerated = build_phase5_configurations()
    if [item.key for item in first] != [item.key for item in regenerated]:
        raise ValueError("regenerated Phase 5 configuration keys differ")

    def fingerprints(item: BenchmarkConfiguration) -> tuple[str, str]:
        config_fingerprint = _hash_json({
            "case_id": item.case.case_id,
            "seed": item.seed,
            "shots": item.shots,
            "config": item.config.to_dict(),
        })
        reference_fingerprint = _hash_json({
            "expected_circuit": item.case.expected_circuit,
            "exact_d": item.case.exact_d,
            "exact_tvd": item.case.exact_tvd,
        })
        return config_fingerprint, reference_fingerprint

    manifest = []
    for left, right in zip(first, regenerated):
        config_left, reference_left = fingerprints(left)
        config_right, reference_right = fingerprints(right)
        manifest.append({
            "case_id": left.case.case_id, "seed": left.seed, "shots": left.shots,
            "configuration_fingerprint": config_left,
            "regenerated_configuration_fingerprint": config_right,
            "reference_fingerprint": reference_left,
            "regenerated_reference_fingerprint": reference_right,
            "input_fingerprint": _hash_json({"configuration": config_left, "reference": reference_left}),
        })
    if any(row["configuration_fingerprint"] != row["regenerated_configuration_fingerprint"]
           or row["reference_fingerprint"] != row["regenerated_reference_fingerprint"] for row in manifest):
        raise ValueError("deterministic Phase 5 configuration/reference regeneration mismatch")
    return tuple(manifest)


def run_phase5_benchmark(
    *,
    dry_run: bool = True,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Preflight always; execute only when explicitly called with dry_run=False."""
    preflight = preflight_phase5_benchmark(result_dir=result_dir, artifact_path=artifact_path)
    if dry_run:
        return preflight
    from experiments.run_experiment import _build_hardware_model, run_experiment

    configs = build_phase5_configurations()
    reproducibility_manifest = build_phase5_reproducibility_manifest(configs)
    reproducibility_by_key = {
        (row["case_id"], row["seed"], row["shots"]): row for row in reproducibility_manifest
    }
    model = PortableLogisticRegressionAdapter(artifact_path) if artifact_path else PortableLogisticRegressionAdapter()
    output_dir = Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    observations = []
    for item in configs:
        output_path = output_dir / f"{item.config.experiment_name}.json"
        if output_path.exists():
            raise FileExistsError(f"refusing existing Phase 5 result: {output_path}")
        experiment_result = run_experiment(item.config, output_path=output_path)
        if experiment_result.status != "COMPLETED" or experiment_result.errors:
            raise RuntimeError(f"Phase 5 case failed at {item.key}: {experiment_result.errors}")
        diagnosis = diagnose_execution(
            experiment_result.ideal_distribution,
            experiment_result.observed_distribution,
            expected_circuit=item.config.circuit,
            observed_circuit=item.config.observed_circuit,
            hardware_model=_build_hardware_model(item.config),
            threshold=.05,
            shots=item.shots,
            classifier=model,
            ml_min_confidence=MIN_CONFIDENCE,
        )
        confidence = diagnosis.ml_confidence
        if confidence is None:
            raise RuntimeError(f"Phase 5 classifier did not run at {item.key}")
        disposition = "INSUFFICIENT_EVIDENCE" if confidence < MIN_CONFIDENCE else diagnosis.category.value
        predicted_causes = []
        predicted_targets = []
        localized = diagnosis if confidence >= MIN_CONFIDENCE else None
        if localized is not None:
            if localized.program_evidence_candidates:
                predicted_causes.append("PROGRAM_FAULT")
                candidate = localized.program_evidence_candidates[0]
                if hasattr(candidate, "index"):
                    predicted_targets.append(f"program_index:{candidate.index}")
            if localized.hardware_evidence_candidates:
                predicted_causes.append("HARDWARE_ANOMALY")
                candidate = localized.hardware_evidence_candidates[0]
                if candidate.component_type == "qubit":
                    predicted_targets.append(f"qubit:{candidate.component}")
                elif candidate.component_type == "gate":
                    gate = candidate.gate_name or candidate.reason.partition("Gate calibration for ")[2].split(" ", 1)[0]
                    if gate:
                        predicted_targets.append(f"gate:{gate}:{','.join(map(str, candidate.component))}")
        if item.case.suffix == "C15" and confidence >= MIN_CONFIDENCE:
            from quantum_simulator.diagnosis_engine import diagnose_execution
            combined = diagnose_execution(
                experiment_result.ideal_distribution, experiment_result.observed_distribution,
                expected_circuit=item.config.circuit,
                observed_circuit=item.config.observed_circuit,
                hardware_model=_build_hardware_model(item.config), threshold=.05, shots=item.shots,
            )
            combined_program = combined.program_evidence_candidates
            combined_hardware = combined.hardware_evidence_candidates
            predicted_causes = []
            if combined_program:
                predicted_causes.append("PROGRAM_FAULT")
                candidate = combined_program[0]
                if hasattr(candidate, "index"):
                    predicted_targets = [t for t in predicted_targets if not t.startswith("program_index:")]
                    predicted_targets.append(f"program_index:{candidate.index}")
            if combined_hardware:
                predicted_causes.append("HARDWARE_ANOMALY")
                candidate = combined_hardware[0]
                if candidate.component_type == "qubit":
                    predicted_targets = [t for t in predicted_targets if not t.startswith("qubit:")]
                    predicted_targets.append(f"qubit:{candidate.component}")
                elif candidate.component_type == "gate":
                    gate = candidate.gate_name or candidate.reason.partition("Gate calibration for ")[2].split(" ", 1)[0]
                    if gate:
                        predicted_targets = [t for t in predicted_targets if not t.startswith("gate:")]
                        predicted_targets.append(f"gate:{gate}:{','.join(map(str, candidate.component))}")
            disposition = "AMBIGUOUS" if combined.category.value == "AMBIGUOUS / INSUFFICIENT_EVIDENCE" and len(predicted_causes) == 2 else combined.category.value
        anomaly = p5.anomaly_observation(
            diagnosis.ml_features["total_variation_distance"], item.shots,
        )
        reproducibility_row = reproducibility_by_key[item.key]
        input_fingerprint = _hash_json({
            "configuration": reproducibility_row["configuration_fingerprint"],
            "reference": reproducibility_row["reference_fingerprint"],
        })
        artifact_fingerprint = _hash_json({"result_fingerprint": experiment_result.deterministic_fingerprint(), "prediction": diagnosis.ml_predicted_class, "probabilities": diagnosis.ml_class_probabilities, "disposition": disposition})
        observations.append({
            "case_id": item.case.case_id, "seed": item.seed, "shots": item.shots,
            "anomaly_detected": anomaly["detected"],
            "raw_model_prediction": diagnosis.ml_predicted_class,
            "raw_model_confidence": confidence,
            "disposition": disposition,
            "adjudicated_class": adjudicated_class_from_disposition(disposition),
            "predicted_targets": predicted_targets, "predicted_causes": predicted_causes,
            "artifact_fingerprint": artifact_fingerprint, "input_fingerprint": input_fingerprint,
        })
    return {"status": "EXECUTED", "preflight": preflight, "observation_count": len(observations),
            "observations": observations, "reproducibility_manifest": reproducibility_manifest}


def evaluate_phase5_acceptance(
    observations: Sequence[Mapping[str, Any]],
    *, reproducibility_manifest: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Apply every numerical Phase 5.4 acceptance target to a complete ledger."""
    cases = {case.case_id: case for case in p5.build_ledger()}
    expected = {(case_id, seed, shots) for case_id in cases for seed in p5.SEEDS for shots in p5.SHOT_COUNTS}
    schema = {"case_id", "seed", "shots", "anomaly_detected", "raw_model_prediction", "raw_model_confidence", "disposition", "adjudicated_class", "predicted_targets", "predicted_causes", "artifact_fingerprint", "input_fingerprint"}
    indexed = {}
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != schema:
            raise ValueError("scoring row schema does not match Phase 5 acceptance contract")
        key = (row["case_id"], row["seed"], row["shots"])
        if key not in expected:
            raise ValueError(f"unsupported scoring row: {key}")
        if key in indexed:
            raise ValueError(f"duplicate scoring row: {key}")
        if type(row["anomaly_detected"]) is not bool or row["raw_model_prediction"] not in CLASSES:
            raise ValueError("invalid anomaly decision or raw model prediction")
        if not isinstance(row["raw_model_confidence"], (int, float)) or isinstance(row["raw_model_confidence"], bool) or not math.isfinite(row["raw_model_confidence"]) or not 0 <= row["raw_model_confidence"] <= 1:
            raise ValueError("invalid confidence value")
        if not all(isinstance(row[name], str) and row[name] for name in ("disposition", "artifact_fingerprint", "input_fingerprint")):
            raise ValueError("invalid disposition or missing fingerprints")
        scored_class = adjudicated_class_from_disposition(row["disposition"])
        if row["adjudicated_class"] != scored_class:
            raise ValueError("adjudicated_class does not match the evidence disposition")
        if not isinstance(row["predicted_targets"], (list, tuple)) or not all(isinstance(x, str) for x in row["predicted_targets"]):
            raise ValueError("predicted_targets must contain strings")
        if not isinstance(row["predicted_causes"], (list, tuple)) or not set(row["predicted_causes"]) <= {"PROGRAM_FAULT", "HARDWARE_ANOMALY"}:
            raise ValueError("predicted_causes must contain supported causes")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"missing scoring rows: {len(expected - set(indexed))}")

    per_shot = {}
    for shots in p5.SHOT_COUNTS:
        pos = [indexed[(case_id, seed, shots)] for case_id in SENSITIVITY_CASES for seed in p5.SEEDS]
        neg = [indexed[(case_id, seed, shots)] for case_id in SPECIFICITY_CASES for seed in p5.SEEDS]
        sensitivity = sum(x["anomaly_detected"] for x in pos) / len(pos)
        specificity = sum(not x["anomaly_detected"] for x in neg) / len(neg)
        seed_scores = [sum(indexed[(case_id, seed, shots)]["anomaly_detected"] for case_id in SENSITIVITY_CASES) / len(SENSITIVITY_CASES) for seed in p5.SEEDS]
        family_sensitivity = {}
        for family in p5.FAMILIES:
            family_templates = [case_id for case_id in SENSITIVITY_CASES if case_id.startswith(family + "-")]
            vals = [indexed[(case_id, seed, shots)]["anomaly_detected"] for case_id in family_templates for seed in p5.SEEDS]
            family_sensitivity[family] = {"numerator": sum(vals), "denominator": len(vals), "sensitivity": sum(vals) / len(vals)}
        per_shot[shots] = {
            "sensitivity": sensitivity, "sensitivity_numerator": sum(x["anomaly_detected"] for x in pos), "sensitivity_denominator": len(pos), "sensitivity_pass": sensitivity >= .80,
            "specificity": specificity, "specificity_numerator": sum(not x["anomaly_detected"] for x in neg), "specificity_denominator": len(neg), "specificity_pass": specificity >= .90,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "per_seed_sensitivity": dict(zip(p5.SEEDS, seed_scores)), "seed_population_sd": pstdev(seed_scores),
            "seed_mean": sum(seed_scores) / len(seed_scores), "seed_min": min(seed_scores), "seed_max": max(seed_scores),
            "sensitivity_by_family": family_sensitivity,
            "seed_sd_pass": pstdev(seed_scores) <= .10,
        }

    # F1-F5 single-cause rows, excluding C15 and p=.02 descriptive strata.
    class_case_ids = {c.case_id for c in cases.values() if c.suffix not in ("C15", "R02", "G02")}
    class_labels = {c.case_id: c.truth for c in cases.values() if isinstance(c.truth, str)}
    confusion = {label: {prediction: 0 for prediction in (*CLASSES, "ABSTAIN")} for label in CLASSES}
    raw_model_confusion = {label: {prediction: 0 for prediction in CLASSES} for label in CLASSES}
    for (case_id, _, _), row in indexed.items():
        if case_id in class_case_ids:
            confusion[class_labels[case_id]][row["adjudicated_class"]] += 1
            raw_model_confusion[class_labels[case_id]][row["raw_model_prediction"]] += 1
    per_class = {}
    for label in CLASSES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in CLASSES)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(CLASSES)
    class_pass = macro_f1 >= .75 and per_class["PROGRAM_FAULT"]["recall"] >= .80 and per_class["HARDWARE_ANOMALY"]["recall"] >= .80
    classification_by_family = {}
    for family in p5.FAMILIES:
        family_ids = {case_id for case_id in class_case_ids if case_id.startswith(family + "-")}
        family_confusion = {label: {prediction: 0 for prediction in (*CLASSES, "ABSTAIN")} for label in CLASSES}
        for (case_id, _, _), row in indexed.items():
            if case_id in family_ids:
                family_confusion[class_labels[case_id]][row["adjudicated_class"]] += 1
        family_recalls = {
            label: family_confusion[label][label] / sum(family_confusion[label].values())
            if sum(family_confusion[label].values()) else 0.0
            for label in CLASSES
        }
        classification_by_family[family] = {"confusion_matrix": family_confusion, "class_recall": family_recalls,
                                            "denominator": sum(sum(row.values()) for row in family_confusion.values())}
    p02_ids = {case.case_id for case in cases.values() if case.suffix in ("R02", "G02")}
    p02_descriptive = {
        "denominator": sum((case_id, seed, shots) in indexed for case_id in p02_ids for seed in p5.SEEDS for shots in p5.SHOT_COUNTS),
        "raw_model_prediction_counts": {label: sum(row["raw_model_prediction"] == label for (case_id, _, _), row in indexed.items() if case_id in p02_ids) for label in CLASSES},
        "adjudicated_class_counts": {label: sum(row["adjudicated_class"] == label for (case_id, _, _), row in indexed.items() if case_id in p02_ids) for label in (*CLASSES, "ABSTAIN")},
        "anomaly_detected": sum(row["anomaly_detected"] for (case_id, _, _), row in indexed.items() if case_id in p02_ids),
    }

    target_stats = {name: {"correct": 0, "denominator": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
                    for name in ("program", "qubit", "gate_tuple")}
    localization_by_family = {family: {"correct": 0, "denominator": 0} for family in p5.FAMILIES}
    c15_target_exact = []
    for (case_id, _, _), row in indexed.items():
        case = cases[case_id]
        predicted = set(row["predicted_targets"])
        if case.suffix in ("P1", "P2"):
            index = p5._PROGRAM_MUTATIONS[case.family_id][0 if case.suffix == "P1" else 1][0]
            expected_target, kind = f"program_index:{index}", "program"
        elif case.suffix in ("R06", "R15", "R30"):
            expected_target, kind = f"qubit:{p5._R_TARGET[case.family_id]}", "qubit"
        elif case.suffix in ("G06", "G15", "G30"):
            gate = p5._G_TARGET[case.family_id]
            expected_target, kind = f"gate:{gate[0]}:{','.join(map(str, gate[1:]))}", "gate_tuple"
        elif case.suffix == "C15":
            program_index = p5._PROGRAM_MUTATIONS[case.family_id][0][0]
            qubit = p5._R_TARGET[case.family_id]
            c15_target_exact.append({f"program_index:{program_index}", f"qubit:{qubit}"} <= predicted)
            continue
        else:
            continue
        allowed_prefix = {"program": "program_index:", "qubit": "qubit:", "gate_tuple": "gate:"}[kind]
        predicted_of_type = {value for value in predicted if value.startswith(allowed_prefix)}
        correct = expected_target in predicted_of_type
        stats = target_stats[kind]
        stats["denominator"] += 1
        stats["correct"] += int(correct)
        stats["true_positive"] += int(correct)
        stats["false_negative"] += int(not correct)
        stats["false_positive"] += len(predicted_of_type - {expected_target})
        localization_by_family[case.family_id]["denominator"] += 1
        localization_by_family[case.family_id]["correct"] += int(correct)
    target_denominator = sum(stats["denominator"] for stats in target_stats.values())
    target_correct_total = sum(stats["correct"] for stats in target_stats.values())
    localization_accuracy = target_correct_total / target_denominator if target_denominator else 0.0
    localization_pass = bool(target_denominator) and localization_accuracy >= .70
    for stats in target_stats.values():
        stats["accuracy"] = stats["correct"] / stats["denominator"] if stats["denominator"] else None
        predicted_count = stats["true_positive"] + stats["false_positive"]
        stats["precision"] = stats["true_positive"] / predicted_count if predicted_count else 0.0
        stats["recall"] = stats["true_positive"] / (stats["true_positive"] + stats["false_negative"]) if stats["denominator"] else 0.0
    for stats in localization_by_family.values():
        stats["accuracy"] = stats["correct"] / stats["denominator"] if stats["denominator"] else None

    c15_ids = {f"{family}-C15" for family in p5.FAMILIES}
    c15 = [row for (case_id, _, _), row in indexed.items() if case_id in c15_ids]
    c15_ambiguous = [_is_c15_ambiguous(row["disposition"]) for row in c15]
    c15_recall = sum(c15_ambiguous) / len(c15_ambiguous)
    c15_by_family = {
        family: {
            "correct_ambiguous": sum(_is_c15_ambiguous(row["disposition"]) for (case_id, _, _), row in indexed.items() if case_id == f"{family}-C15"),
            "denominator": sum(case_id == f"{family}-C15" for case_id, _, _ in indexed),
        } for family in p5.FAMILIES
    }
    for stats in c15_by_family.values():
        stats["ambiguous_recall"] = stats["correct_ambiguous"] / stats["denominator"] if stats["denominator"] else 0.0
    all_rows = list(indexed.values())
    covered = [row for row in all_rows if row["raw_model_confidence"] >= MIN_CONFIDENCE]
    single_covered = [row for (case_id, _, _), row in indexed.items() if case_id not in c15_ids and row["raw_model_confidence"] >= MIN_CONFIDENCE]
    selective_risk = sum(row["adjudicated_class"] != class_labels[row["case_id"]] for row in single_covered) / len(single_covered) if single_covered else 0.0
    confident_single_c15 = sum(
        row["raw_model_confidence"] >= MIN_CONFIDENCE
        and row["raw_model_prediction"] in ("PROGRAM_FAULT", "HARDWARE_ANOMALY")
        for row in c15
    )
    f3_gate_diagnostics = {
        shots: {
            case_id: {
                "false_positives": sum(indexed[(case_id, seed, shots)]["anomaly_detected"] for seed in p5.SEEDS),
                "denominator": len(p5.SEEDS),
                "sampled_detection_rate": sum(indexed[(case_id, seed, shots)]["anomaly_detected"] for seed in p5.SEEDS) / len(p5.SEEDS),
            }
            for case_id in sorted(F3_GATE_DIAGNOSTIC_CASES)
        }
        for shots in p5.SHOT_COUNTS
    }

    pairs = list(reproducibility_manifest or ())
    pair_schema = {
        "case_id", "seed", "shots", "configuration_fingerprint", "regenerated_configuration_fingerprint",
        "reference_fingerprint", "regenerated_reference_fingerprint", "input_fingerprint",
    }
    if len(pairs) != len(expected) or any(not isinstance(pair, Mapping) or set(pair) != pair_schema for pair in pairs):
        raise ValueError("reproducibility manifest must cover all 900 configurations and reference artifacts")
    pair_keys = [(pair["case_id"], pair["seed"], pair["shots"]) for pair in pairs]
    if len(set(pair_keys)) != len(pair_keys) or set(pair_keys) != expected:
        raise ValueError("reproducibility manifest must map each scored configuration exactly once")
    repro_pass = all(
        pair["configuration_fingerprint"] == pair["regenerated_configuration_fingerprint"]
        and pair["reference_fingerprint"] == pair["regenerated_reference_fingerprint"]
        and pair["input_fingerprint"] == indexed[(pair["case_id"], pair["seed"], pair["shots"])]["input_fingerprint"]
        for pair in pairs
    )
    checks = {
        "sensitivity_each_shot": all(v["sensitivity_pass"] for v in per_shot.values()),
        "specificity_each_shot": all(v["specificity_pass"] for v in per_shot.values()),
        "classification": class_pass,
        "localization": localization_pass,
        "c15_ambiguous_recall": c15_recall >= .80,
        "reproducibility_seed_sd": all(v["seed_sd_pass"] for v in per_shot.values()),
        "reproducibility_deterministic_config_reference_artifacts": repro_pass,
    }
    return {
        "acceptance_checks": checks, "overall_acceptance_pass": all(checks.values()),
        "output_detection_by_shot": per_shot,
        "output_detection_by_case": {
            case_id: {shots: {"detected": sum(indexed[(case_id, seed, shots)]["anomaly_detected"] for seed in p5.SEEDS),
                              "denominator": len(p5.SEEDS),
                              "rate": sum(indexed[(case_id, seed, shots)]["anomaly_detected"] for seed in p5.SEEDS) / len(p5.SEEDS)}
                      for shots in p5.SHOT_COUNTS}
            for case_id in sorted(cases)
        },
        "reserved_family_f3_gate_diagnostic": f3_gate_diagnostics,
        "classification": {"confusion_matrix": confusion, "raw_model_confusion_matrix": raw_model_confusion,
                           "per_class": per_class, "macro_f1": macro_f1,
                           "denominator": sum(sum(row.values()) for row in confusion.values()),
                           "by_family": classification_by_family, "p02_descriptive": p02_descriptive},
        "localization": {"by_type": target_stats, "by_family": localization_by_family,
                         "exact_target_accuracy": localization_accuracy, "denominator": target_denominator,
                         "c15_combined_target_recovery": {"correct": sum(c15_target_exact), "denominator": len(c15_target_exact)}},
        "abstention": {"coverage": len(covered) / len(all_rows), "covered_rows": len(covered), "denominator": len(all_rows), "adjudicated_non_abstain_coverage": sum(row["adjudicated_class"] != "ABSTAIN" for row in all_rows) / len(all_rows), "adjudicated_abstentions": sum(row["adjudicated_class"] == "ABSTAIN" for row in all_rows), "selective_risk_single_cause": selective_risk, "selective_risk_denominator": len(single_covered), "c15_ambiguous_recall": c15_recall, "c15_correct_ambiguous": sum(c15_ambiguous), "c15_denominator": len(c15), "c15_by_family": c15_by_family, "c15_incorrect_confident_single_cause_rate": confident_single_c15 / len(c15)},
        "paired_control_differences": {
            case_id: {
                shots: {
                    "differences_vs_family_n0_by_seed": [
                        int(indexed[(case_id, seed, shots)]["anomaly_detected"])
                        - int(indexed[(f"{cases[case_id].family_id}-N0", seed, shots)]["anomaly_detected"])
                        for seed in p5.SEEDS
                    ],
                    "denominator": len(p5.SEEDS),
                } for shots in p5.SHOT_COUNTS
            } for case_id in sorted(cases) if cases[case_id].suffix != "N0"
        },
        "reproducibility": {"seed_sd_limit": .10, "deterministic_config_reference_artifacts": len(pairs),
                            "deterministic_config_reference_pass": repro_pass,
                            "stochastic_sampled_artifacts_repeated": False,
                            "additional_shot_circuit_executions": 0,
                            "input_fingerprints": {"|".join(map(str, key)): row["input_fingerprint"] for key, row in indexed.items()},
                            "artifact_fingerprints": {"|".join(map(str, key)): row["artifact_fingerprint"] for key, row in indexed.items()}},
        "specificity_templates": sorted(SPECIFICITY_CASES), "specificity_includes_f3_gate_cases": False,
        "specificity_decision_record": "Phase 5.8 decision memo owner addendum, 2026-10-13",
        "c15_excluded_from_three_class_metrics": True,
    }


__all__ = [
    "BenchmarkConfiguration", "SPECIFICITY_CASES", "F3_GATE_DIAGNOSTIC_CASES", "SENSITIVITY_CASES",
    "build_phase5_configurations", "preflight_phase5_benchmark",
    "run_phase5_benchmark", "evaluate_phase5_acceptance", "build_phase5_reproducibility_manifest",
    "adjudicated_class_from_disposition",
]
