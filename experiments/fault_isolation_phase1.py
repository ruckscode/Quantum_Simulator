"""Small paired fault-isolation experiment using the production runner APIs.

Ground-truth categories are declared in the scenario table, independently of
the diagnosis result. All execution and diagnosis are delegated to
``experiments.run_experiment.run_experiment``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiment_config import ExperimentConfig
from experiments.run_experiment import run_experiment
from quantum_simulator.diagnosis_engine import DiagnosisCategory


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "experiments" / "controlled_fault_isolation_phase1"
SHOTS = 512
SEED = 20261013
NORMAL_CLASS = "NORMAL"
PROGRAM_CLASS = "PROGRAM_FAULT"
HARDWARE_CLASS = "HARDWARE_ANOMALY"
UNCERTAIN_CLASS = DiagnosisCategory.AMBIGUOUS.value

REFERENCE = [("x", 0), ("measure", [0, 1])]
PROGRAM_MUTATION = [("measure", [0, 1])]
COHERENCE_CIRCUIT = [("x", 0), ("i", 0), ("measure", [0, 1])]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    pair_memberships: tuple[tuple[str, str], ...]
    ground_truth_cause: str
    ground_truth_class: str
    expected_category: str
    affected_component: str | None
    expected_detection: bool | None
    localization_targets: tuple[dict[str, Any], ...]
    config: ExperimentConfig


def _profile(n_qubits: int = 2) -> dict[str, Any]:
    return {
        "coupling_map": [],
        "qubits": {
            str(q): {"t1": 1e6, "t2": 1e6, "readout_error": 0.0}
            for q in range(n_qubits)
        },
        "gates": [],
    }


def _healthy_coherence_profile() -> dict[str, Any]:
    profile = _profile()
    profile["gates"] = [{"gate": "i", "qubits": [0], "error": 0.0, "duration": 2.0}]
    return profile


def _make_scenario(
    scenario_id: str,
    *,
    circuit: list[Any] = REFERENCE,
    observed_circuit: list[Any] | None = None,
    hardware: dict[str, Any] | None = None,
    observed_counts: dict[str, int] | None = None,
    pair_memberships: tuple[tuple[str, str], ...] = (),
    cause: str,
    ground_truth_class: str,
    affected_component: str | None = None,
    expected_detection: bool | None,
    targets: tuple[dict[str, Any], ...] = (),
) -> Scenario:
    config_label = ground_truth_class if ground_truth_class in {
        NORMAL_CLASS, PROGRAM_CLASS, HARDWARE_CLASS,
    } else None
    config_values: dict[str, Any] = {
        "experiment_name": f"fault_isolation_{scenario_id}",
        "num_qubits": 2,
        "shots": SHOTS,
        "random_seed": SEED,
        "circuit": copy.deepcopy(circuit),
        "observed_circuit": copy.deepcopy(observed_circuit) if observed_circuit is not None else None,
        "hardware_parameters": copy.deepcopy(hardware),
        "observed_counts": copy.deepcopy(observed_counts),
        "anomaly_threshold": 0.05,
        "ground_truth_label": config_label,
        "circuit_family_id": "fault_isolation_family_01",
    }
    if ground_truth_class == PROGRAM_CLASS:
        config_values.update({
            "program_fault_id": "fault_remove_x_q0",
            "program_fault_description": "Remove the X operation on qubit 0.",
            "program_fault_operation_index": 0,
            "program_fault_qubits": [0],
        })
    if ground_truth_class == HARDWARE_CLASS:
        config_values.update({
            "hardware_condition_id": f"condition_{scenario_id}",
            "hardware_affected_component": affected_component,
        })
    return Scenario(
        scenario_id=scenario_id,
        pair_memberships=pair_memberships,
        ground_truth_cause=cause,
        ground_truth_class=ground_truth_class,
        expected_category=ground_truth_class,
        affected_component=affected_component,
        expected_detection=expected_detection,
        localization_targets=targets,
        config=ExperimentConfig(**config_values),
    )


def build_scenarios() -> list[Scenario]:
    """Return nine fixed scenarios with independently assigned truth labels."""
    healthy = _profile()

    readout_q0 = _profile()
    readout_q0["qubits"]["0"]["readout_error"] = 0.4
    readout_q1 = _profile()
    readout_q1["qubits"]["1"]["readout_error"] = 0.4

    gate_noise = _profile()
    gate_noise["gates"] = [{"gate": "x", "qubits": [0], "error": 0.4, "duration": 0.0}]

    coherence_healthy = _healthy_coherence_profile()
    coherence_degraded = copy.deepcopy(coherence_healthy)
    coherence_degraded["qubits"]["0"].update({"t1": 2.0, "t2": 1.0})

    faulty_with_readout = copy.deepcopy(readout_q0)

    return [
        _make_scenario(
            "healthy_ideal", hardware=healthy, cause="healthy_program_ideal_hardware",
            ground_truth_class=NORMAL_CLASS, expected_detection=False,
            pair_memberships=(
                ("healthy_vs_readout_q0", "readout_error_on_qubit_0"),
                ("healthy_vs_gate_noise", "x_gate_error_on_qubit_0"),
                ("healthy_vs_program_mutation", "observed_circuit_mutation"),
            ),
        ),
        _make_scenario(
            "readout_noise_q0", hardware=readout_q0,
            pair_memberships=(
                ("healthy_vs_readout_q0", "readout_error_on_qubit_0"),
                ("readout_location_q0_vs_q1", "affected_readout_qubit"),
            ), cause="readout_degradation",
            ground_truth_class=HARDWARE_CLASS, affected_component="readout qubit 0",
            expected_detection=True,
            targets=({"kind": "hardware", "component_type": "qubit", "component": 0},),
        ),
        _make_scenario(
            "readout_noise_q1", hardware=readout_q1,
            pair_memberships=(("readout_location_q0_vs_q1", "affected_readout_qubit"),),
            cause="readout_degradation",
            ground_truth_class=HARDWARE_CLASS, affected_component="readout qubit 1",
            expected_detection=True,
            targets=({"kind": "hardware", "component_type": "qubit", "component": 1},),
        ),
        _make_scenario(
            "gate_noise_x_q0", hardware=gate_noise,
            pair_memberships=(("healthy_vs_gate_noise", "x_gate_error_on_qubit_0"),),
            cause="gate_calibration_degradation",
            ground_truth_class=HARDWARE_CLASS, affected_component="x gate on qubit 0",
            expected_detection=True,
            targets=({"kind": "hardware", "component_type": "gate", "component": [0], "gate": "x"},),
        ),
        _make_scenario(
            "coherence_healthy_control", circuit=COHERENCE_CIRCUIT, hardware=coherence_healthy,
            pair_memberships=(("healthy_vs_coherence_degradation", "t1_t2_on_qubit_0"),),
            cause="healthy_program_ideal_hardware", ground_truth_class=NORMAL_CLASS,
            expected_detection=False,
        ),
        _make_scenario(
            "coherence_degradation_q0", circuit=COHERENCE_CIRCUIT, hardware=coherence_degraded,
            pair_memberships=(("healthy_vs_coherence_degradation", "t1_t2_on_qubit_0"),),
            cause="coherence_degradation", ground_truth_class=HARDWARE_CLASS,
            affected_component="coherence qubit 0", expected_detection=True,
            targets=({"kind": "hardware", "component_type": "qubit", "component": 0},),
        ),
        _make_scenario(
            "program_fault_ideal", observed_circuit=PROGRAM_MUTATION, hardware=healthy,
            pair_memberships=(
                ("healthy_vs_program_mutation", "observed_circuit_mutation"),
                ("program_fault_ideal_vs_plus_hardware", "readout_error_on_qubit_0"),
            ),
            cause="program_fault_remove_x_q0", ground_truth_class=PROGRAM_CLASS,
            affected_component="remove x operation at index 0", expected_detection=True,
            targets=({"kind": "program", "index": 0},),
        ),
        _make_scenario(
            "program_fault_plus_readout_noise", observed_circuit=PROGRAM_MUTATION,
            hardware=faulty_with_readout,
            pair_memberships=(("program_fault_ideal_vs_plus_hardware", "readout_error_on_qubit_0"),),
            cause="overlapping_program_and_hardware_faults",
            ground_truth_class=UNCERTAIN_CLASS, affected_component="remove x at index 0 plus readout qubit 0",
            expected_detection=True,
            targets=(
                {"kind": "program", "index": 0},
                {"kind": "hardware", "component_type": "qubit", "component": 0},
            ),
        ),
        _make_scenario(
            "deviation_without_cause_evidence", observed_counts={"00": 256, "10": 256},
            hardware=None, cause="execution_deviation_with_calibration_and_program_evidence_withheld",
            ground_truth_class=UNCERTAIN_CLASS, affected_component=None, expected_detection=True,
        ),
    ]


def _diagnosis_class(result: Any) -> str:
    if result.status != "COMPLETED" or result.diagnosis is None:
        error = " ".join(result.errors).lower()
        return "UNSUPPORTED" if "unsupported operation" in error else "FAILED"
    category = getattr(result.diagnosis.category, "value", result.diagnosis.category)
    if category == DiagnosisCategory.NO_ANOMALY.value:
        return NORMAL_CLASS
    return category


def _json_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _json_plain(asdict(value))
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, SimpleNamespace):
        return _json_plain(vars(value))
    return value


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _target_found(target: dict[str, Any], diagnosis: Any) -> bool:
    if diagnosis is None:
        return False
    if target["kind"] == "program":
        candidates = diagnosis.program_evidence_candidates
        return any(candidate.index == target["index"] for candidate in candidates)
    for candidate in diagnosis.hardware_evidence_candidates:
        if candidate.component_type != target["component_type"]:
            continue
        component = _json_plain(candidate.component)
        if component != target["component"]:
            continue
        if target.get("gate") and f"gate calibration for {target['gate']}" not in candidate.reason.lower():
            continue
        return True
    return False


def case_record(scenario: Scenario, result: Any) -> dict[str, Any]:
    """Build an audit row while taking truth only from the scenario definition."""
    diagnosis = result.diagnosis
    predicted = _diagnosis_class(result)
    found = [_target_found(target, diagnosis) for target in scenario.localization_targets]
    localization_correct = all(found) if scenario.localization_targets else None
    statistics = diagnosis.statistics_result if diagnosis is not None else None
    return {
        "scenario_id": scenario.scenario_id,
        "pair_memberships": [{"pair_id": pair, "changed_factor": factor} for pair, factor in scenario.pair_memberships],
        "ground_truth_cause": scenario.ground_truth_cause,
        "ground_truth_class": scenario.ground_truth_class,
        "predicted_class": predicted,
        "status": result.status,
        "expected_category": scenario.expected_category,
        "category_correct": predicted == scenario.expected_category,
        "expected_detection": scenario.expected_detection,
        "anomaly_detected": diagnosis.anomaly_detected if diagnosis is not None else None,
        "affected_component": scenario.affected_component,
        "localization_targets": _json_plain(list(scenario.localization_targets)),
        "localized_targets_found": found,
        "localization_supported": bool(scenario.localization_targets),
        "localization_correct": localization_correct,
        "abstained": predicted == UNCERTAIN_CLASS,
        "abstention_reason": diagnosis.summary if diagnosis is not None and predicted == UNCERTAIN_CLASS else None,
        "program_candidates": _json_plain(diagnosis.program_evidence_candidates) if diagnosis is not None else [],
        "hardware_candidates": _json_plain(diagnosis.hardware_evidence_candidates) if diagnosis is not None else [],
        "evidence": list(diagnosis.evidence) if diagnosis is not None else [],
        "missing_evidence": list(diagnosis.missing_evidence) if diagnosis is not None else [],
        "observable_metrics": {
            "total_variation_distance": statistics.total_variation_distance if statistics is not None else None,
            "js_divergence": statistics.js_divergence if statistics is not None else None,
            "hellinger_distance": statistics.hellinger_distance if statistics is not None else None,
            "threshold": statistics.threshold if statistics is not None else None,
            "finite_shot_guard": statistics.finite_shot_guard if statistics is not None else None,
        },
        "settings": {
            "shots": scenario.config.shots,
            "seed": scenario.config.random_seed,
            "num_qubits": scenario.config.num_qubits,
            "anomaly_threshold": scenario.config.anomaly_threshold,
            "circuit": scenario.config.circuit,
            "observed_circuit": scenario.config.observed_circuit or scenario.config.circuit,
            "hardware_parameters": scenario.config.hardware_parameters,
            "observation_source": result.execution_metadata.get("observation_source"),
            "hardware_model_applied": result.execution_metadata.get("hardware_model_applied"),
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate separate detection, classification and localization summaries."""
    classes = [NORMAL_CLASS, PROGRAM_CLASS, HARDWARE_CLASS, UNCERTAIN_CLASS]
    predicted_classes = [*classes, "FAILED", "UNSUPPORTED", "OTHER"]
    confusion = {truth: {pred: 0 for pred in predicted_classes} for truth in classes}
    for record in records:
        truth, prediction = record["ground_truth_class"], record["predicted_class"]
        if truth in confusion:
            confusion[truth][prediction if prediction in predicted_classes else "OTHER"] += 1
    class_metrics: dict[str, dict[str, float | int]] = {}
    for label in classes:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[row][label] for row in classes)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    positives = [row for row in records if row["expected_detection"] is True]
    negatives = [row for row in records if row["expected_detection"] is False]
    detected = sum(row["anomaly_detected"] is True for row in positives)
    false_alarms = sum(row["anomaly_detected"] is True for row in negatives)
    localizable = [row for row in records if row["localization_supported"]]
    return {
        "case_count": len(records),
        "status_counts": {key: sum(row["status"] == key for row in records) for key in ("COMPLETED", "FAILED", "UNSUPPORTED")},
        "ground_truth_counts": {label: sum(row["ground_truth_class"] == label for row in records) for label in classes},
        "classification": {
            "confusion_matrix_rows_truth_columns_prediction": confusion,
            "per_class": class_metrics,
            "accuracy": sum(row["category_correct"] for row in records) / len(records) if records else 0.0,
            "balanced_accuracy": sum(row["recall"] for row in class_metrics.values()) / len(classes),
            "macro_f1": sum(row["f1"] for row in class_metrics.values()) / len(classes),
            "abstention_count": sum(row["abstained"] for row in records),
            "failure_count": sum(row["predicted_class"] == "FAILED" for row in records),
            "unsupported_count": sum(row["predicted_class"] == "UNSUPPORTED" for row in records),
        },
        "detection": {
            "expected_positive_cases": len(positives),
            "detected_positive_cases": detected,
            "detection_rate": detected / len(positives) if positives else None,
            "expected_negative_cases": len(negatives),
            "false_alarms": false_alarms,
            "false_alarm_rate": false_alarms / len(negatives) if negatives else None,
        },
        "localization": {
            "cases_with_expected_targets": len(localizable),
            "cases_with_all_targets_found": sum(row["localization_correct"] is True for row in localizable),
            "correct_localization_rate": (
                sum(row["localization_correct"] is True for row in localizable) / len(localizable)
                if localizable else None
            ),
        },
        "case_outcomes": records,
    }


def run_experiment_set(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run this fixed set once; refuse to overwrite any existing destination."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output_dir}")
    scenarios = build_scenarios()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for scenario in scenarios:
        result_path = output_dir / "cases" / f"{scenario.scenario_id}.json"
        result = run_experiment(scenario.config, output_path=result_path)
        records.append(case_record(scenario, result))
    report = summarize(records)
    _write_aggregate_outputs(output_dir, scenarios, report)
    return report


def _write_aggregate_outputs(output_dir: Path, scenarios: list[Scenario], report: dict[str, Any]) -> None:
    """Write aggregate outputs only when none of their destinations exist."""
    targets = [output_dir / "phase1_results.json", output_dir / "case_summary.csv", output_dir / "REPORT.md"]
    conflicts = [str(path) for path in targets if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite aggregate output(s): {conflicts}")
    records = report["case_outcomes"]
    report["protocol"] = {
        "name": "controlled_fault_isolation_phase1_v1",
        "execution": "local simulator via experiments.run_experiment.run_experiment",
        "shots": SHOTS,
        "seed": SEED,
        "anomaly_threshold": 0.05,
        "ground_truth_source": "predeclared scenario cause/component in build_scenarios; never inferred from diagnosis output",
        "physical_hardware_validation": False,
        "matched_pairs": sorted({pair for scenario in scenarios for pair, _ in scenario.pair_memberships}),
        "case_count": len(scenarios),
    }
    (output_dir / "phase1_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "scenario_id", "pair_memberships", "ground_truth_cause", "ground_truth_class",
            "predicted_class", "status", "category_correct", "expected_detection", "anomaly_detected",
            "affected_component", "localization_supported", "localization_correct", "abstained",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            **{key: row.get(key) for key in fields if key != "pair_memberships"},
            "pair_memberships": json.dumps(row["pair_memberships"], sort_keys=True),
        } for row in records)
    (output_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")


def finalize_existing_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Build missing aggregates from saved run reports without executing circuits."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"No existing experiment output directory: {output_dir}")
    scenarios = build_scenarios()
    case_dir = output_dir / "cases"
    records = []
    for scenario in scenarios:
        result_path = case_dir / f"{scenario.scenario_id}.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing saved case result: {result_path}")
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        diagnosis = _namespace(saved.get("diagnosis")) if saved.get("diagnosis") is not None else None
        result = SimpleNamespace(
            status=saved.get("status"),
            errors=saved.get("errors", []),
            execution_metadata=saved.get("execution_metadata", {}),
            diagnosis=diagnosis,
        )
        records.append(case_record(scenario, result))
    report = summarize(records)
    _write_aggregate_outputs(output_dir, scenarios, report)
    return report


def render_report(report: dict[str, Any]) -> str:
    classification = report["classification"]
    detection = report["detection"]
    localization = report["localization"]
    lines = [
        "# Controlled Fault-Isolation Validation — Phase 1",
        "",
        "## Setup and ground truth",
        "",
        "This bounded study uses the production local execution and diagnosis path (`experiments.run_experiment.run_experiment`) with 512 shots, seed 20261013, and anomaly threshold 0.05. The scenario table declares ground-truth cause and expected category before execution; diagnosis outputs do not assign truth. Hardware profiles are simulated calibration inputs. This is not physical-hardware validation.",
        "",
        "Nine scenarios cover a healthy ideal control, readout noise on each of two measured qubits, X-gate error, a healthy/degraded coherence pair using a calibrated idle duration, a program mutation, combined program plus readout faults, and an observed-distribution deviation with cause evidence withheld. Paired cases share circuit/reference, seed, shot count, and all non-target settings as applicable. The q0/q1 readout pair changes only the affected location.",
        "",
        "The uncertain observation is explicitly a controlled counts fixture with no accompanying hardware profile or program mismatch. The combined-fault case has both mutation and readout calibration evidence and is expected to remain ambiguous.",
        "",
        "## Results",
        "",
        f"- Cases: {report['case_count']} total; status counts: {report['status_counts']}",
        f"- Ground-truth counts: {report['ground_truth_counts']}",
        f"- Detection: {detection['detected_positive_cases']}/{detection['expected_positive_cases']} injected-deviation cases detected ({_percent(detection['detection_rate'])}); false alarms {detection['false_alarms']}/{detection['expected_negative_cases']} ({_percent(detection['false_alarm_rate'])}).",
        f"- Classification: accuracy {classification['accuracy']:.3f}, balanced accuracy {classification['balanced_accuracy']:.3f}, macro-F1 {classification['macro_f1']:.3f}; abstentions {classification['abstention_count']}; failures {classification['failure_count']}; unsupported {classification['unsupported_count']}.",
        f"- Localization: {localization['cases_with_all_targets_found']}/{localization['cases_with_expected_targets']} cases with all declared targets found ({_percent(localization['correct_localization_rate'])}).",
        "",
        "### Per-class classification",
        "",
        "| Ground-truth class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in classification["per_class"].items():
        lines.append(f"| {label} | {metrics['support']} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    lines.extend([
        "",
        "### Per-case outcomes",
        "",
        "| Scenario | Ground truth cause | Expected class | Predicted class | Detected | Localized as expected |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in report["case_outcomes"]:
        localized = "n/a" if row["localization_correct"] is None else str(row["localization_correct"])
        lines.append(f"| {row['scenario_id']} | {row['ground_truth_cause']} | {row['ground_truth_class']} | {row['predicted_class']} | {row['anomaly_detected']} | {localized} |")
    lines.extend([
        "",
        "### Confusion matrix",
        "",
        "Rows are ground truth and columns are predicted class. Failures and unsupported runs remain separate columns.",
        "",
        "| Truth \\ Prediction | NORMAL | PROGRAM_FAULT | HARDWARE_ANOMALY | AMBIGUOUS / INSUFFICIENT_EVIDENCE | FAILED | UNSUPPORTED | OTHER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for truth, values in classification["confusion_matrix_rows_truth_columns_prediction"].items():
        lines.append(f"| {truth} | " + " | ".join(str(values[label]) for label in (NORMAL_CLASS, PROGRAM_CLASS, HARDWARE_CLASS, UNCERTAIN_CLASS, "FAILED", "UNSUPPORTED", "OTHER")) + " |")
    lines.extend([
        "",
        "## Control failures",
        "",
    ])
    control_misses = [
        row for row in report["case_outcomes"]
        if row["expected_detection"] is False and not row["category_correct"]
    ]
    if control_misses:
        for row in control_misses:
            reasons = [item.get("reason", "") for item in row["hardware_candidates"]]
            evidence = "; ".join(reasons) if reasons else "no retained hardware candidate reason"
            lines.append(
                f"- **False positive: `{row['scenario_id']}`.** The expected class was "
                f"{row['ground_truth_class']}, predicted {row['predicted_class']}, while "
                f"`anomaly_detected` was {row['anomaly_detected']}. Hardware candidate evidence: {evidence}."
            )
    else:
        lines.append("No negative-control classification errors were observed.")
    lines.extend([
        "",
        "## Failures, interpretation, and limitations",
        "",
        "Any failed or unsupported case is retained separately and is not counted as a successful diagnosis. Ambiguous outcomes are abstentions, not ordinary correct fault labels unless the predeclared composite/unknown scenario expects uncertainty. Detection (distribution/direct anomaly status), diagnosis class, and component localization are reported separately.",
        "",
        "This is a small, deliberately constructed smoke/validation set, not a representative accuracy estimate. It uses one two-qubit circuit family, a handful of strong injected profiles, one fixed seed, and only 512 shots. The combined-fault result tests the existing engine's evidence boundary; the counts fixture tests abstention when causal evidence is missing. Results apply only to this local simulator configuration. They do not establish physical-hardware performance or calibration fidelity.",
        "",
        "The noise-free coherence control is a concrete false positive: its calibrated identity gate has duration 2.0, and `src/quantum_simulator/hardware_fault_localizer.py::_score_gate` treats durations above 1.0 as degraded hardware evidence. The observed distribution matched the reference (`anomaly_detected=false`), but the rule engine returned HARDWARE_ANOMALY based on that duration. The matched coherence pair still changes only T1/T2; this result shows the current duration heuristic can dominate that healthy control.",
        "",
        "## Recommended next experiment",
        "",
        "Expand to multiple independently designed circuit families and several calibrated noise levels per mechanism. Pair every intervention to a same-seed, same-shot healthy control; vary affected qubits and gates while ensuring each is measured and causally observable. Add repeated seeds to quantify shot variation, and keep all variants of a circuit family together in any evaluation split. Predeclare which causes should be identifiable from calibration evidence and which must yield uncertainty. If physical claims are desired, validate separately against independently documented hardware runs.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `phase1_results.json`: complete settings, evidence, localization candidates, per-case predictions, metrics and confusion matrix.",
        "- `case_summary.csv`: compact case-level result table.",
        "- `cases/*.json`: full existing experiment-runner result for each scenario.",
        "- `experiments/fault_isolation_phase1.py`: fixed scenario definitions and output builder.",
    ])
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--finalize-existing", action="store_true", help="build missing summaries from saved case JSON without rerunning")
    args = parser.parse_args()
    report = finalize_existing_experiment(args.output_dir) if args.finalize_existing else run_experiment_set(args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "case_count": report["case_count"],
        "classification_macro_f1": report["classification"]["macro_f1"],
        "detection": report["detection"],
        "localization": report["localization"],
        "status_counts": report["status_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
