"""Independent circuit/fault-scenario evaluation for the local diagnosis path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.fault_isolation_phase1 import (
    HARDWARE_CLASS,
    NORMAL_CLASS,
    PROGRAM_CLASS,
    UNCERTAIN_CLASS,
    _diagnosis_class,
    _json_plain,
    _target_found,
    case_record,
)
from experiments.run_experiment import run_experiment
from experiment_config import ExperimentConfig
from quantum_simulator.diagnosis_engine import DiagnosisCategory


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "experiments" / "independent_generalization_phase2"
PROTOCOL_VERSION = "independent_generalization_phase2_v1"
SHOTS = 512
SEEDS = (20262001, 20262002, 20262003)
NUM_QUBITS = 4
CLASS_LABELS = (NORMAL_CLASS, PROGRAM_CLASS, HARDWARE_CLASS, UNCERTAIN_CLASS)
PREDICTION_LABELS = (*CLASS_LABELS, "FAILED", "UNSUPPORTED", "OTHER")


@dataclass(frozen=True)
class Phase2Scenario:
    scenario_id: str
    circuit_family_id: str
    ground_truth_cause: str
    ground_truth_class: str
    expected_category: str
    affected_component: str | None
    expected_detection: bool | None
    localization_targets: tuple[dict[str, Any], ...]
    novelty_basis: str
    pair_id: str | None
    config: ExperimentConfig

    @property
    def pair_memberships(self) -> tuple[tuple[str, str], ...]:
        if self.pair_id is None:
            return ()
        return ((self.pair_id, self.ground_truth_cause),)


def _calibration_name(name: str) -> str:
    return {"cnot": "cx", "id": "i", "identity": "i"}.get(name, name)


def _operation_qubits(operation: tuple[Any, ...]) -> tuple[int, ...]:
    name = str(operation[0]).lower()
    if name in {"measure", "barrier", "reset"}:
        return ()
    if name in {"rx", "ry", "rz", "p", "u1", "u2", "u3"}:
        return (int(operation[1]),)
    if name in {"cx", "cnot", "cz", "swap", "ch", "rxx", "rzz", "cp", "crx", "cry", "crz"}:
        return int(operation[1]), int(operation[2])
    if name in {"ccx", "rccx", "toffoli", "cswap", "fredkin"}:
        return int(operation[1]), int(operation[2]), int(operation[3])
    return (int(operation[1]),) if len(operation) > 1 else ()


def _profile(
    circuits: tuple[list[tuple[Any, ...]], ...],
    *,
    readout_errors: dict[int, float] | None = None,
    gate_errors: dict[tuple[str, tuple[int, ...]], float] | None = None,
    edge_errors: dict[tuple[int, int], float] | None = None,
) -> dict[str, Any]:
    readout_errors = readout_errors or {}
    gate_errors = gate_errors or {}
    edge_errors = edge_errors or {}
    gate_entries: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    edge_pairs: set[tuple[int, int]] = set()
    for circuit in circuits:
        for operation in circuit:
            name = str(operation[0]).lower()
            if name in {"measure", "barrier", "reset"}:
                continue
            qubits = _operation_qubits(operation)
            calibration_name = _calibration_name(name)
            gate_entries[(calibration_name, qubits)] = {
                "gate": calibration_name,
                "qubits": list(qubits),
                "error": gate_errors.get((calibration_name, qubits), 0.0),
                "duration": 0.1,
            }
            if len(qubits) == 2:
                edge_pairs.add(tuple(sorted(qubits)))
    edge_pairs.update(tuple(sorted(pair)) for pair in edge_errors)
    return {
        "coupling_map": [list(pair) for pair in sorted(edge_pairs)],
        "qubits": {
            str(qubit): {
                "t1": 1e6,
                "t2": 1e6,
                "readout_error": readout_errors.get(qubit, 0.0),
            }
            for qubit in range(NUM_QUBITS)
        },
        "gates": list(gate_entries.values()),
        "edges": [
            {"qubits": list(tuple(sorted(pair))), "error": error, "duration": 0.1}
            for pair, error in sorted(edge_errors.items())
        ],
    }


def _scenario(
    scenario_id: str,
    circuit: list[tuple[Any, ...]],
    *,
    observed_circuit: list[tuple[Any, ...]] | None = None,
    hardware: dict[str, Any] | None,
    cause: str,
    truth: str,
    affected: str | None = None,
    targets: tuple[dict[str, Any], ...] = (),
    novelty: str,
    family: str,
    expected_detection: bool | None,
    observed_counts: dict[str, int] | None = None,
    fault_index: int | None = None,
    fault_qubits: list[int] | None = None,
    pair_id: str | None = None,
) -> Phase2Scenario:
    program_fault = truth == PROGRAM_CLASS
    hardware_fault = truth == HARDWARE_CLASS
    uncertain = truth == UNCERTAIN_CLASS
    config = ExperimentConfig(
        experiment_name=f"phase2_{scenario_id}",
        num_qubits=NUM_QUBITS,
        shots=SHOTS,
        random_seed=SEEDS[0],
        circuit=circuit,
        observed_circuit=observed_circuit,
        hardware_parameters=hardware,
        observed_counts=observed_counts,
        anomaly_threshold=0.05,
        ground_truth_label=truth if truth in {NORMAL_CLASS, PROGRAM_CLASS, HARDWARE_CLASS} else None,
        circuit_family_id=family,
        program_fault_id=f"phase2_{scenario_id}_mutation" if program_fault else None,
        program_fault_description=f"Predeclared Phase 2 program mutation for {scenario_id}." if program_fault else None,
        program_fault_operation_index=fault_index if program_fault else None,
        program_fault_qubits=fault_qubits if program_fault else None,
        hardware_condition_id=f"phase2_{scenario_id}_condition" if hardware_fault else None,
        hardware_affected_component=affected if hardware_fault else None,
        description=f"Phase 2 scenario: {cause}",
    )
    if uncertain and (truth != UNCERTAIN_CLASS or expected_detection is None):
        raise ValueError("uncertain controls require an explicit uncertainty truth and detection expectation")
    return Phase2Scenario(
        scenario_id=scenario_id,
        circuit_family_id=family,
        ground_truth_cause=cause,
        ground_truth_class=truth,
        expected_category=truth,
        affected_component=affected,
        expected_detection=expected_detection,
        localization_targets=targets,
        novelty_basis=novelty,
        pair_id=pair_id,
        config=config,
    )


def build_scenarios() -> list[Phase2Scenario]:
    """Declare twelve new scenarios independently of any diagnosis output."""
    bell = [("h", 2), ("cnot", 2, 3), ("measure", [2, 3])]
    phase = [
        ("ry", 0, 0.6), ("rx", 2, 0.4), ("rzz", 0, 2, 0.7),
        ("h", 0), ("measure", [0, 2]),
    ]
    phase_without_rzz = [operation for index, operation in enumerate(phase) if index != 2]
    rotation_expected = [
        ("h", 0), ("x", 1), ("ry", 2, math.pi), ("measure", [0, 1, 2]),
    ]
    rotation_mutated = [
        ("h", 0), ("x", 1), ("rz", 2, math.pi), ("measure", [0, 1, 2]),
    ]
    pair_id = "program_ry_to_rz_vs_q2_readout_inversion"
    readout_circuit = [("h", 1), ("cnot", 1, 3), ("measure", [1, 3])]
    coupler_circuit = [("h", 1), ("cnot", 1, 3), ("measure", [1, 3])]
    edge_circuit = [("h", 1), ("cnot", 1, 3), ("measure", [1, 3])]
    unknown_circuit = [("ry", 3, math.pi / 2), ("measure", [3])]

    scenarios = [
        _scenario(
            "healthy_bell_q2_q3", bell, hardware=_profile((bell,)),
            cause="healthy four-register Bell preparation and measurement", truth=NORMAL_CLASS,
            novelty="Four-qubit register with entanglement on q2-q3; Phase 1 used a two-qubit X/measurement structure.",
            family="phase2_bell_q2_q3", expected_detection=False,
        ),
        _scenario(
            "healthy_rzz_phase_depth", phase, hardware=_profile((phase,)),
            cause="healthy rotation and RZZ phase sequence", truth=NORMAL_CLASS,
            novelty="New RX/RY/RZZ sequence, operation depth, and q0-q2 interaction absent from Phases 1/1.3/1.4.",
            family="phase2_rzz_phase_q0_q2", expected_detection=False,
        ),
        _scenario(
            "program_fault_ry_to_rz_q2", rotation_expected, observed_circuit=rotation_mutated,
            hardware=_profile((rotation_expected, rotation_mutated)),
            cause="program mutation replaces RY with RZ at operation index 2", truth=PROGRAM_CLASS,
            affected="replace RY with RZ at index 2 on q2", targets=({"kind": "program", "index": 2},),
            novelty="Parameterized single-qubit gate substitution at operation index 2 on q2; Phase 1 removed X at index 0 on q0.",
            family="phase2_rotation_readout_q2", expected_detection=True, fault_index=2,
            fault_qubits=[2], pair_id=pair_id,
        ),
        _scenario(
            "hardware_readout_inversion_q2", rotation_expected,
            hardware=_profile((rotation_expected,), readout_errors={2: 1.0}),
            cause="complete readout inversion on q2", truth=HARDWARE_CLASS,
            affected="readout q2", targets=({"kind": "hardware", "component_type": "qubit", "component": 2},),
            novelty="New q2 readout location and a maximal readout error; its observed distribution is paired with the RY-to-RZ program fault.",
            family="phase2_rotation_readout_q2", expected_detection=True, pair_id=pair_id,
        ),
        _scenario(
            "program_fault_remove_rzz_q0_q2", phase, observed_circuit=phase_without_rzz,
            hardware=_profile((phase, phase_without_rzz)),
            cause="program mutation removes RZZ at operation index 2", truth=PROGRAM_CLASS,
            affected="remove RZZ at index 2 on q0-q2", targets=({"kind": "program", "index": 2},),
            novelty="Two-qubit phase-gate deletion at index 2 in a rotation/entanglement circuit; no prior Phase 1 program fault involved RZZ.",
            family="phase2_rzz_phase_q0_q2", expected_detection=True, fault_index=2,
            fault_qubits=[0, 2],
        ),
        _scenario(
            "hardware_readout_q3_moderate", readout_circuit,
            hardware=_profile((readout_circuit,), readout_errors={3: 0.20}),
            cause="readout degradation at q3 with 0.20 error", truth=HARDWARE_CLASS,
            affected="readout q3", targets=({"kind": "hardware", "component_type": "qubit", "component": 3},),
            novelty="New q3 location measured in a q1-q3 entangled circuit; Phase 1 readout faults were q0/q1 at strength 0.40.",
            family="phase2_cx_q1_q3_readout", expected_detection=True,
        ),
        _scenario(
            "hardware_readout_q3_near_threshold", readout_circuit,
            hardware=_profile((readout_circuit,), readout_errors={3: 0.055}),
            cause="subtle q3 readout degradation near the 0.05 decision threshold", truth=HARDWARE_CLASS,
            affected="readout q3", targets=({"kind": "hardware", "component_type": "qubit", "component": 3},),
            novelty="Readout error 0.055 is near the configured 0.05 anomaly threshold and is evaluated across three seeds.",
            family="phase2_cx_q1_q3_readout", expected_detection=True,
        ),
        _scenario(
            "hardware_readout_q3_subthreshold", readout_circuit,
            hardware=_profile((readout_circuit,), readout_errors={3: 0.04}),
            cause="known q3 readout perturbation below the 0.05 calibration-evidence threshold", truth=HARDWARE_CLASS,
            affected="readout q3", targets=({"kind": "hardware", "component_type": "qubit", "component": 3},),
            novelty="Readout error 0.04 is below both the nominal detection threshold and localizer readout evidence cutoff; an abstention/miss is informative.",
            family="phase2_cx_q1_q3_readout", expected_detection=True,
        ),
        _scenario(
            "hardware_rzz_error_near_threshold", phase,
            hardware=_profile((phase,), gate_errors={("rzz", (0, 2)): 0.055}),
            cause="RZZ calibration error 0.055 near the decision threshold", truth=HARDWARE_CLASS,
            affected="RZZ q0-q2", targets=({"kind": "hardware", "component_type": "gate", "component": [0, 2], "gate": "rzz"},),
            novelty="Non-Clifford entangling phase gate with a subtle calibrated operation failure probability absent from Phase 1.",
            family="phase2_rzz_phase_q0_q2", expected_detection=True,
        ),
        _scenario(
            "hardware_cx_coupler_q1_q3", coupler_circuit,
            hardware=_profile(
                (coupler_circuit,), gate_errors={("cx", (1, 3)): 0.18},
                edge_errors={(1, 3): 0.18},
            ),
            cause="CX failure on a new q1-q3 coupling, with matching edge calibration evidence", truth=HARDWARE_CLASS,
            affected="coupling edge q1-q3", targets=({"kind": "hardware", "component_type": "edge", "component": [1, 3]},),
            novelty="New non-adjacent q1-q3 coupling. The simulator applies the CX gate-error probability; edge metadata supplies the edge localization evidence.",
            family="phase2_cx_edge_q1_q3", expected_detection=True,
        ),
        _scenario(
            "overlap_ry_to_rz_plus_readout_q2", rotation_expected,
            observed_circuit=rotation_mutated,
            hardware=_profile((rotation_expected, rotation_mutated), readout_errors={2: 0.20}),
            cause="RY-to-RZ program mutation combined with q2 readout degradation", truth=UNCERTAIN_CLASS,
            affected="program operation 2 plus readout q2",
            targets=(
                {"kind": "program", "index": 2},
                {"kind": "hardware", "component_type": "qubit", "component": 2},
            ),
            novelty="Combined gate-substitution and readout effect; the evidence should preserve both causes and avoid a forced single-cause class.",
            family="phase2_rotation_readout_q2", expected_detection=True,
        ),
        _scenario(
            "cause_unknown_phase_observation", unknown_circuit,
            hardware=None, observed_counts={"0": SHOTS},
            cause="observed deviation with no hardware calibration or program mismatch supplied", truth=UNCERTAIN_CLASS,
            affected=None, novelty="New q3 rotation measurement structure with an explicit observation fixture and causal evidence withheld.",
            family="phase2_unknown_q3_rotation", expected_detection=True,
        ),
    ]
    return scenarios


def _case_outcome(scenario: Phase2Scenario, result: Any, seed: int) -> dict[str, Any]:
    row = case_record(scenario, result)
    row["evaluation_seed"] = seed
    row["settings"]["seed"] = seed
    row["circuit_family_id"] = scenario.circuit_family_id
    row["novelty_basis"] = scenario.novelty_basis
    row["pair_id"] = scenario.pair_id
    row["execution_disposition"] = (
        "UNSUPPORTED" if row["predicted_class"] == "UNSUPPORTED"
        else "FAILED" if result.status != "COMPLETED"
        else "COMPLETED"
    )
    if not scenario.localization_targets:
        localization_outcome = "NOT_APPLICABLE"
    elif result.status == "FAILED":
        localization_outcome = "FAILED"
    elif row["predicted_class"] == "UNSUPPORTED":
        localization_outcome = "UNSUPPORTED"
    elif row["localization_correct"] is True:
        localization_outcome = "CORRECT"
    else:
        diagnosis = result.diagnosis
        supported_candidates = (
            list(getattr(diagnosis, "program_evidence_candidates", []))
            + list(getattr(diagnosis, "hardware_evidence_candidates", []))
        ) if diagnosis is not None else []
        localization_outcome = "INCORRECT" if supported_candidates else "NOT_JUSTIFIED"
    row["localization_outcome"] = localization_outcome
    return row


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {truth: {prediction: 0 for prediction in PREDICTION_LABELS} for truth in CLASS_LABELS}
    for row in records:
        truth = row["ground_truth_class"]
        prediction = row["predicted_class"]
        if truth in confusion:
            confusion[truth][prediction if prediction in PREDICTION_LABELS else "OTHER"] += 1
    per_class = {}
    for label in CLASS_LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in CLASS_LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    positives = [row for row in records if row["expected_detection"] is True]
    negatives = [row for row in records if row["expected_detection"] is False]
    localizable = [row for row in records if row["localization_targets"]]
    localization_counts = {
        outcome.lower(): sum(row["localization_outcome"] == outcome for row in localizable)
        for outcome in ("CORRECT", "INCORRECT", "NOT_JUSTIFIED", "FAILED", "UNSUPPORTED")
    }
    accuracy = sum(row["category_correct"] for row in records) / len(records) if records else 0.0
    return {
        "case_count": len(records),
        "status_counts": {status: sum(row["execution_disposition"] == status for row in records) for status in ("COMPLETED", "FAILED", "UNSUPPORTED")},
        "ground_truth_counts": {label: sum(row["ground_truth_class"] == label for row in records) for label in CLASS_LABELS},
        "classification": {
            "accuracy": accuracy,
            "balanced_accuracy": sum(item["recall"] for item in per_class.values()) / len(CLASS_LABELS),
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(CLASS_LABELS),
            "per_class": per_class,
            "confusion_matrix_rows_truth_columns_prediction": confusion,
            "abstentions": sum(row["predicted_class"] == UNCERTAIN_CLASS for row in records),
            "failure_count": sum(row["predicted_class"] == "FAILED" for row in records),
            "unsupported_count": sum(row["predicted_class"] == "UNSUPPORTED" for row in records),
        },
        "detection": {
            "positive_cases": len(positives),
            "detected_positives": sum(row["anomaly_detected"] is True for row in positives),
            "detection_rate": sum(row["anomaly_detected"] is True for row in positives) / len(positives) if positives else None,
            "negative_cases": len(negatives),
            "false_positives": sum(row["anomaly_detected"] is True for row in negatives),
            "false_positive_rate": sum(row["anomaly_detected"] is True for row in negatives) / len(negatives) if negatives else None,
            "false_negatives": sum(row["anomaly_detected"] is not True for row in positives),
        },
        "localization": {
            "target_case_count": len(localizable),
            "outcomes": localization_counts,
            "correct_rate_over_target_cases": localization_counts["correct"] / len(localizable) if localizable else None,
        },
    }


def _source_provenance() -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    sources = [
        "experiments/fault_isolation_phase2.py",
        "experiments/fault_isolation_phase1.py",
        "experiments/run_experiment.py",
        "src/quantum_simulator/diagnosis_engine.py",
        "src/quantum_simulator/hardware_fault_localizer.py",
    ]
    return {
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_head_available": revision.returncode == 0,
        "working_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in sources
        },
    }


def _scenario_manifest(scenarios: list[Phase2Scenario]) -> list[dict[str, Any]]:
    return [{
        "scenario_id": scenario.scenario_id,
        "circuit_family_id": scenario.circuit_family_id,
        "ground_truth_cause": scenario.ground_truth_cause,
        "ground_truth_class": scenario.ground_truth_class,
        "expected_category": scenario.expected_category,
        "expected_detection": scenario.expected_detection,
        "affected_component": scenario.affected_component,
        "localization_targets": _json_plain(scenario.localization_targets),
        "novelty_basis": scenario.novelty_basis,
        "pair_id": scenario.pair_id,
        "configuration": asdict(scenario.config),
    } for scenario in scenarios]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, overwrite: bool = False) -> None:
    with path.open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], sort_keys=True) if isinstance(row.get(field), (dict, list)) else row.get(field)
                for field in fields
            })


def _write_report(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Independent Generalization and Fault Localization — Phase 2",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`. This report uses only new Phase 2 scenarios; Phases 1, 1.3, and 1.4 are inventory references, not observations included in these metrics.",
        "",
        "## Prior-phase coverage and Phase 2 novelty",
        "",
        "Phases 1, 1.3, and 1.4 used the same two-qubit X/measurement circuit family: readout noise on q0/q1, X-gate error on q0, an X-based program mutation at index 0, and one identity-duration/T1/T2 pair. Phase 1.3 fixed duration-only evidence semantics; Phase 1.4 repeated those same nine scenarios across seeds.",
        "",
        "Phase 2 introduces four-qubit register usage; RX/RY/RZ/RZZ sequences; q2/q3 readout and gate locations; q0-q2 RZZ and q1-q3 CX couplings; program mutations at operation index 2; a program/hardware pair with matched observed distributions; readout strengths 0.04, 0.055, 0.20, and 1.0; a subtle RZZ gate error; and overlapping/unknown-cause cases. These are new scenario structures relative to the prior nine, although several Phase 2 strength variants intentionally share a circuit family.",
        "",
        "The q1-q3 coupling case supplies both a CX gate error (applied by the simulator) and matching coupling-edge calibration metadata (used for edge localization). Edge error metadata alone is not applied as a shot-noise process by the current simulator.",
        "",
        "## Protocol and provenance",
        "",
        f"- Scenarios: {payload['scenario_count']}; seeds: {payload['seeds']}; shots per run: {SHOTS}; total runs: {payload['run_count']}.",
        "- Scenario ground truth and affected components are declared in `scenario_manifest.json` before execution; they do not depend on diagnosis output.",
        "- Each scenario-seed result records circuit family, circuit and observed circuit, hardware/noise settings, observable statistics, evidence, missing evidence, candidates, and localization assessment.",
        f"- Source revision: `{payload['source_provenance']['git_head']}`; working tree dirty: `{payload['source_provenance']['working_tree_dirty']}`. Source hashes are recorded in the manifest.",
        *(["- Aggregate summaries were reconstructed from the saved case JSON files after correcting the per-seed settings field; no circuit was rerun. Rebuild code provenance is recorded in `aggregate_metrics.json`."] if payload.get("aggregate_rebuild_provenance") else []),
        "- All executions use the local simulator. No physical hardware was used.",
        "",
        "## Results",
        "",
        f"- Run status: {metrics['status_counts']}; unsupported and failed runs remain separate from diagnosis errors.",
        f"- Ground-truth counts: {metrics['ground_truth_counts']}.",
        f"- Classification: accuracy {metrics['classification']['accuracy']:.3f}, balanced accuracy {metrics['classification']['balanced_accuracy']:.3f}, macro-F1 {metrics['classification']['macro_f1']:.3f}; abstentions {metrics['classification']['abstentions']}.",
        f"- Detection: {metrics['detection']['detected_positives']}/{metrics['detection']['positive_cases']} positives ({metrics['detection']['detection_rate']:.3f}); false positives {metrics['detection']['false_positives']}/{metrics['detection']['negative_cases']} ({metrics['detection']['false_positive_rate']:.3f}); false negatives {metrics['detection']['false_negatives']}.",
        f"- Localization: {metrics['localization']['outcomes']}; correct rate across target cases {metrics['localization']['correct_rate_over_target_cases']:.3f}.",
        "",
        "### Per-class metrics",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in metrics["classification"]["per_class"].items():
        lines.append(f"| {label} | {item['support']} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    lines.extend([
        "",
        "### Confusion matrix (truth rows, prediction columns)",
        "",
        "| Truth \\ Predicted | " + " | ".join(PREDICTION_LABELS) + " |",
        "|---|" + "---:|" * len(PREDICTION_LABELS),
    ])
    for truth, columns in metrics["classification"]["confusion_matrix_rows_truth_columns_prediction"].items():
        lines.append(f"| {truth} | " + " | ".join(str(columns[label]) for label in PREDICTION_LABELS) + " |")
    lines.extend([
        "",
        "### Per-scenario and per-seed results",
        "",
        "| Scenario | Family | Truth | Prediction | Detection | Localization | Abstain | Seed | TVD |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ])
    for row in payload["case_outcomes"]:
        tvd = row["observable_metrics"]["total_variation_distance"]
        lines.append(
            f"| {row['scenario_id']} | {row['circuit_family_id']} | {row['ground_truth_class']} | {row['predicted_class']} | "
            f"{row['anomaly_detected']} | {row['localization_outcome']} | {row['abstained']} | "
            f"{row['evaluation_seed']} | {tvd if tvd is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "Full source configurations, case outputs, and aggregate metrics are saved in the sibling JSON/CSV files and `cases/` directory.",
        "",
        "## Limitations",
        "",
        "This is a small constructed simulator study over six new circuit families, with three fixed seeds and 512 shots. Strength variants and paired cases share circuits; runs are not independent samples of generalization. Near-threshold outcomes are sensitive to this finite-shot design. The ground truth is scenario-defined, not independently adjudicated. Simulator calibration profiles do not establish physical-hardware calibration fidelity or generalization.",
        "",
    ])
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def _as_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _as_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_namespace(item) for item in value]
    return value


def finalize_existing_phase2(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Rebuild summaries from saved case JSON without re-executing circuits."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"No existing Phase 2 output directory: {output_dir}")
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    records = []
    for seed in SEEDS:
        for scenario_id, scenario in scenarios.items():
            case_path = output_dir / "cases" / f"seed_{seed}" / f"{scenario_id}.json"
            if not case_path.is_file():
                raise FileNotFoundError(f"Missing saved Phase 2 case: {case_path}")
            saved = json.loads(case_path.read_text(encoding="utf-8"))
            result = SimpleNamespace(
                status=saved["status"],
                errors=saved.get("errors", []),
                execution_metadata=saved.get("execution_metadata", {}),
                diagnosis=_as_namespace(saved.get("diagnosis")),
            )
            records.append(_case_outcome(scenario, result, seed))
    metrics = _metrics(records)
    manifest = json.loads((output_dir / "scenario_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_count": len(scenarios),
        "run_count": len(records),
        "seeds": list(SEEDS),
        "shots_per_case": SHOTS,
        "simulator_only": True,
        "source_provenance": manifest["source_provenance"],
        "aggregate_rebuild_provenance": {
            "method": "recomputed case records and aggregates from saved per-case JSON files",
            "source_file": "experiments/fault_isolation_phase2.py::finalize_existing_phase2",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "simulation_rerun": False,
            "corrected_field": "case_outcomes[].settings.seed now matches the recorded evaluation_seed",
        },
        "metrics": metrics,
        "case_outcomes": records,
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["scenario_id", "circuit_family_id", "evaluation_seed", "ground_truth_cause", "ground_truth_class", "expected_category", "predicted_class", "status", "execution_disposition", "category_correct", "expected_detection", "anomaly_detected", "affected_component", "localization_targets", "localized_targets_found", "localization_outcome", "abstained", "abstention_reason", "observable_metrics", "settings", "hardware_candidates", "program_candidates", "evidence", "missing_evidence", "novelty_basis", "pair_id"]
    _write_csv(output_dir / "scenario_seed_results.csv", records, fields, overwrite=True)
    _write_report(output_dir / "REPORT.md", payload, overwrite=True)
    return payload


def run_phase2(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 2 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    scenarios = build_scenarios()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "simulator_only": True,
        "execution": "experiments.run_experiment.run_experiment",
        "shots_per_case": SHOTS,
        "seeds": list(SEEDS),
        "scenario_count": len(scenarios),
        "run_count": len(scenarios) * len(SEEDS),
        "prior_phase_scenario_sources": [
            "experiments/fault_isolation_phase1.py::build_scenarios",
            "experiments/fault_isolation_phase1_4.py::SEEDS",
        ],
        "source_provenance": _source_provenance(),
        "scenarios": _scenario_manifest(scenarios),
    }
    (output_dir / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records = []
    for seed in SEEDS:
        for scenario in scenarios:
            config = ExperimentConfig.from_mapping({
                **asdict(scenario.config),
                "random_seed": seed,
                "experiment_name": f"phase2_{scenario.scenario_id}_seed_{seed}",
            })
            case_path = output_dir / "cases" / f"seed_{seed}" / f"{scenario.scenario_id}.json"
            case_path.parent.mkdir(parents=True, exist_ok=True)
            result = run_experiment(config, output_path=case_path)
            record = _case_outcome(scenario, result, seed)
            if record["ground_truth_class"] != scenario.ground_truth_class:
                raise AssertionError(f"Ground truth changed for {scenario.scenario_id}")
            records.append(record)

    metrics = _metrics(records)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_count": len(scenarios),
        "run_count": len(records),
        "seeds": list(SEEDS),
        "shots_per_case": SHOTS,
        "simulator_only": True,
        "source_provenance": manifest["source_provenance"],
        "metrics": metrics,
        "case_outcomes": records,
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        output_dir / "scenario_seed_results.csv", records,
        ["scenario_id", "circuit_family_id", "evaluation_seed", "ground_truth_cause", "ground_truth_class", "expected_category", "predicted_class", "status", "execution_disposition", "category_correct", "expected_detection", "anomaly_detected", "affected_component", "localization_targets", "localized_targets_found", "localization_outcome", "abstained", "abstention_reason", "observable_metrics", "settings", "hardware_candidates", "program_candidates", "evidence", "missing_evidence", "novelty_basis", "pair_id"],
    )
    _write_report(output_dir / "REPORT.md", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--finalize-existing", action="store_true", help="rebuild summaries from saved case JSON without re-execution")
    args = parser.parse_args()
    payload = finalize_existing_phase2(args.output_dir) if args.finalize_existing else run_phase2(args.output_dir)
    metrics = payload["metrics"]
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "scenario_count": payload["scenario_count"],
        "run_count": payload["run_count"],
        "accuracy": metrics["classification"]["accuracy"],
        "macro_f1": metrics["classification"]["macro_f1"],
        "detection_rate": metrics["detection"]["detection_rate"],
        "localization": metrics["localization"]["outcomes"],
        "abstentions": metrics["classification"]["abstentions"],
        "status_counts": metrics["status_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
