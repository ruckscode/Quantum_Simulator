"""Offline end-to-end demonstration of the quantum diagnosis pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from diagnosis_engine import DiagnosisCategory
from hardware_model import HardwareModel
from quantum_simulator import QuantumSimulator
from validation_framework import ValidationCase, ValidationRecord, run_validation_case


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "final_demo_report.json"
DEMO_SEED = 1729
DEMO_SHOTS = 1000


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _ideal_distribution(circuit: Sequence[Any], n_qubits: int) -> dict[str, float]:
    """Return exact statevector probabilities; measurement instructions are deferred."""

    simulator = QuantumSimulator(n_qubits, seed=DEMO_SEED)
    unitary_operations = [
        operation for operation in circuit
        if not isinstance(operation, (tuple, list)) or not operation or str(operation[0]).lower() != "measure"
    ]
    simulator.run_circuit(unitary_operations)
    return {
        f"{index:0{n_qubits}b}": float(probability)
        for index, probability in enumerate(simulator.get_probabilities())
    }


def _scenario_cases() -> list[tuple[str, str, ValidationCase, dict[str, Any]]]:
    healthy_circuit = [("h", 0)]
    healthy_distribution = _ideal_distribution(healthy_circuit, 1)

    expected_program = [("x", 0)]
    observed_program = [("h", 0)]
    program_expected_distribution = _ideal_distribution(expected_program, 1)
    program_observed_distribution = _ideal_distribution(observed_program, 1)

    hardware_circuit = [("x", 0), ("measure", 0)]
    hardware_expected_distribution = _ideal_distribution(hardware_circuit, 1)
    hardware_observed_counts = {"0": 400, "1": 600}
    hardware_observed_distribution = {outcome: count / DEMO_SHOTS for outcome, count in hardware_observed_counts.items()}
    degraded_hardware = HardwareModel(1)
    degraded_hardware.set_qubit_properties(0, readout_error=0.4)

    ambiguous_expected_distribution = _ideal_distribution(expected_program, 1)
    ambiguous_observed_distribution = _ideal_distribution(observed_program, 1)
    ambiguous_hardware = HardwareModel(1)
    ambiguous_hardware.set_qubit_properties(0, readout_error=0.4)

    insufficient_circuit = [("h", 0)]
    insufficient_expected_distribution = _ideal_distribution(insufficient_circuit, 1)
    insufficient_observed_counts = {"0": DEMO_SHOTS, "1": 0}
    insufficient_observed_distribution = {outcome: count / DEMO_SHOTS for outcome, count in insufficient_observed_counts.items()}

    return [
        (
            "HEALTHY CIRCUIT",
            "NO_ANOMALY",
            ValidationCase(
                case_id="demo_healthy_circuit",
                expected_category=DiagnosisCategory.NO_ANOMALY,
                expected_behavior="NO_ANOMALY",
                expected_circuit=healthy_circuit,
                observed_circuit=healthy_circuit,
                expected_distribution=healthy_distribution,
                observed_distribution=healthy_distribution,
                hardware_model=HardwareModel(1),
                shots=DEMO_SHOTS,
                source="controlled offline demo",
            ),
            {"counts": {"0": 500, "1": 500}, "observed_source": "deterministic controlled observation fixture"},
        ),
        (
            "PROGRAM-LEVEL FAULT",
            "PROGRAM_FAULT",
            ValidationCase(
                case_id="demo_program_fault",
                expected_category=DiagnosisCategory.PROGRAM_FAULT,
                expected_behavior="PROGRAM_FAULT",
                expected_circuit=expected_program,
                observed_circuit=observed_program,
                expected_distribution=program_expected_distribution,
                observed_distribution=program_observed_distribution,
                hardware_model=HardwareModel(1),
                shots=DEMO_SHOTS,
                source="controlled offline demo",
            ),
            {"counts": {"0": 500, "1": 500}, "observed_source": "QuantumSimulator statevector probabilities for the observed program"},
        ),
        (
            "HARDWARE-LEVEL ANOMALY",
            "HARDWARE_ANOMALY",
            ValidationCase(
                case_id="demo_hardware_anomaly",
                expected_category=DiagnosisCategory.HARDWARE_ANOMALY,
                expected_behavior="HARDWARE_ANOMALY",
                expected_circuit=hardware_circuit,
                observed_circuit=hardware_circuit,
                expected_distribution=hardware_expected_distribution,
                observed_distribution=hardware_observed_distribution,
                hardware_model=degraded_hardware,
                shots=DEMO_SHOTS,
                source="controlled offline demo",
            ),
            {"counts": hardware_observed_counts, "observed_source": "deterministic controlled observation fixture; not hardware execution"},
        ),
        (
            "AMBIGUOUS CASE",
            "AMBIGUOUS",
            ValidationCase(
                case_id="demo_ambiguous",
                expected_category=DiagnosisCategory.AMBIGUOUS,
                expected_behavior="AMBIGUOUS",
                expected_circuit=expected_program,
                observed_circuit=observed_program,
                expected_distribution=ambiguous_expected_distribution,
                observed_distribution=ambiguous_observed_distribution,
                hardware_model=ambiguous_hardware,
                shots=DEMO_SHOTS,
                source="controlled offline demo",
            ),
            {"counts": {"0": 500, "1": 500}, "observed_source": "QuantumSimulator statevector probabilities for the observed program"},
        ),
        (
            "INSUFFICIENT-EVIDENCE CASE",
            "INSUFFICIENT_EVIDENCE",
            ValidationCase(
                case_id="demo_insufficient_evidence",
                expected_category=DiagnosisCategory.INSUFFICIENT_EVIDENCE,
                expected_behavior="INSUFFICIENT_EVIDENCE",
                expected_circuit=insufficient_circuit,
                observed_circuit=insufficient_circuit,
                expected_distribution=insufficient_expected_distribution,
                observed_distribution=insufficient_observed_distribution,
                hardware_model=None,
                shots=DEMO_SHOTS,
                source="controlled offline demo",
            ),
            {"counts": insufficient_observed_counts, "observed_source": "deterministic controlled observation fixture; hardware calibration unavailable"},
        ),
    ]


def _candidate_summaries(record: ValidationRecord, candidate_type: str) -> list[dict[str, Any]]:
    diagnosis = record.diagnosis_result
    if diagnosis is None:
        return []
    if candidate_type == "program":
        result = diagnosis.program_result
        if result is None:
            return []
        return [_json_value(candidate) for candidate in result.candidates]
    result = diagnosis.hardware_result
    if result is None:
        return []
    return [_json_value(candidate) for candidate in result.candidates]


def build_demo_report() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for name, expected_label, case, execution_metadata in _scenario_cases():
        record = run_validation_case(case)
        diagnosis = record.diagnosis_result
        stats = diagnosis.statistics_result if diagnosis is not None else None
        scenarios.append({
            "scenario_name": name,
            "case_id": record.case_id,
            "expected_category": expected_label,
            "actual_category": diagnosis.category.value if diagnosis is not None else None,
            "pass": record.validation_satisfied,
            "diagnosis": _json_value(diagnosis),
            "program_evidence": {
                "summary": diagnosis.program_result.summary if diagnosis and diagnosis.program_result else None,
                "candidates": _candidate_summaries(record, "program"),
                "evidence": list(diagnosis.program_result.evidence) if diagnosis and diagnosis.program_result else [],
            },
            "hardware_evidence": {
                "summary": diagnosis.hardware_result.summary if diagnosis and diagnosis.hardware_result else None,
                "candidates": _candidate_summaries(record, "hardware"),
                "evidence": list(diagnosis.hardware_result.evidence) if diagnosis and diagnosis.hardware_result else [],
            },
            "statistical_evidence": _json_value(stats),
            "execution_metadata": {
                **execution_metadata,
                "expected_circuit": _json_value(case.expected_circuit),
                "observed_circuit": _json_value(case.resolved_observed_circuit()),
                "ideal_distribution": _json_value(case.expected_distribution),
                "observed_distribution": _json_value(case.observed_distribution),
                "shots": case.shots,
                "simulator_seed": DEMO_SEED,
                "execution_kind": "offline controlled validation; no IBM backend used",
            },
            "errors": [record.error] if record.error else [],
            "validation_source": record.source,
        })

    passed = sum(scenario["pass"] for scenario in scenarios)
    return {
        "report_name": "Quantum Diagnostic System Offline Demo",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline": True,
        "synthetic_validation_cases": True,
        "real_ibm_execution": False,
        "bugs4q_dataset_used": False,
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": len(scenarios) - passed,
        "all_passed": passed == len(scenarios),
        "note": "Controlled scenarios demonstrate pipeline behavior; they are not Bugs4Q ground truth or IBM hardware results.",
        "scenarios": scenarios,
    }


def _format_operation(operation: Any) -> str:
    if not isinstance(operation, (tuple, list)) or not operation:
        return str(operation)
    name = str(operation[0]).upper()
    operands = operation[1:]
    formatted = []
    for value in operands:
        if isinstance(value, int):
            formatted.append(f"q{value}")
        elif isinstance(value, (tuple, list)):
            formatted.append(",".join(f"q{qubit}" for qubit in value))
        else:
            formatted.append(str(value))
    return f"{name} " + ",".join(formatted) if formatted else name


def render_human_report(report: Mapping[str, Any]) -> str:
    lines = ["=" * 58, "QUANTUM DIAGNOSTIC SYSTEM", "OFFLINE END-TO-END DEMO", "=" * 58]
    for scenario in report["scenarios"]:
        lines.extend(["", f"Scenario: {scenario['scenario_name']}", "Circuit:"])
        for operation in scenario["execution_metadata"]["expected_circuit"]:
            lines.append(f"    {_format_operation(operation)}")
        observed_circuit = scenario["execution_metadata"]["observed_circuit"]
        if observed_circuit != scenario["execution_metadata"]["expected_circuit"]:
            lines.append("Observed program:")
            for operation in observed_circuit:
                lines.append(f"    {_format_operation(operation)}")
        lines.extend([
            f"Expected: {scenario['expected_category']}",
            f"Detected: {scenario['actual_category']}",
            "Program candidates:",
        ])
        program_candidates = scenario["program_evidence"]["candidates"]
        lines.extend([
            f"    step {candidate['index']}: {candidate['operation']}"
            + (f" on qubits {candidate['qubits']}" if candidate["qubits"] else "")
            + f" ({candidate['reason']})"
            for candidate in program_candidates
        ] or ["    none"])
        lines.append("Hardware candidates:")
        hardware_candidates = scenario["hardware_evidence"]["candidates"]
        lines.extend([f"    {candidate['component_type']} {candidate['component']}: {candidate['reason']}" for candidate in hardware_candidates] or ["    none"])
        stats = scenario["statistical_evidence"]
        lines.append("Statistical evidence:")
        if stats is None:
            lines.append("    unavailable")
        else:
            lines.append(f"    TVD={stats['total_variation_distance']:.6f}; threshold={max(stats['threshold'], stats['finite_shot_guard']):.6f}; anomaly={stats['anomaly_detected']}")
        lines.append("Evidence:")
        evidence = scenario["diagnosis"]["evidence"] if scenario["diagnosis"] else []
        lines.extend([f"    {entry}" for entry in evidence[:3]] or ["    none"])
        if len(evidence) > 3:
            lines.append(f"    ... {len(evidence) - 3} additional evidence item(s) in JSON report")
        lines.append("Confidence/evidence strength:")
        lines.append(f"    no confidence score calculated; {len(program_candidates)} program and {len(hardware_candidates)} hardware candidate(s)")
        lines.append(f"Status: {'PASS' if scenario['pass'] else 'FAIL'}")
        lines.append("-" * 58)

    lines.extend([
        "",
        f"Scenarios passed: {report['passed']}/{report['scenario_count']}",
        "All observations are offline controlled fixtures or simulator distributions; no IBM hardware was used.",
        "The actual Bugs4Q dataset was not used.",
    ])
    return "\n".join(lines)


def run_demo(output_path: str | Path = DEFAULT_REPORT) -> dict[str, Any]:
    report = build_demo_report()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline quantum diagnostic pipeline demo.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a human-readable summary")
    parser.add_argument("--output", default=str(DEFAULT_REPORT), help="JSON report output path")
    args = parser.parse_args()

    report = run_demo(args.output)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_human_report(report))
    return 0 if report["all_passed"] else 1


__all__ = ["build_demo_report", "render_human_report", "run_demo"]


if __name__ == "__main__":
    raise SystemExit(main())
