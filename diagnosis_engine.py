"""Integrated, evidence-preserving diagnosis for circuit executions.

This module composes the existing statistics, program-localization, and
hardware-localization layers. It does not reimplement their scoring logic or
force a program-versus-hardware decision when execution evidence is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from fault_localizer import DiagnosticTrace, FaultLocalizationResult, localize_program_fault
from hardware_fault_localizer import (
    HardwareCandidate,
    HardwareFaultLocalizationResult,
    localize_hardware_fault,
)
from hardware_model import HardwareModel
from statistics import DistributionComparisonResult, compare_distributions
from gates import get_gate_qubit_count


class DiagnosisCategory(str, Enum):
    NO_ANOMALY = "NO_ANOMALY"
    PROGRAM_FAULT = "PROGRAM_FAULT"
    HARDWARE_ANOMALY = "HARDWARE_ANOMALY"
    AMBIGUOUS = "AMBIGUOUS / INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "AMBIGUOUS / INSUFFICIENT_EVIDENCE"


@dataclass
class DiagnosisResult:
    anomaly_detected: bool | None
    category: DiagnosisCategory
    statistics_result: DistributionComparisonResult | None
    program_result: FaultLocalizationResult | None
    hardware_result: HardwareFaultLocalizationResult | None
    program_evidence_candidates: list[Any] = field(default_factory=list)
    hardware_evidence_candidates: list[HardwareCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    summary: str = ""


def _operation_details(operation: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(operation, Mapping):
        name = str(operation.get("name") or operation.get("op") or operation.get("gate") or "unknown").lower()
        qubits = operation.get("qubits", [])
        if isinstance(qubits, int):
            qubits = [qubits]
        return name, tuple(int(qubit) for qubit in qubits)

    if isinstance(operation, (tuple, list)) and operation:
        name = str(operation[0]).lower()
        if name == "measure" and len(operation) > 1 and isinstance(operation[1], (tuple, list)):
            return name, tuple(int(value) for value in operation[1])
        qubits = tuple(int(value) for value in operation[1:] if isinstance(value, int) and not isinstance(value, bool))
        return name, qubits

    return "unknown", ()


def _canonicalize_circuit(circuit: Sequence[Any] | None) -> list[tuple[Any, ...]] | None:
    if circuit is None:
        return None

    parameterized = {"rx", "ry", "rz", "p", "u1", "u2", "u3", "cp", "crx", "cry", "crz"}
    normalized: list[tuple[Any, ...]] = []
    for operation in circuit:
        if isinstance(operation, Mapping):
            name = str(operation.get("name") or operation.get("op") or operation.get("gate") or "unknown").lower()
            qubits = operation.get("qubits", [])
            parameters = operation.get("parameters", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            if name == "measure":
                classical_bits = operation.get("classical_bits", operation.get("clbits", []))
                if isinstance(classical_bits, int):
                    classical_bits = [classical_bits]
                normalized.append((name, tuple(int(qubit) for qubit in qubits), tuple(int(bit) for bit in classical_bits)))
                continue
            normalized.append((name, *[int(qubit) for qubit in qubits], *[float(value) for value in parameters]))
            continue

        if not isinstance(operation, (tuple, list)) or not operation:
            normalized.append(("unknown",))
            continue

        name = str(operation[0]).lower()
        operands = list(operation[1:])
        if name == "measure":
            if len(operands) > 1 and isinstance(operands[0], (tuple, list)):
                qubits, classical_bits = operands[:2]
            elif len(operands) > 1 and isinstance(operands[0], int) and isinstance(operands[1], (tuple, list)):
                qubits, classical_bits = [operands[0]], operands[1]
            else:
                qubits, classical_bits = operands, []
            normalized.append((name, tuple(int(value) for value in qubits), tuple(int(value) for value in classical_bits)))
            continue
        elif name == "reset":
            qubits = operands[:1]
            parameters = []
        elif name == "barrier":
            qubits = operands
            parameters = []
        elif name in parameterized:
            qubits = operands[:1]
            parameters = operands[1:]
        else:
            try:
                qubit_count = get_gate_qubit_count(name)
            except (TypeError, ValueError):
                qubit_count = len(operands)
            qubits = operands[:qubit_count]
            parameters = operands[qubit_count:]

        normalized.append((name, *[int(qubit) for qubit in qubits], *[float(value) for value in parameters]))

    return normalized


def _deterministic_program_circuits(
    expected: list[tuple[Any, ...]] | None,
    observed: list[tuple[Any, ...]] | None,
) -> tuple[list[tuple[Any, ...]] | None, list[tuple[Any, ...]] | None]:
    if expected is None or observed is None or len(expected) != len(observed):
        return expected, observed

    expected_ops = [_operation_details(operation) for operation in expected]
    observed_ops = [_operation_details(operation) for operation in observed]
    if any(
        (expected_sig[0] == "measure") != (observed_sig[0] == "measure")
        or (expected_sig[0] == "measure" and expected != observed)
        for expected, observed, expected_sig, observed_sig
        in zip(expected, observed, expected_ops, observed_ops)
    ):
        return expected, observed

    expected_deterministic = [
        ("barrier", *qubits) if name == "measure" else operation
        for operation, (name, qubits) in zip(expected, expected_ops)
    ]
    observed_deterministic = [
        ("barrier", *qubits) if name == "measure" else operation
        for operation, (name, qubits) in zip(observed, observed_ops)
    ]
    return expected_deterministic, observed_deterministic


def _calibration_gaps(hardware_model: HardwareModel, circuit: Sequence[Any] | None) -> list[str]:
    if circuit is None:
        return []

    aliases = {"cnot": "cx", "id": "i", "identity": "i"}
    gaps: list[str] = []
    reported: set[str] = set()
    for operation in circuit:
        name, qubits = _operation_details(operation)
        if name in {"measure", "reset", "barrier", "unknown"}:
            continue
        calibration_name = aliases.get(name, name)
        key = f"{calibration_name} on {qubits}"
        if qubits and not hardware_model.get_gate_calibration(calibration_name, qubits=qubits) and key not in reported:
            gaps.append(f"No gate calibration entry is available for {calibration_name} on qubits {qubits}.")
            reported.add(key)
        if len(qubits) >= 2:
            for left, right in zip(qubits, qubits[1:]):
                edge_key = tuple(sorted((left, right)))
                if hardware_model.get_edge(*edge_key) is None:
                    message = f"No coupling-edge calibration is available for edge {edge_key}."
                    if message not in reported:
                        gaps.append(message)
                        reported.add(message)

    return gaps


def _has_calibration_fault_evidence(candidate: HardwareCandidate) -> bool:
    calibration_markers = (
        "elevated",
        "short at",
        "not available",
        "unsupported",
        "long duration",
    )
    text = " ".join([candidate.reason, *candidate.evidence]).lower()
    return any(marker in text for marker in calibration_markers)


def _is_relevant_to_circuit(candidate: HardwareCandidate, circuit: Sequence[Any] | None) -> bool:
    if circuit is None:
        return False

    operations = [_operation_details(operation) for operation in circuit]
    used_qubits = {qubit for _, qubits in operations for qubit in qubits}

    if candidate.component_type == "qubit":
        measures_all = any(name == "measure" and not qubits for name, qubits in operations)
        return candidate.component in used_qubits or measures_all

    if candidate.component_type == "edge":
        target = tuple(sorted(candidate.component))
        return any(len(qubits) >= 2 and tuple(sorted(qubits[:2])) == target for _, qubits in operations)

    if candidate.component_type == "gate":
        target_qubits = tuple(candidate.component) if isinstance(candidate.component, (tuple, list)) else ()
        return any(len(qubits) >= len(target_qubits) and tuple(qubits[:len(target_qubits)]) == target_qubits for _, qubits in operations)

    return False


def diagnose_execution(
    expected_distribution: Mapping[str, float] | Sequence[float] | None = None,
    observed_distribution: Mapping[str, float] | Sequence[float] | None = None,
    *,
    expected_circuit: Sequence[Any] | None = None,
    observed_circuit: Sequence[Any] | None = None,
    hardware_model: HardwareModel | None = None,
    threshold: float = 0.05,
    shots: int | None = None,
    precomputed_trace: DiagnosticTrace | None = None,
) -> DiagnosisResult:
    """Combine execution anomaly evidence with program and hardware candidates.

    Distributions are optional as a pair. Without both, anomaly status is
    unknown and the final category remains insufficient evidence. Circuits and
    hardware calibration can still be analyzed and are returned independently.
    """

    if (expected_distribution is None) != (observed_distribution is None):
        raise ValueError("expected_distribution and observed_distribution must be provided together")
    if expected_circuit is not None and not isinstance(expected_circuit, Sequence):
        raise TypeError("expected_circuit must be a sequence of operations")
    if observed_circuit is not None and not isinstance(observed_circuit, Sequence):
        raise TypeError("observed_circuit must be a sequence of operations")
    if hardware_model is not None and not isinstance(hardware_model, HardwareModel):
        raise TypeError("hardware_model must be a HardwareModel instance or None")

    expected_localizer_circuit = _canonicalize_circuit(expected_circuit)
    observed_localizer_circuit = _canonicalize_circuit(observed_circuit)

    statistics_result = None
    anomaly_detected: bool | None = None
    if expected_distribution is not None and observed_distribution is not None:
        statistics_result = compare_distributions(
            expected_distribution,
            observed_distribution,
            threshold=threshold,
            shots=shots,
        )
        anomaly_detected = statistics_result.anomaly_detected

    program_result = None
    if expected_localizer_circuit is not None and observed_localizer_circuit is not None:
        expected_program_circuit, observed_program_circuit = _deterministic_program_circuits(
            expected_localizer_circuit,
            observed_localizer_circuit,
        )
        program_result = localize_program_fault(
            expected_program_circuit,
            observed_program_circuit,
            threshold=threshold,
            shots=shots,
            precomputed_trace=precomputed_trace,
        )

    hardware_result = None
    missing_evidence: list[str] = []
    hardware_evidence_candidates: list[HardwareCandidate] = []
    if hardware_model is None:
        missing_evidence.append("Hardware calibration information was not provided.")
    else:
        circuit_for_calibration = observed_localizer_circuit or expected_localizer_circuit
        missing_evidence.extend(_calibration_gaps(hardware_model, circuit_for_calibration))
        if expected_localizer_circuit is not None or observed_localizer_circuit is not None:
            hardware_result = localize_hardware_fault(
                hardware_model,
                expected_localizer_circuit,
                observed_localizer_circuit,
                threshold=threshold,
            )
            hardware_evidence_candidates = [
                candidate
                for candidate in hardware_result.candidates
                if _has_calibration_fault_evidence(candidate)
                and _is_relevant_to_circuit(candidate, circuit_for_calibration)
            ]

    if expected_distribution is None:
        missing_evidence.append("Expected and observed execution distributions were not provided.")

    program_candidates = program_result.candidates if program_result is not None else []
    evidence: list[str] = []
    if statistics_result is not None:
        evidence.append(statistics_result.message)
    if program_result is not None:
        evidence.extend(program_result.evidence)
    if hardware_result is not None:
        evidence.extend(hardware_result.evidence)
    evidence.extend(missing_evidence)

    if anomaly_detected is False:
        category = DiagnosisCategory.NO_ANOMALY
        summary = "No statistically meaningful execution anomaly was detected."
    elif anomaly_detected is None:
        category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        summary = "Execution anomaly status is unknown because distribution evidence is incomplete."
    elif program_candidates and hardware_evidence_candidates:
        category = DiagnosisCategory.AMBIGUOUS
        summary = "Both program-level and calibration-backed hardware evidence are present; neither cause is isolated."
    elif program_candidates:
        category = DiagnosisCategory.PROGRAM_FAULT
        summary = "An execution anomaly is accompanied by program-operation fault evidence."
    elif hardware_evidence_candidates:
        category = DiagnosisCategory.HARDWARE_ANOMALY
        summary = "An execution anomaly is accompanied by relevant hardware calibration evidence."
    else:
        category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        summary = "An execution anomaly was detected, but available evidence does not isolate a program or hardware cause."

    return DiagnosisResult(
        anomaly_detected=anomaly_detected,
        category=category,
        statistics_result=statistics_result,
        program_result=program_result,
        hardware_result=hardware_result,
        program_evidence_candidates=list(program_candidates),
        hardware_evidence_candidates=hardware_evidence_candidates,
        evidence=evidence,
        missing_evidence=missing_evidence,
        summary=summary,
    )


__all__ = [
    "DiagnosisCategory",
    "DiagnosisResult",
    "diagnose_execution",
]
