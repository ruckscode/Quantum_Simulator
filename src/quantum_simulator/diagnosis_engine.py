"""Integrated, evidence-preserving diagnosis for circuit executions.

This module composes the existing statistics, program-localization, and
hardware-localization layers. It does not reimplement their scoring logic or
force a program-versus-hardware decision when execution evidence is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .fault_localizer import (
    DiagnosticTrace,
    FaultLocalizationResult,
    _infer_n_qubits,
    _trace_deviation,
    _unsupported_operations,
    localize_program_fault,
)
from .hardware_fault_localizer import (
    HardwareCandidate,
    HardwareFaultLocalizationResult,
    _hardware_model_base_candidates,
    localize_hardware_fault,
)
from .hardware_model import HardwareModel, assess_calibration_freshness
from .statistics import DistributionComparisonResult, compare_distributions
from .quantum_simulator import QuantumSimulator
from .gates import get_gate_qubit_count


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
    trace_deviation_candidates: list[Any] = field(default_factory=list)
    program_behavioral_candidates: list[Any] = field(default_factory=list)
    calibration_freshness: dict[str, Any] | None = None


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


def _is_simulator_supported_operation(name: str) -> bool:
    if name in {"measure", "barrier", "reset"}:
        return True
    try:
        get_gate_qubit_count(name)
    except (TypeError, ValueError):
        return False
    return True


def _canonicalize_circuit(circuit: Sequence[Any] | None) -> list[tuple[Any, ...]] | None:
    if circuit is None:
        return None

    parameterized = {"rx", "ry", "rz", "p", "u1", "u2", "u3"}
    two_qubit_parameterized = {"cp", "crx", "cry", "crz", "rxx", "rzz"}
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
            if not operands:
                qubits, classical_bits = (), ()
            elif isinstance(operands[0], (tuple, list)):
                # measure_all() emits a qubit list without explicit classical
                # destinations; explicit mappings may provide them as operand 2.
                qubits = operands[0]
                classical_bits = operands[1] if len(operands) > 1 else ()
            else:
                # measure(q) and measure(q, [classical_bit]) are both supported.
                qubits = [operands[0]]
                classical_bits = operands[1] if len(operands) > 1 else ()
            if isinstance(qubits, int):
                qubits = [qubits]
            if isinstance(classical_bits, int):
                classical_bits = [classical_bits]
            normalized.append((name, tuple(int(value) for value in qubits), tuple(int(value) for value in classical_bits)))
            continue
        elif name == "reset":
            qubits = operands[:1]
            parameters = []
        elif name == "barrier":
            qubits = operands
            parameters = []
        elif name in two_qubit_parameterized:
            qubits = operands[:2]
            parameters = operands[2:]
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
        if not _is_simulator_supported_operation(name):
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

    operations = []
    for operation in circuit:
        details = _operation_details(operation)
        if details[0] in {"measure", "reset", "barrier"} or _is_simulator_supported_operation(details[0]):
            operations.append(details)
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


def _candidate_can_affect_measured_outputs(
    candidate: Any,
    expected_circuit: Sequence[Any] | None,
    observed_circuit: Sequence[Any] | None,
) -> bool:
    """Conservatively check whether a source candidate reaches an observed output.

    This is structural relevance only. A separate counterfactual execution
    comparison must also show that the source change can alter the output.
    """
    candidate_qubits = set(candidate.qubits or ())
    if not candidate_qubits:
        return candidate.operation == "measure"

    circuits = [circuit for circuit in (expected_circuit, observed_circuit) if circuit is not None]
    measured_qubits: set[int] = set()
    all_qubits: set[int] = set(candidate_qubits)
    entangling_sets: list[set[int]] = []
    measure_all = False
    for circuit in circuits:
        for operation in circuit:
            name, qubits = _operation_details(operation)
            qset = set(qubits)
            all_qubits.update(qset)
            if name == "measure":
                if qset:
                    measured_qubits.update(qset)
                else:
                    measure_all = True
            elif name != "barrier" and len(qset) > 1:
                entangling_sets.append(qset)

    if not measured_qubits and not measure_all:
        measured_qubits = all_qubits
    elif measure_all:
        measured_qubits = all_qubits

    reachable = set(candidate_qubits)
    changed = True
    while changed:
        changed = False
        for operands in entangling_sets:
            if reachable & operands and not operands <= reachable:
                reachable.update(operands)
                changed = True
    return bool(reachable & measured_qubits)


def has_relevant_calibration_fault_evidence(
    hardware_model: HardwareModel | None,
    circuit: Sequence[Any] | None,
) -> bool:
    """Check existing calibration scoring for fault evidence on this circuit.

    This preflight reuses the hardware localizer's calibration candidates and
    this engine's evidence/relevance filters without invoking either localizer.
    """
    if hardware_model is None or circuit is None:
        return False
    if not isinstance(hardware_model, HardwareModel):
        raise TypeError("hardware_model must be a HardwareModel instance or None")
    normalized_circuit = _canonicalize_circuit(circuit)
    return any(
        _has_calibration_fault_evidence(candidate)
        and _is_relevant_to_circuit(candidate, normalized_circuit)
        for candidate in _hardware_model_base_candidates(hardware_model)
    )


def diagnose_execution(
    expected_distribution: Mapping[str, float] | Sequence[float] | None = None,
    observed_distribution: Mapping[str, float] | Sequence[float] | None = None,
    *,
    expected_circuit: Sequence[Any] | None = None,
    observed_circuit: Sequence[Any] | None = None,
    hardware_model: HardwareModel | None = None,
    threshold: float = 0.05,
    shots: int | None = None,
    precomputed_trace: DiagnosticTrace | Sequence[dict[str, Any]] | None = None,
    max_calibration_age_seconds: float | None = None,
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

    # The traced runner returns a list for API compatibility. Its rows already
    # carry per-step evidence, so wrap them in the localizer's existing format.
    if isinstance(precomputed_trace, list):
        metric = float(precomputed_trace[-1].get("difference", 0.0)) if precomputed_trace else 0.0
        precomputed_trace = (metric, precomputed_trace)

    expected_localizer_circuit = _canonicalize_circuit(expected_circuit)
    observed_localizer_circuit = _canonicalize_circuit(observed_circuit)
    unsupported_operations = _unsupported_operations(
        expected_localizer_circuit, observed_localizer_circuit,
    )

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
    expected_program_circuit = expected_localizer_circuit
    observed_program_circuit = observed_localizer_circuit
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
    calibration_freshness = None
    missing_evidence: list[str] = []
    hardware_evidence_candidates: list[HardwareCandidate] = []
    if hardware_model is None:
        missing_evidence.append("Hardware calibration information was not provided.")
    else:
        calibration_freshness = assess_calibration_freshness(
            hardware_model.calibration_timestamp,
            provenance=hardware_model.calibration_timestamp_provenance,
            max_age_seconds=max_calibration_age_seconds,
        )
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
            # Synthetic calibration describes the simulator fixture itself and
            # has no physical age claim. Real/reported calibration contributes
            # current hardware evidence only when the caller's age policy says
            # it is fresh.
            if calibration_freshness["provenance"] != "synthetic" and calibration_freshness["status"] != "FRESH":
                if hardware_evidence_candidates:
                    missing_evidence.append(
                        "Hardware calibration freshness is not established; calibration candidates are retained for review but excluded from causal attribution."
                    )
                hardware_evidence_candidates = []

    if expected_distribution is None:
        missing_evidence.append(
            "Expected and observed execution distributions were not provided; "
            "source differences cannot establish behavioral impact."
        )
    if unsupported_operations:
        missing_evidence.append(
            "Complete simulator trace localization is unavailable because these operations are "
            f"unsupported: {', '.join(unsupported_operations)}."
        )

    all_program_candidates = program_result.candidates if program_result is not None else []
    trace_deviation_candidates = [
        candidate for candidate in all_program_candidates
        if candidate.reason == "Largest traced deviation without direct operation mismatch"
    ]
    # A state/distribution deviation locates a suspicious step, but cannot by
    # itself distinguish a program fault from hardware-induced execution noise.
    program_candidates = [
        candidate for candidate in all_program_candidates
        if candidate.reason != "Largest traced deviation without direct operation mismatch"
    ]
    counterfactual_metric = program_result.anomaly_metric if program_result is not None else 0.0
    if (
        precomputed_trace is not None
        and expected_program_circuit is not None
        and observed_program_circuit is not None
        and not unsupported_operations
    ):
        # Traced execution rows describe step-local changes. Recompute the
        # source-only terminal-output counterfactual instead of interpreting
        # the final trace row as a whole-circuit impact metric.
        counterfactual_metric, _ = _trace_deviation(
            expected_program_circuit,
            observed_program_circuit,
            max(
                _infer_n_qubits(expected_program_circuit),
                _infer_n_qubits(observed_program_circuit),
            ),
        )
    counterfactual_output_changed = (
        program_result is not None
        and not unsupported_operations
        and counterfactual_metric > threshold
    )
    program_behavioral_candidates = [
        candidate for candidate in program_candidates
        if (
            (
                candidate.operation == "measure"
                and candidate.expected_signature != candidate.actual_signature
            )
            or counterfactual_output_changed
        )
        and _candidate_can_affect_measured_outputs(
            candidate, expected_localizer_circuit, observed_localizer_circuit,
        )
    ]
    direct_program_mismatch = bool(program_candidates)
    evidence: list[str] = []
    if statistics_result is not None:
        evidence.append(statistics_result.message)
    if program_result is not None:
        evidence.extend(program_result.evidence)
    if direct_program_mismatch:
        evidence.append("A direct circuit-operation mismatch was localized.")
    if program_behavioral_candidates:
        evidence.append(
            "Counterfactual circuit execution changes computational-basis outcomes on measured outputs."
        )
    elif trace_deviation_candidates:
        evidence.append(
            "A traced execution deviation was localized, but it does not by itself identify a program fault."
        )
    if hardware_result is not None:
        evidence.extend(hardware_result.evidence)
    evidence.extend(missing_evidence)

    if anomaly_detected is False and program_candidates and hardware_evidence_candidates:
        category = DiagnosisCategory.AMBIGUOUS
        summary = (
            "The observed distribution matches the reference, but both program-operation and "
            "calibration-backed hardware fault evidence remain; neither cause is isolated."
        )
        evidence.append(
            "Observed distributions match, but direct program and relevant hardware fault evidence remain."
        )
    elif anomaly_detected is False and program_candidates:
        category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        summary = (
            "A source-level mismatch was localized, but the observed distribution matches "
            "the reference; behavioral fault classification is unsupported."
        )
        evidence.append("Source mismatch alone does not establish an observable program fault.")
    elif anomaly_detected is False and hardware_evidence_candidates:
        category = DiagnosisCategory.HARDWARE_ANOMALY
        summary = "The observed distribution matches the reference, but relevant hardware calibration fault evidence remains."
    elif anomaly_detected is False:
        if unsupported_operations and not program_candidates and not hardware_evidence_candidates:
            category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
            summary = "No anomaly was detected, but unsupported operations prevent complete trace-based verification."
        else:
            category = DiagnosisCategory.NO_ANOMALY
            summary = "No statistically meaningful execution anomaly or direct fault evidence was detected."
    elif anomaly_detected is None:
        category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        summary = "Execution anomaly status is unknown because distribution evidence is incomplete."
    elif program_candidates and hardware_evidence_candidates:
        category = DiagnosisCategory.AMBIGUOUS
        summary = "Both a program source mismatch and calibration-backed hardware evidence remain; neither cause is isolated."
    elif program_behavioral_candidates:
        category = DiagnosisCategory.PROGRAM_FAULT
        summary = "An execution anomaly is accompanied by a source mismatch that changes modeled measured outputs."
    elif hardware_evidence_candidates:
        category = DiagnosisCategory.HARDWARE_ANOMALY
        summary = "An execution anomaly is accompanied by relevant hardware calibration evidence."
    else:
        category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        summary = (
            "An execution anomaly was detected, but available evidence does not isolate a cause."
            if not program_candidates else
            "An execution anomaly was detected, but the source mismatch is not linked to the measured outputs and hardware evidence is insufficient."
        )

    return DiagnosisResult(
        anomaly_detected=anomaly_detected,
        category=category,
        statistics_result=statistics_result,
        program_result=program_result,
        hardware_result=hardware_result,
        program_evidence_candidates=list(program_candidates),
        trace_deviation_candidates=trace_deviation_candidates,
        hardware_evidence_candidates=hardware_evidence_candidates,
        evidence=evidence,
        missing_evidence=missing_evidence,
        summary=summary,
        program_behavioral_candidates=list(program_behavioral_candidates),
        calibration_freshness=calibration_freshness,
    )


def diagnose_circuit(
    circuit: Sequence[Any] | Any,
    *,
    reference_circuit: Sequence[Any] | Any | None = None,
    hardware_model: HardwareModel | None = None,
    shots: int = 1000,
    seed: int | None = None,
    threshold: float = 0.05,
    max_calibration_age_seconds: float | None = None,
) -> DiagnosisResult:
    """Run and diagnose a submitted circuit, optionally against a reference.

    ``circuit`` is the observed/submitted program. When ``reference_circuit``
    is supplied, its ideal sampled distribution is compared with the submitted
    circuit's sampled distribution; ``hardware_model`` is applied only while
    sampling the submitted circuit and is also supplied as calibration evidence
    to diagnosis. Both circuits must emit compatible measurement schemas:
    the same measurement-operation structure, output widths, and classical-bit
    destinations. With no reference, the historical same-circuit comparison
    is unchanged.

    A source-level operation mismatch can be localized even when the measured
    distributions agree. In that case the result is abstention with the source
    candidate retained: a syntactic difference is not proof that behavior
    changed. If observable and evidence channels are insufficient or
    conflicting, diagnosis remains ambiguous/insufficient; observationally
    indistinguishable faults cannot be discovered from these inputs alone.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if hardware_model is not None and not isinstance(hardware_model, HardwareModel):
        raise TypeError("hardware_model must be a HardwareModel instance or None")
    instructions = getattr(circuit, "instructions", circuit)
    if not isinstance(instructions, Sequence):
        raise TypeError("circuit must be a sequence of supported instructions")
    expected_instructions = instructions if reference_circuit is None else getattr(
        reference_circuit, "instructions", reference_circuit,
    )
    if not isinstance(expected_instructions, Sequence):
        raise TypeError("reference_circuit must be a sequence of supported instructions")

    # Normalize through the simulator so tuple and dictionary instructions use
    # exactly the same operation parsing as execution.
    probe = QuantumSimulator(1)
    highest_qubit = -1
    for program in (instructions,) if reference_circuit is None else (expected_instructions, instructions):
        for instruction in program:
            name, operands = probe._normalize_operation(instruction)
            qubits = []
            if name == "measure":
                if operands:
                    qubits = list(operands[0]) if isinstance(operands[0], (list, tuple)) else [operands[0]]
            elif name == "barrier":
                qubits = list(operands)
            elif name not in {"reset"}:
                count = 3 if name in {"ccx", "toffoli", "cswap", "fredkin"} else 2 if name in {"cx", "cnot", "cz", "swap", "ch"} else (1 if operands else 0)
                qubits = list(operands[:count])
            elif operands:
                qubits = [operands[0]]
            if qubits:
                highest_qubit = max(highest_qubit, *(int(q) for q in qubits))
    n_qubits = max(highest_qubit + 1, hardware_model.n_qubits if hardware_model else 0)
    if n_qubits <= 0:
        raise ValueError("circuit must reference at least one qubit")

    def measurement_schema(program: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
        schema = []
        for instruction in program:
            name, operands = probe._normalize_operation(instruction)
            if name != "measure":
                continue
            if not operands:
                schema.append(("all", n_qubits))
                continue
            qubits = list(operands[0]) if isinstance(operands[0], (list, tuple)) else [operands[0]]
            classical_bits = operands[1] if len(operands) > 1 else None
            if classical_bits is None or (
                isinstance(classical_bits, (list, tuple)) and not classical_bits
            ):
                schema.append(("unmapped", len(qubits)))
            else:
                destinations = tuple(classical_bits) if isinstance(classical_bits, (list, tuple)) else (classical_bits,)
                if len(destinations) != len(qubits):
                    raise ValueError("measurement classical-bit count must match measured-qubit count")
                schema.append(("mapped", destinations))
        return tuple(schema) if schema else (("statevector", n_qubits),)

    if reference_circuit is not None:
        expected_measurements = measurement_schema(expected_instructions)
        observed_measurements = measurement_schema(instructions)
        if expected_measurements != observed_measurements:
            raise ValueError("reference_circuit and circuit must have compatible measurement schemas")

    classical_width = 0
    has_classical_mapping = False
    for instruction in instructions:
        name, operands = probe._normalize_operation(instruction)
        if name == "measure" and len(operands) > 1 and operands[1] is not None:
            bits = operands[1]
            bits = list(bits) if isinstance(bits, (list, tuple)) else [bits]
            if bits:
                has_classical_mapping = True
                classical_width = max(classical_width, max(map(int, bits)) + 1)

    def sample(
        program: Sequence[Any], model: HardwareModel | None, stream_seed: int | None,
    ) -> dict[str, float]:
        rng = np.random.default_rng(stream_seed)
        counts: dict[str, int] = {}
        for _ in range(shots):
            simulator = QuantumSimulator(n_qubits, seed=int(rng.integers(0, 2**31 - 1)))
            results = simulator.run_circuit(program, hardware_model=model)
            if has_classical_mapping:
                classical = [0] * classical_width
                for result in results:
                    if isinstance(result, dict):
                        for bit, value in result.items():
                            classical[int(bit)] = int(value)
                outcome = "".join(str(bit) for bit in reversed(classical))
            elif simulator.measurement_results:
                outcome = str(simulator.measurement_results[-1])
            else:
                index = int(simulator.rng.choice(simulator.dim, p=simulator.get_probabilities()))
                outcome = f"{index:0{n_qubits}b}"
            counts[outcome] = counts.get(outcome, 0) + 1
        return {key: value / shots for key, value in counts.items()}

    expected = sample(expected_instructions, None, seed)
    if reference_circuit is None and hardware_model is None:
        observed = expected
    else:
        observed = sample(
            instructions, hardware_model, None if seed is None else seed + 1,
        )
    all_outcomes = set(expected) | set(observed)
    expected = {key: expected.get(key, 0.0) for key in all_outcomes}
    observed = {key: observed.get(key, 0.0) for key in all_outcomes}
    result = diagnose_execution(
        expected,
        observed,
        expected_circuit=expected_instructions,
        observed_circuit=instructions,
        hardware_model=hardware_model,
        threshold=threshold,
        shots=shots,
        max_calibration_age_seconds=max_calibration_age_seconds,
    )
    if (
        reference_circuit is not None
        and result.statistics_result is not None
        and not result.statistics_result.anomaly_detected
        and result.program_evidence_candidates
        and not result.hardware_evidence_candidates
    ):
        result.category = DiagnosisCategory.INSUFFICIENT_EVIDENCE
        result.summary = (
            "A source-level mismatch was localized, but the sampled outputs "
            "show no behavioral anomaly; abstaining on fault classification."
        )
        result.evidence.append(
            "Source mismatch alone does not establish an observable program fault."
        )
    return result


def _format_gate_signature(signature: Any | None) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, (tuple, list)) and signature:
        name = str(signature[0]).upper()
        qubits = signature[1] if len(signature) > 1 else ()
        if isinstance(qubits, int):
            qubits = (qubits,)
        try:
            return f"{name}({', '.join(str(qubit) for qubit in qubits)})"
        except TypeError:
            return f"{name}"
    return str(signature).upper()


def format_program_fault_diagnosis(result: DiagnosisResult) -> str:
    """Format a supported program diagnosis or label an unproven source candidate."""
    if not isinstance(result, DiagnosisResult):
        raise TypeError("result must be a DiagnosisResult")
    candidates = result.program_evidence_candidates
    if result.category is not DiagnosisCategory.PROGRAM_FAULT:
        if not candidates:
            return result.summary
        candidate = candidates[0]
        actual = _format_gate_signature(candidate.actual_signature)
        expected = _format_gate_signature(candidate.expected_signature)
        return "\n".join([
            "=== PROGRAM SOURCE MISMATCH CANDIDATE ===",
            "Causal attribution is not established.",
            f"Candidate location : Step {candidate.index}",
            f"Observed operation : {actual or candidate.operation}",
            f"Expected operation : {expected or 'No operation'}",
            f"Localization score : {candidate.score:.2f}",
            f"Diagnosis status   : {result.category.value}",
            f"Summary            : {result.summary}",
        ])
    if not candidates:
        return result.summary

    candidate = candidates[0]
    actual = _format_gate_signature(candidate.actual_signature)
    expected = _format_gate_signature(candidate.expected_signature)
    if actual is None:
        actual = _format_gate_signature((candidate.operation, candidate.qubits)) or "Unknown"
    if expected is None:
        expected = "No operation"
    lines = [
        "=== PROGRAM FAULT DETECTED ===",
        "",
        f"Fault location : Step {candidate.index}",
        f"Faulty gate    : {actual}",
        f"Expected gate  : {expected}",
        f"Mismatch score : {candidate.score:.2f}",
        "",
        "Reason:",
        candidate.reason,
        "",
        "Suggested correction:",
        f"Replace {actual} with {expected}",
    ]
    return "\n".join(lines)


__all__ = [
    "DiagnosisCategory",
    "DiagnosisResult",
    "diagnose_circuit",
    "diagnose_execution",
    "format_program_fault_diagnosis",
]
