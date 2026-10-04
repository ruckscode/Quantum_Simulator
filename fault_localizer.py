"""Program-fault localization for quantum circuits.

This layer is intentionally limited to operation-level diagnosis for the program
itself. It does not classify hardware anomalies or localize hardware faults.
It builds on the existing simulator trace and statistical layer to compare the
expected circuit with an observed circuit and identify suspicious operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, TypedDict

from gates import get_gate_qubit_count
from quantum_simulator import QuantumSimulator
from statistics import compare_distributions


@dataclass
class LocalizationCandidate:
    index: int
    operation: str
    qubits: tuple[int, ...]
    expected_signature: Any | None
    actual_signature: Any | None
    score: float
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class FaultLocalizationResult:
    candidates: List[LocalizationCandidate]
    anomaly_metric: float
    threshold: float
    summary: str
    evidence: List[str] = field(default_factory=list)


def _normalize_operation(op: Any) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    if isinstance(op, dict):
        name = str(op.get("name") or op.get("op") or op.get("gate") or "unknown").lower()
        raw_qubits = op.get("qubits", [])
        if isinstance(raw_qubits, int):
            raw_qubits = [raw_qubits]
        raw_params = op.get("parameters", op.get("params", []))
        if isinstance(raw_params, (int, float)):
            raw_params = [raw_params]
        qubits = tuple(int(q) for q in raw_qubits)
        if name == "measure":
            classical_bits = op.get("classical_bits", op.get("clbits", []))
            if isinstance(classical_bits, int):
                classical_bits = [classical_bits]
            return name, qubits, tuple(int(bit) for bit in classical_bits)
        params = tuple(float(v) for v in raw_params)
        return name, qubits, params

    if isinstance(op, (tuple, list)) and op:
        name = str(op[0]).lower()
        operands = list(op[1:])
        if name == "measure":
            if not operands:
                return name, (), ()
            measured = operands[0] if isinstance(operands[0], (tuple, list)) else operands[:1]
            classical = operands[1] if len(operands) > 1 and isinstance(operands[1], (tuple, list)) else ()
            return name, tuple(int(qubit) for qubit in measured), tuple(int(bit) for bit in classical)
        if name == "barrier":
            qubits = [qubit for value in operands for qubit in (value if isinstance(value, (tuple, list)) else [value])]
            return name, tuple(int(qubit) for qubit in qubits), ()
        if name == "reset":
            return name, tuple(int(qubit) for qubit in operands[:1]), ()

        try:
            qubit_count = get_gate_qubit_count(name)
        except (TypeError, ValueError):
            # Unknown operations have no registry arity; retain integer targets
            # as qubits and floating-point operands as parameters.
            qubits = tuple(int(value) for value in operands if isinstance(value, int) and not isinstance(value, bool))
            params = tuple(float(value) for value in operands if isinstance(value, float) or isinstance(value, bool))
            return name, qubits, params

        qubits = tuple(int(value) for value in operands[:qubit_count])
        params = tuple(float(value) for value in operands[qubit_count:])
        return name, qubits, params

    return "unknown", (), ()


def _signature_for_instruction(instr: Any) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    return _normalize_operation(instr)


def _operation_score(expected: Any, actual: Any) -> float:
    expected_sig = _signature_for_instruction(expected)
    actual_sig = _signature_for_instruction(actual)

    score = 0.0
    if expected_sig[0] != actual_sig[0]:
        score += 0.5
    if expected_sig[1] != actual_sig[1]:
        score += 0.25
    if expected_sig[2] != actual_sig[2]:
        score += 0.25
    return score


def _align_operations(expected: Sequence[Any], actual: Sequence[Any]) -> list[tuple[str, int | None, int | None]]:
    """Return an edit-distance alignment of normalized operation signatures."""
    expected_sigs = [_signature_for_instruction(op) for op in expected]
    actual_sigs = [_signature_for_instruction(op) for op in actual]
    rows, cols = len(expected_sigs) + 1, len(actual_sigs) + 1
    costs = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i, "delete"
    for j in range(1, cols):
        costs[0][j], moves[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if expected_sigs[i - 1] == actual_sigs[j - 1]:
                costs[i][j], moves[i][j] = costs[i - 1][j - 1], "match"
            else:
                choices = (
                    (costs[i - 1][j - 1] + 1, "substitute"),
                    (costs[i][j - 1] + 1, "insert"),
                    (costs[i - 1][j] + 1, "delete"),
                )
                costs[i][j], moves[i][j] = min(choices, key=lambda item: item[0])
    alignment = []
    i, j = len(expected_sigs), len(actual_sigs)
    while i or j:
        move = moves[i][j]
        if move in {"match", "substitute"}:
            alignment.append((move, i - 1, j - 1)); i -= 1; j -= 1
        elif move == "insert":
            alignment.append((move, None, j - 1)); j -= 1
        else:
            alignment.append((move, i - 1, None)); i -= 1
    return list(reversed(alignment))


def _infer_n_qubits(circuit: Sequence[Any]) -> int:
    max_index = 0
    for instruction in circuit:
        if isinstance(instruction, dict):
            qubits = instruction.get("qubits", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            for value in qubits:
                max_index = max(max_index, int(value))
            continue
        if isinstance(instruction, (tuple, list)) and instruction:
            if str(instruction[0]).lower() == "measure" and len(instruction) > 1 and isinstance(instruction[1], (tuple, list)):
                for value in instruction[1]:
                    max_index = max(max_index, int(value))
                continue
            for value in instruction[1:]:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    max_index = max(max_index, int(value))
    return max(1, max_index + 1)


class DiagnosticTraceStep(TypedDict):
    """One operation's evidence in a diagnostic trace."""

    step: int
    operation: str
    expected: Any
    actual: Any
    difference: float


DiagnosticTrace = tuple[float, list[DiagnosticTraceStep]]
"""Anomaly metric and per-step evidence produced by ``_trace_deviation``."""


def _trace_deviation(expected_circuit: Sequence[Any], actual_circuit: Sequence[Any], n_qubits: int) -> DiagnosticTrace:
    expected_sim = QuantumSimulator(n_qubits)
    actual_sim = QuantumSimulator(n_qubits)
    def without_measurements(circuit: Sequence[Any]) -> list[Any]:
        return [
            ("barrier", *(_normalize_operation(instruction)[1]))
            if _normalize_operation(instruction)[0] == "measure" else instruction
            for instruction in circuit
        ]

    expected_sim.run_circuit(without_measurements(expected_circuit))
    actual_sim.run_circuit(without_measurements(actual_circuit))

    expected_probs = expected_sim.get_probabilities()
    actual_probs = actual_sim.get_probabilities()
    comparison = compare_distributions(expected_probs, actual_probs, threshold=0.05)

    trace: list[DiagnosticTraceStep] = []
    for idx, actual_instr in enumerate(actual_circuit):
        if idx < len(expected_circuit):
            expected_instr = expected_circuit[idx]
        else:
            expected_instr = None
        trace.append({
            "step": idx,
            "operation": str(actual_instr[0]).lower() if isinstance(actual_instr, (tuple, list)) else "unknown",
            "expected": expected_instr,
            "actual": actual_instr,
            "difference": _operation_score(expected_instr, actual_instr) if expected_instr is not None else 1.0,
        })

    return comparison.total_variation_distance, trace


def localize_program_fault(
    expected_circuit: Sequence[Any],
    actual_circuit: Sequence[Any],
    *,
    threshold: float = 0.05,
    shots: int | None = None,
    precomputed_trace: DiagnosticTrace | None = None,
) -> FaultLocalizationResult:
    """Localize likely program-level faulty operations.

    The algorithm is deliberately simple but systematic:
    1. Compare operation signatures in sequence.
    2. Flag mismatched gates/parameters/insertions/deletions.
    3. If no direct mismatch is found, use the state-distribution deviation to
       identify which traced step contributed the most significant change.
    """

    if not isinstance(expected_circuit, Sequence):
        raise TypeError("expected_circuit must be a sequence of operations")
    if not isinstance(actual_circuit, Sequence):
        raise TypeError("actual_circuit must be a sequence of operations")

    n_qubits = max(_infer_n_qubits(expected_circuit), _infer_n_qubits(actual_circuit))

    candidates: List[LocalizationCandidate] = []
    trace_deviation_result: DiagnosticTrace | None = precomputed_trace
    for move, expected_idx, actual_idx in _align_operations(expected_circuit, actual_circuit):
        expected_instr = expected_circuit[expected_idx] if expected_idx is not None else None
        actual_instr = actual_circuit[actual_idx] if actual_idx is not None else None
        idx = actual_idx if actual_idx is not None else expected_idx

        if move == "insert":
            sig = _signature_for_instruction(actual_instr)
            candidates.append(LocalizationCandidate(
                index=idx,
                operation=sig[0],
                qubits=sig[1],
                expected_signature=None,
                actual_signature=sig,
                score=1.0,
                reason="Inserted operation relative to expected circuit",
                evidence=[f"Step {idx}: inserted {sig[0]} on qubits {sig[1]} without corresponding expected operation."],
            ))
            continue

        if move == "delete":
            sig = _signature_for_instruction(expected_instr)
            candidates.append(LocalizationCandidate(
                index=idx,
                operation=sig[0],
                qubits=sig[1],
                expected_signature=sig,
                actual_signature=None,
                score=1.0,
                reason="Expected operation missing in observed circuit",
                evidence=[f"Step {idx}: expected {sig[0]} on qubits {sig[1]} but it was not executed."],
            ))
            continue

        if move == "substitute":
            expected_sig = _signature_for_instruction(expected_instr)
            actual_sig = _signature_for_instruction(actual_instr)
            score = _operation_score(expected_instr, actual_instr)
            if score > 0:
                reason = "Operation signature mismatch"
                if expected_sig[0] != actual_sig[0]:
                    reason = "Gate operation differs from expected circuit"
                elif expected_sig[2] != actual_sig[2]:
                    reason = "Parameter values differ from expected circuit"
                elif expected_sig[1] != actual_sig[1]:
                    reason = "Target qubits differ from expected circuit"
                candidates.append(LocalizationCandidate(
                    index=idx,
                    operation=actual_sig[0],
                    qubits=actual_sig[1],
                    expected_signature=expected_sig,
                    actual_signature=actual_sig,
                    score=float(score),
                    reason=reason,
                    evidence=[
                        f"Step {idx}: expected {expected_sig[0]}{expected_sig[1]}{expected_sig[2]} but observed {actual_sig[0]}{actual_sig[1]}{actual_sig[2]}",
                        f"Mismatch score = {score:.2f}",
                    ],
                ))

    if trace_deviation_result is None:
        trace_deviation_result = _trace_deviation(expected_circuit, actual_circuit, n_qubits)

    if not candidates:
        anomaly_metric, trace = trace_deviation_result
        if anomaly_metric <= threshold:
            return FaultLocalizationResult(
                candidates=[],
                anomaly_metric=anomaly_metric,
                threshold=threshold,
                summary="No program-level fault was localized; the circuit behavior remains within threshold.",
                evidence=[f"Final distribution difference TVD={anomaly_metric:.6f} is within threshold={threshold:.6f}."],
            )

        best_step = max(range(len(trace)), key=lambda item: trace[item]["difference"])
        best = trace[best_step]
        name = str(best["operation"])
        qubits = tuple()
        if isinstance(best["actual"], (tuple, list)):
            qubits = tuple(int(q) for q in best["actual"][1:] if isinstance(q, (int, float)))
        candidates.append(LocalizationCandidate(
            index=best_step,
            operation=name,
            qubits=qubits,
            expected_signature=best["expected"],
            actual_signature=best["actual"],
            score=float(best["difference"]),
            reason="Largest traced deviation without direct operation mismatch",
            evidence=[
                f"Step {best_step}: operation {name} had the largest traced deviation.",
                f"Distribution deviation measured as TVD={anomaly_metric:.6f}.",
            ],
        ))

    candidates.sort(key=lambda item: item.score, reverse=True)
    evidence = []
    for candidate in candidates:
        evidence.extend(candidate.evidence)

    summary = (
        "Program fault localization identified the following likely offending operation(s): "
        + ", ".join(f"step {cand.index} ({cand.operation})" for cand in candidates[:3])
        if candidates else
        "No program-level fault candidates were identified above the configured threshold."
    )

    return FaultLocalizationResult(
        candidates=candidates,
        anomaly_metric=trace_deviation_result[0],
        threshold=threshold,
        summary=summary,
        evidence=evidence,
    )


__all__ = [
    "LocalizationCandidate",
    "FaultLocalizationResult",
    "DiagnosticTraceStep",
    "DiagnosticTrace",
    "localize_program_fault",
]
