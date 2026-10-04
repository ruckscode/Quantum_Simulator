"""Hardware-fault localization for anomalous quantum circuit execution.

This layer is intentionally backend-independent and uses the existing
hardware_model calibration representation plus the circuit tracing signal from the
program-localization layer to isolate the most likely faulty hardware component.
The implementation does not classify program-vs-hardware faults or build a
final diagnosis engine; it simply identifies suspicious hardware entries with
clear evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

from .fault_localizer import localize_program_fault
from .hardware_model import HardwareModel


@dataclass
class HardwareCandidate:
    component_type: str
    component: Any
    score: float
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class HardwareFaultLocalizationResult:
    candidates: List[HardwareCandidate]
    anomaly_metric: float
    threshold: float
    summary: str
    evidence: List[str] = field(default_factory=list)


def _normalize_operation(operation: Any) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    if isinstance(operation, dict):
        name = str(operation.get("name") or operation.get("op") or operation.get("gate") or "unknown").lower()
        qubits = operation.get("qubits", [])
        if isinstance(qubits, int):
            qubits = [qubits]
        params = operation.get("parameters", [])
        return name, tuple(int(q) for q in qubits), tuple(float(v) for v in params)

    if isinstance(operation, (tuple, list)) and operation:
        name = str(operation[0]).lower()
        qubits: list[int] = []
        params: list[float] = []
        for value in operation[1:]:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                qubits.append(int(value))
            elif isinstance(value, float):
                params.append(float(value))
            elif isinstance(value, (tuple, list)):
                for subvalue in value:
                    if isinstance(subvalue, int):
                        qubits.append(int(subvalue))
                    elif isinstance(subvalue, float):
                        params.append(float(subvalue))
        return name, tuple(qubits), tuple(params)

    return "unknown", (), ()


def _extract_qubits(operation: Any) -> tuple[int, ...]:
    return _normalize_operation(operation)[1]


def _validate_circuit_against_model(model: HardwareModel, circuit: Sequence[Any] | None) -> None:
    if circuit is None:
        return
    for op in circuit:
        for qubit in _extract_qubits(op):
            if qubit < 0 or qubit >= model.n_qubits:
                raise ValueError(f"Qubit index {qubit} is out of range for {model.n_qubits} qubits")


def _score_qubit(model: HardwareModel, qubit: int, *, circuit_context: Sequence[Any] | None = None) -> tuple[float, str, List[str]]:
    props = model.get_qubit_properties(qubit)
    score = 0.0
    reasons: List[str] = []
    evidence: List[str] = []

    if not props.available or props.status != "available":
        score += 1.5
        reasons.append(f"qubit status={props.status!r}, available={props.available}")
        evidence.append(f"Qubit {qubit} is not available: status={props.status!r}.")

    if props.readout_error > 0.05:
        score += 4.0 * props.readout_error
        reasons.append(f"readout_error={props.readout_error:.4f}")
        evidence.append(f"Qubit {qubit} readout error is elevated at {props.readout_error:.4f}.")

    if props.t1 < 100.0:
        score += max(0.0, (100.0 - props.t1) / 100.0)
        reasons.append(f"t1={props.t1:.2f}")
        evidence.append(f"Qubit {qubit} T1 is short at {props.t1:.2f} microseconds.")

    if props.t2 < 100.0:
        score += max(0.0, (100.0 - props.t2) / 100.0)
        reasons.append(f"t2={props.t2:.2f}")
        evidence.append(f"Qubit {qubit} T2 is short at {props.t2:.2f} microseconds.")

    if circuit_context:
        relevant_ops = [op for op in circuit_context if qubit in _extract_qubits(op)]
        if relevant_ops:
            score += 0.5 * min(len(relevant_ops), 3)
            evidence.append(f"Qubit {qubit} appears in {len(relevant_ops)} relevant circuit operations.")

    reason = "; ".join(reasons) if reasons else "No notable qubit degradation was detected."
    return score, reason, evidence


def _score_edge(model: HardwareModel, left: int, right: int, *, circuit_context: Sequence[Any] | None = None) -> tuple[float, str, List[str]]:
    edge = model.get_edge(left, right)
    score = 0.0
    reasons: List[str] = []
    evidence: List[str] = []

    if edge is None:
        return score, "No calibrated coupling edge was found.", evidence

    if not edge.supported:
        score += 1.5
        reasons.append("edge unsupported")
        evidence.append(f"Coupling edge ({left}, {right}) is marked unsupported.")

    if edge.error > 0.05:
        score += 8.0 * edge.error
        reasons.append(f"edge_error={edge.error:.4f}")
        evidence.append(f"Coupling edge ({left}, {right}) has elevated error {edge.error:.4f}.")

    if edge.duration > 1.0:
        score += edge.duration / 10.0
        reasons.append(f"edge_duration={edge.duration:.4f}")
        evidence.append(f"Coupling edge ({left}, {right}) has long duration {edge.duration:.4f}.")

    if circuit_context:
        edge_ops = [op for op in circuit_context if _extract_qubits(op) == (left, right) or tuple(sorted(_extract_qubits(op))) == tuple(sorted((left, right)))]
        if edge_ops:
            score += 0.75 * min(len(edge_ops), 3)
            evidence.append(f"Coupling edge ({left}, {right}) participates in {len(edge_ops)} relevant operations.")

    reason = "; ".join(reasons) if reasons else "No notable edge degradation was detected."
    return score, reason, evidence


def _score_gate(model: HardwareModel, gate_name: str, qubits: Sequence[int]) -> tuple[float, str, List[str]]:
    entries = model.get_gate_calibration(gate_name, qubits=qubits)
    if not entries:
        return 0.0, "No gate calibration was available.", []

    best = max(entries, key=lambda item: item.error)
    score = 0.0
    reasons: List[str] = []
    evidence: List[str] = []

    if not best.supported:
        score += 1.5
        reasons.append("gate unsupported")
        evidence.append(f"Gate {gate_name} on qubits {tuple(qubits)} is marked unsupported.")

    if best.error > 0.05:
        score += 10.0 * best.error
        reasons.append(f"gate_error={best.error:.4f}")
        evidence.append(f"Gate {gate_name} on qubits {tuple(qubits)} has elevated calibration error {best.error:.4f}.")

    if best.duration > 1.0:
        score += best.duration / 10.0
        reasons.append(f"gate_duration={best.duration:.4f}")
        evidence.append(f"Gate {gate_name} on qubits {tuple(qubits)} has long duration {best.duration:.4f}.")

    reason = "; ".join(reasons) if reasons else "No notable gate calibration issue detected."
    return score, reason, evidence


def _relevant_gate_candidates(model: HardwareModel, qubits: Sequence[int]) -> list[tuple[str, tuple[int, ...], float, str, List[str]]]:
    if len(qubits) < 2:
        return []
    targets = tuple(qubits[:2])
    results: list[tuple[str, tuple[int, ...], float, str, List[str]]] = []
    for gate_name, entries in model._gates.items():
        for entry in entries:
            if tuple(entry.qubits) == targets:
                score, reason, evidence = _score_gate(model, gate_name, entry.qubits)
                results.append((gate_name, entry.qubits, score, reason, evidence))
    return results


def _impact_from_program_result(model: HardwareModel, expected_circuit: Sequence[Any] | None, actual_circuit: Sequence[Any] | None, threshold: float) -> tuple[list[HardwareCandidate], float]:
    if expected_circuit is None or actual_circuit is None:
        return [], 0.0

    program_result = localize_program_fault(expected_circuit, actual_circuit, threshold=threshold)
    anomaly_metric = program_result.anomaly_metric
    if anomaly_metric <= threshold and not program_result.candidates:
        return [], anomaly_metric

    candidates: List[HardwareCandidate] = []
    seen: dict[tuple[str, Any], HardwareCandidate] = {}

    for candidate in program_result.candidates:
        qubits = candidate.qubits or ()
        if qubits:
            for qubit in qubits:
                score, reason, evidence = _score_qubit(model, qubit, circuit_context=actual_circuit)
                score += 0.5
                hardware_candidate = HardwareCandidate(
                    component_type="qubit",
                    component=qubit,
                    score=score,
                    reason=f"Program trace implicated qubit {qubit}: {reason}",
                    evidence=[*evidence, f"Observed deviation at operation index {candidate.index}: {candidate.reason}"],
                )
                key = ("qubit", qubit)
                existing = seen.get(key)
                if existing is None or score > existing.score:
                    seen[key] = hardware_candidate

        if len(qubits) >= 2:
            left, right = qubits[:2]
            score, reason, evidence = _score_edge(model, left, right, circuit_context=actual_circuit)
            score += 0.5
            hardware_candidate = HardwareCandidate(
                component_type="edge",
                component=(left, right),
                score=score,
                reason=f"Program trace implicated coupling edge ({left}, {right}): {reason}",
                evidence=[*evidence, f"Observed deviation at operation index {candidate.index}: {candidate.reason}"],
            )
            key = ("edge", (left, right))
            existing = seen.get(key)
            if existing is None or score > existing.score:
                seen[key] = hardware_candidate

        if qubits:
            gate_targets = tuple(qubits[: min(2, len(qubits))])
            if len(gate_targets) >= 2:
                for gate_name, gate_qubits, score, reason, evidence in _relevant_gate_candidates(model, gate_targets):
                    score += 0.35
                    hardware_candidate = HardwareCandidate(
                        component_type="gate",
                        component=tuple(gate_qubits),
                        score=score,
                        reason=f"Gate calibration for {gate_name} on qubits {tuple(gate_qubits)}: {reason}",
                        evidence=[*evidence, f"Observed anomaly associated with operation {candidate.operation} at index {candidate.index}."],
                    )
                    key = ("gate", tuple(gate_qubits))
                    existing = seen.get(key)
                    if existing is None or score > existing.score:
                        seen[key] = hardware_candidate

            if not _relevant_gate_candidates(model, gate_targets):
                gate_name = str(candidate.operation).lower()
                score, reason, evidence = _score_gate(model, gate_name, gate_targets)
                score += 0.35
                hardware_candidate = HardwareCandidate(
                    component_type="gate",
                    component=gate_targets,
                    score=score,
                    reason=f"Gate calibration for {gate_name} on qubits {gate_targets}: {reason}",
                    evidence=[*evidence, f"Observed anomaly associated with operation {candidate.operation} at index {candidate.index}."],
                )
                key = ("gate", gate_targets)
                existing = seen.get(key)
                if existing is None or score > existing.score:
                    seen[key] = hardware_candidate

    candidates.extend(seen.values())
    return sorted(candidates, key=lambda item: item.score, reverse=True), anomaly_metric


def _hardware_model_base_candidates(model: HardwareModel) -> List[HardwareCandidate]:
    candidates: List[HardwareCandidate] = []
    seen: dict[tuple[str, Any], HardwareCandidate] = {}

    for qubit in range(model.n_qubits):
        score, reason, evidence = _score_qubit(model, qubit)
        if score <= 0.05:
            continue
        candidate = HardwareCandidate(
            component_type="qubit",
            component=qubit,
            score=score,
            reason=f"Qubit {qubit} shows hardware degradation: {reason}",
            evidence=evidence,
        )
        seen[("qubit", qubit)] = candidate

    for left, right in model.get_coupling_map():
        score, reason, evidence = _score_edge(model, left, right)
        if score <= 0.05:
            continue
        candidate = HardwareCandidate(
            component_type="edge",
            component=(left, right),
            score=score,
            reason=f"Coupling edge ({left}, {right}) shows calibration degradation: {reason}",
            evidence=evidence,
        )
        seen[("edge", (left, right))] = candidate

    for gate_name, entries in {}.items():
        _ = gate_name, entries

    for gate_name in set(model._gates.keys()):
        for entry in model.get_gate_calibration(gate_name):
            gate_qubits = tuple(entry.qubits)
            score, reason, evidence = _score_gate(model, gate_name, gate_qubits)
            if score <= 0.05:
                continue
            candidate = HardwareCandidate(
                component_type="gate",
                component=gate_qubits,
                score=score,
                reason=f"Gate calibration for {gate_name} on qubits {gate_qubits} is degraded: {reason}",
                evidence=evidence,
            )
            seen[("gate", gate_qubits)] = candidate

    candidates.extend(seen.values())
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def localize_hardware_fault(
    hardware_model: HardwareModel,
    expected_circuit: Sequence[Any] | None = None,
    actual_circuit: Sequence[Any] | None = None,
    *,
    threshold: float = 0.05,
) -> HardwareFaultLocalizationResult:
    """Localize likely hardware faults from calibration and execution anomalies.

    The returned candidates are ranked by their calibration-driven severity, with
    additional emphasis when the circuit trace indicates a suspicious operation on
    a specific qubit or edge.
    """

    if not isinstance(hardware_model, HardwareModel):
        raise TypeError("hardware_model must be a HardwareModel instance")

    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric")
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    _validate_circuit_against_model(hardware_model, expected_circuit)
    _validate_circuit_against_model(hardware_model, actual_circuit)

    base_candidates = _hardware_model_base_candidates(hardware_model)
    if expected_circuit is not None and actual_circuit is not None:
        circuit_candidates, anomaly_metric = _impact_from_program_result(
            hardware_model,
            expected_circuit,
            actual_circuit,
            threshold=threshold,
        )
    else:
        circuit_candidates, anomaly_metric = [], 0.0

    merged: dict[tuple[str, Any], HardwareCandidate] = {}
    for candidate in base_candidates:
        merged[(candidate.component_type, candidate.component)] = candidate
    for candidate in circuit_candidates:
        key = (candidate.component_type, candidate.component)
        existing = merged.get(key)
        if existing is None or candidate.score > existing.score:
            merged[key] = candidate

    candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    if not candidates:
        summary = "No hardware fault candidates exceeded the configured threshold."
        evidence = [
            f"No qubit, edge, or gate calibration metric crossed the threshold={threshold:.4f}.",
        ]
        return HardwareFaultLocalizationResult(
            candidates=[],
            anomaly_metric=anomaly_metric,
            threshold=threshold,
            summary=summary,
            evidence=evidence,
        )

    evidence: List[str] = []
    for candidate in candidates:
        evidence.extend(candidate.evidence)

    summary = (
        "Likely hardware component(s): "
        + ", ".join(
            f"{candidate.component_type} {candidate.component}"
            for candidate in candidates[:3]
        )
    )

    return HardwareFaultLocalizationResult(
        candidates=candidates,
        anomaly_metric=anomaly_metric,
        threshold=threshold,
        summary=summary,
        evidence=evidence,
    )


__all__ = [
    "HardwareCandidate",
    "HardwareFaultLocalizationResult",
    "localize_hardware_fault",
]
