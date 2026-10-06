"""Minimal hardware-model layer for the quantum simulator.

This module is intentionally hardware-agnostic. It represents calibration data,
qubit quality metrics, gate properties, and connectivity without depending on
Qiskit or any specific IBM backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _validate_qubit_index(qubit: int, n_qubits: int | None = None) -> int:
    if not isinstance(qubit, int):
        raise TypeError(f"Qubit index must be an integer, got {type(qubit).__name__}")
    if n_qubits is not None and (qubit < 0 or qubit >= n_qubits):
        raise ValueError(f"Qubit index {qubit} is out of range for {n_qubits} qubits")
    return qubit


def _validate_probability(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in the range [0, 1]")
    return numeric


def _validate_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


@dataclass
class QubitProperties:
    qubit: int
    t1: float | None = 1e6
    t2: float | None = 1e6
    readout_error: float | None = 0.0
    status: str = "available"
    available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t1 = _validate_nonnegative(self.t1, "t1")
        self.t2 = _validate_nonnegative(self.t2, "t2")
        self.readout_error = _validate_probability(self.readout_error, "readout_error")
        if self.status not in {"available", "busy", "offline", "maintenance"}:
            self.status = "available"
        self.available = bool(self.available)


@dataclass
class GateProperties:
    gate: str
    qubits: Tuple[int, ...] = ()
    error: float | None = 0.0
    duration: float | None = 0.0
    supported: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.error = _validate_probability(self.error, "gate_error")
        self.duration = _validate_nonnegative(self.duration, "gate_duration")
        self.supported = bool(self.supported)
        if self.qubits is None:
            self.qubits = ()
        else:
            self.qubits = tuple(int(q) for q in self.qubits)


@dataclass
class CouplingEdge:
    qubit_a: int
    qubit_b: int
    direction: Optional[str] = None
    error: float = 0.0
    duration: float = 0.0
    supported: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.error = _validate_probability(self.error, "edge_error")
        self.duration = _validate_nonnegative(self.duration, "edge_duration")
        self.supported = bool(self.supported)
        if self.direction not in {None, "bidirectional", "forward", "reverse"}:
            self.direction = None

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.qubit_a, self.qubit_b)


@dataclass
class CalibrationSnapshot:
    backend_name: str = "generic_backend"
    timestamp: str | None = None
    qubits: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    gates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    edges: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")


class HardwareModel:
    """Generic calibration and topology model for the simulator."""

    def __init__(
        self,
        n_qubits: int,
        coupling_map: Sequence[Tuple[int, int]] | None = None,
        backend_name: str = "generic_backend",
        calibration_timestamp: str | None = None,
        default_t1: float = 1e6,
        default_t2: float = 1e6,
        default_readout_error: float = 0.0,
        default_gate_error: float = 0.0,
        default_gate_duration: float = 0.0,
    ) -> None:
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")

        self.n_qubits = n_qubits
        self.backend_name = backend_name
        self.calibration_timestamp = calibration_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")

        self._qubits: Dict[int, QubitProperties] = {}
        self._gates: Dict[str, List[GateProperties]] = {}
        self._coupling_edges: Dict[Tuple[int, int], CouplingEdge] = {}
        self._directional_connectivity: Dict[int, List[int]] = {q: [] for q in range(n_qubits)}

        for q in range(n_qubits):
            self._qubits[q] = QubitProperties(
                qubit=q,
                t1=default_t1,
                t2=default_t2,
                readout_error=default_readout_error,
                status="available",
                available=True,
            )

        if coupling_map is not None:
            for left, right in coupling_map:
                self.add_edge(left, right)

    def _validate_gate_name(self, gate_name: str) -> str:
        if not isinstance(gate_name, str):
            raise TypeError("Gate name must be a string")
        return gate_name.lower()

    def get_qubit_count(self) -> int:
        return self.n_qubits

    def set_qubit_properties(
        self,
        qubit: int,
        *,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        readout_error: Optional[float] = None,
        status: Optional[str] = None,
        available: Optional[bool] = None,
        **metadata: Any,
    ) -> QubitProperties:
        idx = _validate_qubit_index(qubit, self.n_qubits)
        props = self._qubits.setdefault(idx, QubitProperties(qubit=idx))
        if t1 is not None:
            props.t1 = _validate_nonnegative(t1, "t1")
        if t2 is not None:
            props.t2 = _validate_nonnegative(t2, "t2")
        if readout_error is not None:
            props.readout_error = _validate_probability(readout_error, "readout_error")
        if status is not None:
            props.status = status
        if available is not None:
            props.available = bool(available)
        if metadata:
            props.metadata.update(metadata)
        return props

    def get_qubit_properties(self, qubit: int) -> QubitProperties:
        idx = _validate_qubit_index(qubit, self.n_qubits)
        return self._qubits.setdefault(
            idx,
            QubitProperties(
                qubit=idx,
                t1=1e6,
                t2=1e6,
                readout_error=0.0,
                status="available",
                available=True,
            ),
        )

    def is_qubit_available(self, qubit: int) -> bool:
        return self.get_qubit_properties(qubit).available and self.get_qubit_properties(qubit).status == "available"

    def add_gate_calibration(
        self,
        gate_name: str,
        *,
        qubits: Sequence[int] = (),
        error: float = 0.0,
        duration: float = 0.0,
        supported: bool = True,
        **metadata: Any,
    ) -> GateProperties:
        normalized = self._validate_gate_name(gate_name)
        gate_cal = GateProperties(
            gate=normalized,
            qubits=tuple(int(q) for q in qubits),
            error=error,
            duration=duration,
            supported=supported,
            metadata=dict(metadata),
        )
        self._gates.setdefault(normalized, []).append(gate_cal)
        return gate_cal

    def get_gate_calibration(self, gate_name: str, qubits: Sequence[int] | None = None) -> List[GateProperties]:
        normalized = self._validate_gate_name(gate_name)
        entries = self._gates.get(normalized, [])
        if qubits is None:
            return list(entries)
        target = tuple(int(q) for q in qubits)
        return [entry for entry in entries if entry.qubits == target]

    def gate_supported(self, gate_name: str, qubits: Sequence[int] | None = None) -> bool:
        calib = self.get_gate_calibration(gate_name, qubits=qubits)
        if not calib:
            return True
        return any(entry.supported for entry in calib)

    def gate_error(self, gate_name: str, qubits: Sequence[int] | None = None) -> float:
        calib = self.get_gate_calibration(gate_name, qubits=qubits)
        errors = [entry.error for entry in calib if entry.error is not None]
        if not errors:
            return 0.0
        return min(errors)

    def gate_duration(self, gate_name: str, qubits: Sequence[int] | None = None) -> float:
        calib = self.get_gate_calibration(gate_name, qubits=qubits)
        durations = [entry.duration for entry in calib if entry.duration is not None]
        if not durations:
            return 0.0
        return min(durations)

    def add_edge(
        self,
        qubit_a: int,
        qubit_b: int,
        *,
        direction: Optional[str] = None,
        error: float = 0.0,
        duration: float = 0.0,
        supported: bool = True,
        **metadata: Any,
    ) -> CouplingEdge:
        qa = _validate_qubit_index(qubit_a, self.n_qubits)
        qb = _validate_qubit_index(qubit_b, self.n_qubits)
        if qa == qb:
            raise ValueError("A qubit cannot be coupled to itself")
        pair = tuple(sorted((qa, qb)))
        edge = CouplingEdge(
            qubit_a=pair[0],
            qubit_b=pair[1],
            direction=direction,
            error=error,
            duration=duration,
            supported=supported,
            metadata=dict(metadata),
        )
        self._coupling_edges[pair] = edge
        if qa not in self._directional_connectivity:
            self._directional_connectivity[qa] = []
        if qb not in self._directional_connectivity:
            self._directional_connectivity[qb] = []
        if qb not in self._directional_connectivity[qa]:
            self._directional_connectivity[qa].append(qb)
        if qa not in self._directional_connectivity[qb]:
            self._directional_connectivity[qb].append(qa)
        return edge

    def get_coupling_map(self) -> List[Tuple[int, int]]:
        return sorted(self._coupling_edges.keys())

    def get_directional_connectivity(self, qubit: int) -> List[int]:
        idx = _validate_qubit_index(qubit, self.n_qubits)
        return list(self._directional_connectivity.get(idx, []))

    def get_edge(self, qubit_a: int, qubit_b: int) -> Optional[CouplingEdge]:
        qa = _validate_qubit_index(qubit_a, self.n_qubits)
        qb = _validate_qubit_index(qubit_b, self.n_qubits)
        return self._coupling_edges.get(tuple(sorted((qa, qb))))

    def to_calibration_snapshot(self) -> CalibrationSnapshot:
        snapshot = CalibrationSnapshot(
            backend_name=self.backend_name,
            timestamp=self.calibration_timestamp,
            qubits={
                q: {
                    "t1": props.t1,
                    "t2": props.t2,
                    "readout_error": props.readout_error,
                    "status": props.status,
                    "available": props.available,
                    **props.metadata,
                }
                for q, props in self._qubits.items()
            },
            gates={
                name: [
                    {
                        "qubits": list(entry.qubits),
                        "error": entry.error,
                        "duration": entry.duration,
                        "supported": entry.supported,
                        **entry.metadata,
                    }
                    for entry in entries
                ]
                for name, entries in self._gates.items()
            },
            edges={
                edge.pair: {
                    "qubits": [edge.qubit_a, edge.qubit_b],
                    "direction": edge.direction,
                    "error": edge.error,
                    "duration": edge.duration,
                    "supported": edge.supported,
                    **edge.metadata,
                }
                for edge in self._coupling_edges.values()
            },
            metadata={"n_qubits": self.n_qubits},
        )
        return snapshot

    def from_snapshot(self, snapshot: CalibrationSnapshot) -> None:
        if not isinstance(snapshot, CalibrationSnapshot):
            raise TypeError("snapshot must be a CalibrationSnapshot")
        self.backend_name = snapshot.backend_name
        self.calibration_timestamp = snapshot.timestamp
        self._qubits = {}
        self._gates = {}
        self._coupling_edges = {}
        self._directional_connectivity = {q: [] for q in range(self.n_qubits)}

        for qubit, info in snapshot.qubits.items():
            q = _validate_qubit_index(int(qubit), self.n_qubits)
            self._qubits[q] = QubitProperties(
                qubit=q,
                t1=info.get("t1", 1e6),
                t2=info.get("t2", 1e6),
                readout_error=info.get("readout_error", 0.0),
                status=info.get("status", "available"),
                available=bool(info.get("available", True)),
                metadata={k: v for k, v in info.items() if k not in {"t1", "t2", "readout_error", "status", "available"}},
            )

        for gate_name, entries in snapshot.gates.items():
            for entry in entries:
                self.add_gate_calibration(
                    gate_name,
                    qubits=entry.get("qubits", ()),
                    error=entry.get("error", 0.0),
                    duration=entry.get("duration", 0.0),
                    supported=entry.get("supported", True),
                    **{k: v for k, v in entry.items() if k not in {"qubits", "error", "duration", "supported"}},
                )

        for pair, info in snapshot.edges.items():
            qa, qb = pair
            self.add_edge(
                qa,
                qb,
                direction=info.get("direction"),
                error=info.get("error", 0.0),
                duration=info.get("duration", 0.0),
                supported=info.get("supported", True),
                **{k: v for k, v in info.items() if k not in {"qubits", "direction", "error", "duration", "supported"}},
            )


__all__ = [
    "QubitProperties",
    "GateProperties",
    "CouplingEdge",
    "CalibrationSnapshot",
    "HardwareModel",
]
