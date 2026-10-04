"""User-facing circuit construction helpers."""

from __future__ import annotations

from typing import Any


class QuantumCircuit:
    """Build a sequence of instructions for :class:`QuantumSimulator`."""

    def __init__(self, n_qubits: int):
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        self.n_qubits = n_qubits
        self.instructions: list[tuple[Any, ...]] = []

    def _validate_qubit(self, qubit: int) -> None:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise ValueError(f"Qubit index must be an integer, got {qubit!r}")
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range for {self.n_qubits} qubits"
            )

    def _validate_distinct_qubits(self, *qubits: int) -> None:
        for qubit in qubits:
            self._validate_qubit(qubit)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Multi-qubit operation requires distinct qubit indices")

    def _append_single_qubit(self, gate: str, qubit: int) -> None:
        self._validate_qubit(qubit)
        self.instructions.append((gate, qubit))

    def h(self, qubit: int) -> None:
        self._append_single_qubit("h", qubit)

    def x(self, qubit: int) -> None:
        self._append_single_qubit("x", qubit)

    def y(self, qubit: int) -> None:
        self._append_single_qubit("y", qubit)

    def z(self, qubit: int) -> None:
        self._append_single_qubit("z", qubit)

    def s(self, qubit: int) -> None:
        self._append_single_qubit("s", qubit)

    def sdg(self, qubit: int) -> None:
        self._append_single_qubit("sdg", qubit)

    def t(self, qubit: int) -> None:
        self._append_single_qubit("t", qubit)

    def tdg(self, qubit: int) -> None:
        self._append_single_qubit("tdg", qubit)

    def sx(self, qubit: int) -> None:
        self._append_single_qubit("sx", qubit)

    def sxdg(self, qubit: int) -> None:
        self._append_single_qubit("sxdg", qubit)

    def _append_rotation(self, gate: str, qubit: int, theta: float) -> None:
        self._validate_qubit(qubit)
        self.instructions.append((gate, qubit, theta))

    def rx(self, qubit: int, theta: float) -> None:
        self._append_rotation("rx", qubit, theta)

    def ry(self, qubit: int, theta: float) -> None:
        self._append_rotation("ry", qubit, theta)

    def rz(self, qubit: int, theta: float) -> None:
        self._append_rotation("rz", qubit, theta)

    def cx(self, control: int, target: int) -> None:
        self._validate_distinct_qubits(control, target)
        self.instructions.append(("cx", control, target))

    def cz(self, control: int, target: int) -> None:
        self._validate_distinct_qubits(control, target)
        self.instructions.append(("cz", control, target))

    def swap(self, qubit1: int, qubit2: int) -> None:
        self._validate_distinct_qubits(qubit1, qubit2)
        self.instructions.append(("swap", qubit1, qubit2))

    def ccx(self, control1: int, control2: int, target: int) -> None:
        self._validate_distinct_qubits(control1, control2, target)
        self.instructions.append(("ccx", control1, control2, target))

    def reset(self, qubit: int) -> None:
        self._validate_qubit(qubit)
        self.instructions.append(("reset", qubit))

    def barrier(self) -> None:
        self.instructions.append(("barrier",))

    def measure(self, qubit: int, classical_bit: int | None = None) -> None:
        self._validate_qubit(qubit)
        if classical_bit is None:
            self.instructions.append(("measure", qubit))
            return
        if isinstance(classical_bit, bool) or not isinstance(classical_bit, int) or classical_bit < 0:
            raise ValueError(f"Classical bit index must be a non-negative integer, got {classical_bit!r}")
        self.instructions.append(("measure", qubit, [classical_bit]))

    def measure_all(self) -> None:
        self.instructions.append(("measure", list(range(self.n_qubits))))
