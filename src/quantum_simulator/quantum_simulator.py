"""Phase B: universal statevector simulator.

This simulator intentionally keeps a simple, explicit API while supporting the
legacy tuple-based instruction format expected in the earlier Colab work.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

from .gates import (
    CH,
    CP,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CX,
    CZ,
    H,
    I,
    RX,
    RY,
    RZ,
    SWAP,
    X,
    Y,
    Z,
    get_gate,
    get_gate_qubit_count,
)
from .hardware_model import HardwareModel


class QuantumSimulator:
    """Statevector-based quantum simulator.

    Supports the standard universal gate set defined in the project gate layer,
    plus special operations for measurement, reset, and barrier.
    """

    def __init__(self, n_qubits: int, seed: int | None = None):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")

        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.history: List[str] = []
        self.measurement_results: List[Any] = []
        self.rng = np.random.default_rng(seed)

    def _validate_qubit(self, qubit: int) -> None:
        if not isinstance(qubit, int):
            raise TypeError(f"Qubit index must be int, got {type(qubit).__name__}")
        if qubit < 0 or qubit >= self.n:
            raise ValueError(f"Qubit index {qubit} out of range for {self.n} qubits")

    def _validate_qubits(self, qubits: Sequence[int]) -> List[int]:
        normalized = [int(q) for q in qubits]
        for q in normalized:
            self._validate_qubit(q)
        return normalized

    def _apply_local_gate(self, gate_matrix: np.ndarray, qubits: Sequence[int]) -> None:
        qubits = self._validate_qubits(qubits)
        if len(qubits) == 0:
            return

        matrix = np.asarray(gate_matrix, dtype=complex)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Gate matrix must be square")
        if matrix.shape[0] != 2 ** len(qubits):
            raise ValueError(
                f"Gate matrix dimension {matrix.shape} does not match qubit count {len(qubits)}"
            )

        new_state = np.zeros_like(self.state)
        for basis_index in range(self.dim):
            bits = list(f"{basis_index:0{self.n}b}")
            local_bits = [bits[q] for q in qubits]
            local_index = int("".join(local_bits), 2)
            for row_index in range(matrix.shape[0]):
                row_bits = list(f"{row_index:0{len(qubits)}b}")
                dest_bits = bits.copy()
                for bit_pos, bit_value in zip(qubits, row_bits):
                    dest_bits[bit_pos] = bit_value
                dest_index = int("".join(dest_bits), 2)
                new_state[dest_index] += matrix[row_index, local_index] * self.state[basis_index]

        self.state = new_state

    def _normalize_operation(self, instruction: Any) -> tuple[str, Any]:
        if isinstance(instruction, dict):
            name = str(instruction.get("name") or instruction.get("op") or instruction.get("gate"))
            qubits = instruction.get("qubits", [])
            params = instruction.get("parameters", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            return name.lower(), (qubits, params)

        if isinstance(instruction, (tuple, list)) and instruction:
            op = str(instruction[0]).lower()
            operands = list(instruction[1:])
            return op, operands

        raise ValueError(f"Unsupported instruction format: {instruction!r}")

    def _resolve_gate_matrix(self, name: str, params: Sequence[float] | None = None) -> np.ndarray:
        if name in {"measure", "reset", "barrier"}:
            raise ValueError(f"Special operation {name!r} must be handled separately")

        try:
            gate = get_gate(name)
        except ValueError as exc:
            raise ValueError(f"Unknown gate or operation: {name!r}") from exc

        if callable(gate):
            if params is None:
                raise ValueError(f"Gate {name!r} requires parameters")
            if len(params) == 0:
                raise ValueError(f"Gate {name!r} requires at least one parameter")
            return gate(*params)

        return gate

    def apply_gate(self, gate_name: str | np.ndarray, qubit: int | Sequence[int], *params: float) -> None:
        if isinstance(gate_name, np.ndarray):
            matrix = gate_name
            qubits = list(qubit) if isinstance(qubit, (tuple, list)) else [qubit]
            self._apply_local_gate(matrix, qubits)
            return

        if isinstance(qubit, (tuple, list)):
            qubits = self._validate_qubits(qubit)
            matrix = self._resolve_gate_matrix(str(gate_name), list(params))
            self._apply_local_gate(matrix, qubits)
            return

        self._validate_qubit(int(qubit))
        matrix = self._resolve_gate_matrix(str(gate_name), list(params))
        self._apply_local_gate(matrix, [int(qubit)])

    def apply_cnot(self, control: int, target: int) -> None:
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be distinct for CNOT")
        self._apply_local_gate(CX, [control, target])

    def apply_toffoli(self, control1: int, control2: int, target: int) -> None:
        self._validate_qubit(control1)
        self._validate_qubit(control2)
        self._validate_qubit(target)
        self._apply_local_gate(np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        ), [control1, control2, target])

    def barrier(self, *qubits: int) -> None:
        if not qubits:
            self.history.append("barrier")
            return
        normalized = self._validate_qubits(qubits)
        self.history.append(f"barrier({normalized})")

    def reset_qubit(self, qubit: int) -> None:
        self._validate_qubit(qubit)
        new_state = np.zeros_like(self.state)
        for basis_index in range(self.dim):
            bits = list(f"{basis_index:0{self.n}b}")
            if bits[qubit] == "0":
                new_state[basis_index] = self.state[basis_index]

        norm = np.linalg.norm(new_state)
        if norm == 0:
            new_state = np.zeros_like(self.state)
            zero_state = 0
            if self.n == 1:
                new_state[0] = 1.0
            else:
                zero_bits = "0" * self.n
                zero_state = int(zero_bits, 2)
                new_state[zero_state] = 1.0
        else:
            new_state = new_state / norm

        self.state = new_state
        self.history.append(f"reset({qubit})")

    def measure(self, qubits: int | Sequence[int] | None = None, classical_bits: Sequence[int] | None = None, readout_error: float | Sequence[float] | None = None) -> Any:
        if qubits is None:
            qubits = list(range(self.n))
        elif isinstance(qubits, int):
            qubits = [qubits]
        else:
            qubits = list(qubits)

        if classical_bits is None:
            classical_bits = []
        else:
            classical_bits = list(classical_bits)

        qubits = self._validate_qubits(qubits)
        if len(classical_bits) not in (0, len(qubits)):
            raise ValueError("The number of classical bits must match the number of measured qubits")

        full_probabilities = np.abs(self.state) ** 2
        full_index = int(self.rng.choice(self.dim, p=full_probabilities))
        full_bits = list(f"{full_index:0{self.n}b}")
        selected_bits = [full_bits[q] for q in qubits]
        result = "".join(selected_bits)

        if readout_error is not None:
            errors = ([float(readout_error)] * len(selected_bits)
                      if isinstance(readout_error, (int, float))
                      else [float(error) for error in readout_error])
            if len(errors) != len(selected_bits) or any(not 0 <= error <= 1 for error in errors):
                raise ValueError("readout_error must contain one probability per measured qubit")
            flipped = []
            for bit, error in zip(selected_bits, errors):
                if self.rng.random() < error:
                    flipped.append("1" if bit == "0" else "0")
                else:
                    flipped.append(bit)
            result = "".join(flipped)
            selected_bits = list(result)

        # Collapse the state to the observed measurement outcome.
        new_state = np.zeros_like(self.state)
        for basis_index in range(self.dim):
            bits = list(f"{basis_index:0{self.n}b}")
            observed = [bits[q] for q in qubits]
            if observed == selected_bits:
                new_state[basis_index] = self.state[basis_index]

        norm = np.linalg.norm(new_state)
        if norm > 0:
            self.state = new_state / norm
        else:
            self.state = np.zeros_like(self.state)
            zero_index = 0
            self.state[zero_index] = 1.0

        self.history.append(f"measure({qubits}) -> {result}")
        self.measurement_results.append(result)

        if classical_bits:
            return {classical_bit: int(bit) for classical_bit, bit in zip(classical_bits, selected_bits)}

        return result

    def get_probabilities(self) -> np.ndarray:
        probs = np.abs(self.state) ** 2
        return probs / probs.sum() if probs.sum() > 0 else probs

    def _apply_hardware_relaxation(
        self, qubits: Sequence[int], duration: float, hardware_model: HardwareModel
    ) -> None:
        """Apply T1 relaxation and the T2-only dephasing over a gate duration."""
        if duration <= 0:
            return
        for qubit in qubits:
            props = hardware_model.get_qubit_properties(qubit)
            t1, t2 = props.t1, props.t2
            # Calibration fields can be absent or malformed in imported models.
            # Only finite, positive times are meaningful for this evolution.
            valid_t1 = isinstance(t1, (int, float, np.number)) and np.isfinite(t1) and t1 > 0
            valid_t2 = isinstance(t2, (int, float, np.number)) and np.isfinite(t2) and t2 > 0
            gamma = float(-np.expm1(-duration / t1)) if valid_t1 else 0.0

            if gamma > 0:
                zero_indices = []
                one_indices = []
                for index in range(self.dim):
                    bits = f"{index:0{self.n}b}"
                    (one_indices if bits[qubit] == "1" else zero_indices).append(index)
                old_state = self.state.copy()
                jump_probability = gamma * float(np.sum(np.abs(old_state[one_indices]) ** 2))
                if self.rng.random() < jump_probability:
                    self.state.fill(0)
                    self.state[zero_indices] = np.sqrt(gamma) * old_state[one_indices]
                    self.state /= np.linalg.norm(self.state)
                else:
                    self.state[one_indices] *= np.sqrt(1.0 - gamma)
                    self.state /= np.linalg.norm(self.state)

            # T2 includes the coherence loss already caused by T1. Apply only
            # the remaining pure-dephasing rate when it is physically positive.
            t1_rate = 1.0 / (2.0 * t1) if valid_t1 else 0.0
            t2_rate = 1.0 / t2 if valid_t2 else 0.0
            pure_dephasing_rate = max(0.0, t2_rate - t1_rate)
            if pure_dephasing_rate > 0:
                phase_flip_probability = 0.5 * float(-np.expm1(-duration * pure_dephasing_rate))
                if self.rng.random() < phase_flip_probability:
                    self._apply_local_gate(Z, [qubit])

    def run_circuit(
        self,
        instructions: Iterable[Any],
        *,
        readout_error: float | None = None,
        hardware_model: HardwareModel | None = None,
    ) -> List[Any]:
        if hardware_model is not None:
            if not isinstance(hardware_model, HardwareModel):
                raise TypeError("hardware_model must be a HardwareModel instance or None")
            if hardware_model.n_qubits != self.n:
                raise ValueError("hardware_model qubit count must match the simulator")

        results: List[Any] = []
        for instruction in instructions:
            op_name, operands = self._normalize_operation(instruction)
            gate_qubits: tuple[int, ...] = ()
            calibration_name = ""
            gate_duration = 0.0
            gate_failed = False

            if hardware_model is not None and op_name not in {"measure", "reset", "barrier"}:
                if op_name in {"cx", "cnot", "cz", "swap", "ch"}:
                    gate_qubits = tuple(int(q) for q in operands[:2])
                elif op_name in {"ccx", "toffoli", "cswap", "fredkin"}:
                    gate_qubits = tuple(int(q) for q in operands[:3])
                else:
                    gate_qubits = tuple(int(q) for q in operands[:1])
                calibration_name = {"cnot": "cx", "id": "i", "identity": "i"}.get(op_name, op_name)
                gate_error = hardware_model.gate_error(calibration_name, gate_qubits)
                gate_duration = hardware_model.gate_duration(calibration_name, gate_qubits)
                # A calibrated gate error represents a chance this operation fails
                # to execute. T1/T2 relaxation still accrues over its duration.
                gate_failed = self.rng.random() < gate_error
                if gate_failed:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                    continue

            if op_name in {"measure"}:
                operands = list(operands)
                measured_qubits = list(range(self.n)) if not operands else (
                    list(operands[0]) if isinstance(operands[0], (list, tuple)) else [operands[0]]
                )
                operation_readout_error = readout_error
                if hardware_model is not None:
                    operation_readout_error = [
                        hardware_model.get_qubit_properties(int(qubit)).readout_error
                        for qubit in measured_qubits
                    ]
                if not operands:
                    result = self.measure(readout_error=operation_readout_error)
                elif isinstance(operands[0], (list, tuple)):
                    qubits = operands[0]
                    classical_bits = operands[1] if len(operands) > 1 else None
                    result = self.measure(qubits=qubits, classical_bits=classical_bits, readout_error=operation_readout_error)
                else:
                    qubits = operands[0]
                    classical_bits = operands[1] if len(operands) > 1 else None
                    result = self.measure(qubits=qubits, classical_bits=classical_bits, readout_error=operation_readout_error)
                results.append(result)
                continue

            if op_name in {"reset"}:
                qubit = operands[0]
                self.reset_qubit(qubit)
                continue

            if op_name in {"barrier"}:
                self.barrier(*operands)
                continue

            if op_name in {"i", "id", "identity", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"}:
                if len(operands) != 1:
                    raise ValueError(f"Gate {op_name!r} expects exactly one qubit operand")
                self.apply_gate(op_name, operands[0])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"rx", "ry", "rz", "p", "u1", "cp", "crx", "cry", "crz"}:
                if len(operands) != 2:
                    raise ValueError(f"Gate {op_name!r} expects a qubit and a parameter")
                qubit = operands[0]
                theta = float(operands[1])
                self.apply_gate(op_name, qubit, theta)
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"u2", "u3"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects a qubit and two parameters")
                qubit = operands[0]
                params = [float(v) for v in operands[1:]]
                self.apply_gate(op_name, qubit, *params)
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"cx", "cnot", "cz", "swap", "ch"}:
                if len(operands) != 2:
                    raise ValueError(f"Gate {op_name!r} expects two qubit operands")
                if op_name in {"cx", "cnot"}:
                    self.apply_cnot(int(operands[0]), int(operands[1]))
                elif op_name == "cz":
                    self._apply_local_gate(CZ, [int(operands[0]), int(operands[1])])
                elif op_name == "swap":
                    self._apply_local_gate(SWAP, [int(operands[0]), int(operands[1])])
                elif op_name == "ch":
                    self._apply_local_gate(CH, [int(operands[0]), int(operands[1])])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"ccx", "toffoli"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects three qubit operands")
                self.apply_toffoli(int(operands[0]), int(operands[1]), int(operands[2]))
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"cswap", "fredkin"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects three qubit operands")
                self._apply_local_gate(CSWAP, [int(operands[0]), int(operands[1]), int(operands[2])])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            raise ValueError(f"Unsupported operation in circuit: {op_name!r}")

        return results

    def run_shots(self, instructions: Iterable[Any], shots: int = 1000, *, seed: int | None = None) -> dict[str, int]:
        if shots <= 0:
            raise ValueError("shots must be positive")

        counts: dict[str, int] = {}
        rng = np.random.default_rng(seed)

        for _ in range(shots):
            sim = QuantumSimulator(self.n, seed=int(rng.integers(0, 2 ** 31 - 1)))
            sim.run_circuit(instructions)
            if sim.measurement_results:
                last = sim.measurement_results[-1]
            else:
                outcome = int(np.argmax(sim.get_probabilities()))
                last = f"{outcome:0{self.n}b}"
            counts[last] = counts.get(last, 0) + 1

        return counts

    def run_circuit_traced(self, correct_instructions: Iterable[Any], actual_instructions: Iterable[Any]):
        clean = QuantumSimulator(self.n)
        noisy = QuantumSimulator(self.n)

        clean.run_circuit(correct_instructions)
        noisy.run_circuit(actual_instructions)

        clean_states = [clean.state.copy()]
        noisy_states = [noisy.state.copy()]
        labels = []
        trace = []

        for instruction in actual_instructions:
            op_name, operands = self._normalize_operation(instruction)
            labels.append(op_name)
            clean_states.append(clean.state.copy())
            noisy_states.append(noisy.state.copy())
            trace.append({
                "operation": op_name,
                "operands": operands,
                "deviation": float(np.linalg.norm(noisy.state - clean.state)),
            })

        return clean_states, noisy_states, labels, trace

    def __repr__(self) -> str:
        return f"QuantumSimulator(n_qubits={self.n}, state_norm={np.linalg.norm(self.state):.6f})"


__all__ = [
    "QuantumSimulator",
]
