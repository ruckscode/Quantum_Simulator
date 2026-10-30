"""Phase B: universal statevector simulator.

This simulator intentionally keeps a simple, explicit API while supporting the
legacy tuple-based instruction format expected in the earlier Colab work.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from numbers import Integral
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

    @staticmethod
    def estimate_working_array_bytes(n_qubits: int) -> int:
        """Estimate principal complex128/probability arrays for a circuit.

        The estimate reserves three statevector-sized blocks (48 bytes per
        amplitude), covering the state, a gate/reset output, and measurement
        probability temporaries. It excludes Python/NumPy/runtime overhead.
        """
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        n_qubits = int(n_qubits)
        max_addressable_qubits = (sys.maxsize // np.dtype(np.complex128).itemsize).bit_length() - 1
        if n_qubits > max_addressable_qubits:
            raise MemoryError("Requested statevector exceeds the platform addressable array size")
        statevector_bytes = np.dtype(np.complex128).itemsize * (1 << n_qubits)
        return 3 * statevector_bytes

    def __init__(
        self,
        n_qubits: int,
        seed: int | None = None,
        *,
        memory_budget_bytes: int | None = None,
    ):
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")

        self.n = int(n_qubits)
        estimated_bytes = self.estimate_working_array_bytes(self.n)
        if memory_budget_bytes is not None:
            if (
                isinstance(memory_budget_bytes, bool)
                or not isinstance(memory_budget_bytes, Integral)
                or memory_budget_bytes <= 0
            ):
                raise ValueError("memory_budget_bytes must be a positive integer or None")
            memory_budget_bytes = int(memory_budget_bytes)
            if estimated_bytes > memory_budget_bytes:
                raise MemoryError(
                    f"Estimated simulator working arrays require {estimated_bytes} bytes, "
                    f"exceeding the configured budget of {memory_budget_bytes} bytes"
                )
        self.memory_budget_bytes = memory_budget_bytes
        self.dim = 2 ** self.n
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.history: List[str] = []
        self.measurement_results: List[Any] = []
        self.rng = np.random.default_rng(seed)

    def _validate_qubit(self, qubit: int) -> None:
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise TypeError(f"Qubit index must be int, got {type(qubit).__name__}")
        qubit = int(qubit)
        if qubit < 0 or qubit >= self.n:
            raise ValueError(f"Qubit index {qubit} out of range for {self.n} qubits")

    def _validate_qubits(self, qubits: Sequence[int]) -> List[int]:
        normalized = []
        for qubit in qubits:
            self._validate_qubit(qubit)
            normalized.append(int(qubit))
        return normalized

    def _apply_local_gate(self, gate_matrix: np.ndarray, qubits: Sequence[int]) -> None:
        qubits = self._validate_qubits(qubits)
        if len(qubits) == 0:
            return
        if len(set(qubits)) != len(qubits):
            raise ValueError("Gate operands must refer to distinct qubits")

        matrix = np.asarray(gate_matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Gate matrix must be square")
        if matrix.shape[0] != 2 ** len(qubits):
            raise ValueError(
                f"Gate matrix dimension {matrix.shape} does not match qubit count {len(qubits)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Gate matrix must contain only finite values")
        if not np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), rtol=0.0, atol=1e-10):
            raise ValueError("Gate matrix must be unitary")

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
            try:
                finite_params = [float(value) for value in params]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Gate {name!r} parameters must be finite real numbers") from exc
            if not np.all(np.isfinite(finite_params)):
                raise ValueError(f"Gate {name!r} parameters must be finite real numbers")
            return gate(*finite_params)

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

        self._validate_qubit(qubit)
        matrix = self._resolve_gate_matrix(str(gate_name), list(params))
        self._apply_local_gate(matrix, [qubit])

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
        """Apply the reset channel using a seeded pure-state trajectory.

        The two Kraus branches are |0><0| and |0><1|. For an entangled
        input, one trajectory conditions the other qubits on the sampled
        branch; averaging trajectories reproduces the mixed channel output.
        """
        self._validate_qubit(qubit)
        bit_mask = 1 << (self.n - 1 - qubit)
        zero_indices = np.array([i for i in range(self.dim) if not (i & bit_mask)])
        one_indices = zero_indices | bit_mask
        zero_probability = float(np.sum(np.abs(self.state[zero_indices]) ** 2))
        one_probability = float(np.sum(np.abs(self.state[one_indices]) ** 2))
        total_probability = zero_probability + one_probability
        if not np.isfinite(total_probability) or total_probability <= 0:
            raise ValueError("Cannot reset an invalid zero-norm state")

        if one_probability <= 0:
            choose_one = False
        elif zero_probability <= 0:
            choose_one = True
        else:
            choose_one = self.rng.random() >= zero_probability / total_probability
        new_state = np.zeros_like(self.state)
        source_indices = one_indices if choose_one else zero_indices
        new_state[zero_indices] = self.state[source_indices]
        norm = np.linalg.norm(new_state)
        if norm <= 0:
            raise ValueError("Cannot reset an invalid zero-norm state")
        self.state = new_state / norm
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
        physical_bits = [full_bits[q] for q in qubits]
        reported_bits = physical_bits.copy()
        result = "".join(reported_bits)

        if readout_error is not None:
            errors = ([float(readout_error)] * len(physical_bits)
                      if isinstance(readout_error, (int, float))
                      else [float(error) for error in readout_error])
            if len(errors) != len(physical_bits) or any(not 0 <= error <= 1 for error in errors):
                raise ValueError("readout_error must contain one probability per measured qubit")
            flipped = []
            for bit, error in zip(physical_bits, errors):
                if self.rng.random() < error:
                    flipped.append("1" if bit == "0" else "0")
                else:
                    flipped.append(bit)
            reported_bits = flipped
            result = "".join(reported_bits)

        # Physical collapse follows the sampled quantum outcome. Readout noise
        # changes only the classical report, not the post-measurement state.
        new_state = np.zeros_like(self.state)
        for basis_index in range(self.dim):
            bits = list(f"{basis_index:0{self.n}b}")
            observed = [bits[q] for q in qubits]
            if observed == physical_bits:
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
            return {classical_bit: int(bit) for classical_bit, bit in zip(classical_bits, reported_bits)}

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
                if op_name in {"cx", "cnot", "cz", "swap", "ch", "rxx", "rzz", "cp", "crx", "cry", "crz"}:
                    gate_qubits = tuple(int(q) for q in operands[:2])
                elif op_name in {"ccx", "rccx", "toffoli", "cswap", "fredkin"}:
                    gate_qubits = tuple(int(q) for q in operands[:3])
                else:
                    gate_qubits = tuple(int(q) for q in operands[:1])
                calibration_name = {"cnot": "cx", "hadamard": "h", "id": "i", "identity": "i"}.get(op_name, op_name)
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

            if op_name in {"i", "id", "identity", "h", "hadamard", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"}:
                if len(operands) != 1:
                    raise ValueError(f"Gate {op_name!r} expects exactly one qubit operand")
                self.apply_gate(op_name, operands[0])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"rx", "ry", "rz", "p", "u1"}:
                if len(operands) != 2:
                    raise ValueError(f"Gate {op_name!r} expects a qubit and a parameter")
                qubit = operands[0]
                theta = float(operands[1])
                self.apply_gate(op_name, qubit, theta)
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"u2", "u3", "u"}:
                expected_operands = 3 if op_name == "u2" else 4
                if len(operands) != expected_operands:
                    parameter_count = expected_operands - 1
                    raise ValueError(f"Gate {op_name!r} expects one qubit and {parameter_count} parameters")
                qubit = operands[0]
                params = [float(v) for v in operands[1:]]
                self.apply_gate(op_name, qubit, *params)
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"rxx", "rzz"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects two qubit operands and one parameter")
                self.apply_gate(op_name, [operands[0], operands[1]], float(operands[2]))
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"cp", "crx", "cry", "crz"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects two qubit operands and one parameter")
                self.apply_gate(op_name, [operands[0], operands[1]], float(operands[2]))
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"cx", "cnot", "cz", "swap", "ch"}:
                if len(operands) != 2:
                    raise ValueError(f"Gate {op_name!r} expects two qubit operands")
                if op_name in {"cx", "cnot"}:
                    self.apply_cnot(operands[0], operands[1])
                elif op_name == "cz":
                    self._apply_local_gate(CZ, [operands[0], operands[1]])
                elif op_name == "swap":
                    self._apply_local_gate(SWAP, [operands[0], operands[1]])
                elif op_name == "ch":
                    self._apply_local_gate(CH, [operands[0], operands[1]])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name == "rccx":
                if len(operands) != 3:
                    raise ValueError("Gate 'rccx' expects three qubit operands")
                self._apply_local_gate(get_gate("rccx"), [operands[0], operands[1], operands[2]])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"ccx", "toffoli"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects three qubit operands")
                self.apply_toffoli(operands[0], operands[1], operands[2])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            if op_name in {"cswap", "fredkin"}:
                if len(operands) != 3:
                    raise ValueError(f"Gate {op_name!r} expects three qubit operands")
                self._apply_local_gate(CSWAP, [operands[0], operands[1], operands[2]])
                if hardware_model is not None:
                    self._apply_hardware_relaxation(gate_qubits, gate_duration, hardware_model)
                continue

            raise ValueError(f"Unsupported operation in circuit: {op_name!r}")

        return results

    def run_shots(
        self,
        instructions: Iterable[Any],
        shots: int = 1000,
        *,
        seed: int | None = None,
        hardware_model: HardwareModel | None = None,
    ) -> dict[str, int]:
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        shots = int(shots)
        # Materialize once so a generator describes the same circuit on every shot.
        instructions = list(instructions)

        counts: dict[str, int] = {}
        rng = np.random.default_rng(seed)

        for _ in range(shots):
            sim = QuantumSimulator(
                self.n,
                seed=int(rng.integers(0, 2 ** 31 - 1)),
                memory_budget_bytes=self.memory_budget_bytes,
            )
            sim.run_circuit(instructions, hardware_model=hardware_model)
            if sim.measurement_results:
                last = sim.measurement_results[-1]
            else:
                probabilities = sim.get_probabilities()
                outcome = int(sim.rng.choice(self.dim, p=probabilities))
                last = f"{outcome:0{self.n}b}"
            counts[last] = counts.get(last, 0) + 1

        return counts

    def run_circuit_traced(
        self,
        correct_instructions: Iterable[Any],
        actual_instructions: Iterable[Any],
        *,
        hardware_model: HardwareModel | None = None,
        seed: int | None = None,
    ):
        """Run aligned expected/actual operations and record state evidence.

        Trace rows follow edit-distance alignment, so an inserted or deleted
        operation cannot shift all later comparisons. Missing instructions are
        represented by ``None`` on their respective side.
        """
        clean = QuantumSimulator(self.n, seed=seed)
        noisy = QuantumSimulator(self.n, seed=seed)
        clean_states = [clean.state.copy()]
        noisy_states = [noisy.state.copy()]
        labels = []
        trace = []
        correct = list(correct_instructions)
        actual = list(actual_instructions)
        expected_sigs = [self._normalize_operation(op) for op in correct]
        actual_sigs = [self._normalize_operation(op) for op in actual]
        rows, cols = len(correct) + 1, len(actual) + 1
        costs = [[0] * cols for _ in range(rows)]
        moves = [[""] * cols for _ in range(rows)]
        for i in range(1, rows): costs[i][0], moves[i][0] = i, "delete"
        for j in range(1, cols): costs[0][j], moves[0][j] = j, "insert"
        for i in range(1, rows):
            for j in range(1, cols):
                if expected_sigs[i-1] == actual_sigs[j-1]:
                    costs[i][j], moves[i][j] = costs[i-1][j-1], "match"
                else:
                    choices = ((costs[i-1][j-1]+1, "substitute"), (costs[i][j-1]+1, "insert"), (costs[i-1][j]+1, "delete"))
                    costs[i][j], moves[i][j] = min(choices, key=lambda item: item[0])
        aligned = []
        i, j = len(correct), len(actual)
        while i or j:
            move = moves[i][j]
            if move in {"match", "substitute"}:
                aligned.append((i-1, j-1)); i -= 1; j -= 1
            elif move == "insert":
                aligned.append((None, j-1)); j -= 1
            else:
                aligned.append((i-1, None)); i -= 1
        aligned.reverse()
        for step, (expected_idx, actual_idx) in enumerate(aligned):
            expected_instruction = correct[expected_idx] if expected_idx is not None else None
            actual_instruction = actual[actual_idx] if actual_idx is not None else None
            trace_instruction = actual_instruction if actual_instruction is not None else expected_instruction
            op_name, operands = self._normalize_operation(trace_instruction)
            metadata = {
                "resolved_calibration_name": None,
                "gate_error": None,
                "gate_duration": None,
                "gate_failed": None,
                "affected_hardware_qubits": None,
                "t1": None,
                "t2": None,
                "readout_error": None,
            }
            if expected_instruction is not None:
                clean.run_circuit([expected_instruction])
            if actual_instruction is not None:
                actual_name, actual_operands = self._normalize_operation(actual_instruction)
                metadata_qubits: tuple[int, ...] = ()
                calibration_name = None
                if hardware_model is not None:
                    if actual_name == "measure":
                        metadata_qubits = tuple(range(self.n)) if not actual_operands else tuple(
                            int(q) for q in (
                                actual_operands[0] if isinstance(actual_operands[0], (list, tuple)) else [actual_operands[0]]
                            )
                        )
                        metadata["readout_error"] = {
                            q: hardware_model.get_qubit_properties(q).readout_error for q in metadata_qubits
                        }
                    elif actual_name not in {"reset", "barrier"}:
                        if actual_name in {"cx", "cnot", "cz", "swap", "ch", "rxx", "rzz", "cp", "crx", "cry", "crz"}:
                            metadata_qubits = tuple(int(q) for q in actual_operands[:2])
                        elif actual_name in {"ccx", "rccx", "toffoli", "cswap", "fredkin"}:
                            metadata_qubits = tuple(int(q) for q in actual_operands[:3])
                        else:
                            metadata_qubits = tuple(int(q) for q in actual_operands[:1])
                        calibration_name = {"cnot": "cx", "id": "i", "identity": "i"}.get(actual_name, actual_name)
                        metadata["resolved_calibration_name"] = calibration_name
                        metadata["gate_error"] = hardware_model.gate_error(calibration_name, metadata_qubits)
                        metadata["gate_duration"] = hardware_model.gate_duration(calibration_name, metadata_qubits)
                        # Predict the exact failure draw, then restore RNG state so
                        # run_circuit consumes it once and execution remains unchanged.
                        rng_state = deepcopy(noisy.rng.bit_generator.state)
                        failed = noisy.rng.random() < metadata["gate_error"]
                        noisy.rng.bit_generator.state = rng_state
                        metadata["gate_failed"] = bool(failed)
                    if metadata_qubits:
                        metadata["affected_hardware_qubits"] = list(metadata_qubits)
                        if actual_name != "measure":
                            metadata["t1"] = {q: hardware_model.get_qubit_properties(q).t1 for q in metadata_qubits}
                            metadata["t2"] = {q: hardware_model.get_qubit_properties(q).t2 for q in metadata_qubits}
                noisy.run_circuit([actual_instruction], hardware_model=hardware_model)
            labels.append(op_name)
            clean_states.append(clean.state.copy())
            noisy_states.append(noisy.state.copy())
            difference = float(np.abs(np.abs(clean.state) ** 2 - np.abs(noisy.state) ** 2).sum() / 2)
            trace.append({
                "step": step,
                "operation": op_name,
                "operands": operands,
                "expected": expected_instruction,
                "actual": actual_instruction,
                "difference": difference,
                "probability_difference": difference,
                **metadata,
            })

        return clean_states, noisy_states, labels, trace

    def __repr__(self) -> str:
        return f"QuantumSimulator(n_qubits={self.n}, state_norm={np.linalg.norm(self.state):.6f})"


__all__ = [
    "QuantumSimulator",
]
