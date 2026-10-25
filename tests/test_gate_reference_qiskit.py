"""Current-tree gate comparisons against Qiskit's independent circuit implementation."""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_simulator.quantum_simulator import QuantumSimulator


CASES = [
    ("id", "id", (), 1), ("x", "x", (), 1), ("y", "y", (), 1),
    ("z", "z", (), 1), ("h", "h", (), 1), ("s", "s", (), 1),
    ("sdg", "sdg", (), 1), ("t", "t", (), 1), ("tdg", "tdg", (), 1),
    ("sx", "sx", (), 1), ("sxdg", "sxdg", (), 1),
    ("rx", "rx", (0.371,), 1), ("ry", "ry", (-0.613,), 1),
    ("rz", "rz", (0.823,), 1), ("p", "p", (-0.287,), 1),
    ("u1", "p", (0.419,), 1), ("u2", "u", (0.23, -0.61), 1),
    ("u3", "u", (0.37, -0.22, 0.81), 1), ("u", "u", (0.37, -0.22, 0.81), 1),
    ("cx", "cx", (), 2), ("cz", "cz", (), 2), ("swap", "swap", (), 2),
    ("ch", "ch", (), 2), ("cp", "cp", (0.73,), 2),
    ("crx", "crx", (-0.48,), 2), ("cry", "cry", (0.56,), 2),
    ("crz", "crz", (0.69,), 2), ("rxx", "rxx", (0.52,), 2),
    ("rzz", "rzz", (-0.44,), 2),
    ("ccx", "ccx", (), 3), ("rccx", "rccx", (), 3),
    ("cswap", "cswap", (), 3),
]


def _qiskit_state(n_qubits, logical_qubits, method, params, basis_index):
    circuit = QuantumCircuit(n_qubits)
    bits = f"{basis_index:0{n_qubits}b}"
    for logical_q, bit in enumerate(bits):
        if bit == "1":
            circuit.x(n_qubits - 1 - logical_q)
    physical_qubits = [n_qubits - 1 - q for q in logical_qubits]
    reference_params = (np.pi / 2, *params) if method == "u" and len(params) == 2 else params
    getattr(circuit, method)(*reference_params, *physical_qubits)
    return np.asarray(Statevector.from_instruction(circuit).data)


@pytest.mark.parametrize("simulator_name,reference_name,params,n_qubits", CASES)
def test_supported_gate_basis_transitions_match_qiskit(simulator_name, reference_name, params, n_qubits):
    operand_orders = [tuple(range(n_qubits))]
    if n_qubits == 2:
        operand_orders.append((1, 0))

    for logical_qubits in operand_orders:
        instruction = (simulator_name, *logical_qubits, *params)
        for basis_index in range(2**n_qubits):
            sim = QuantumSimulator(n_qubits)
            sim.state.fill(0)
            sim.state[basis_index] = 1
            sim.run_circuit([instruction])
            expected = _qiskit_state(n_qubits, logical_qubits, reference_name, params, basis_index)
            np.testing.assert_allclose(sim.state, expected, rtol=0.0, atol=1e-12)


def test_seeded_random_small_circuits_match_qiskit():
    """Compare composed circuits that were not used to build the gate oracle."""
    rng = np.random.default_rng(20261014)
    one_qubit_gates = ("h", "x", "y", "z", "s", "t", "sx", "rx", "ry", "rz", "p", "u")
    two_qubit_gates = ("cx", "cz", "swap", "ch", "cp", "crx", "cry", "crz", "rxx", "rzz")
    three_qubit_gates = ("ccx", "rccx", "cswap")

    for n_qubits in range(1, 5):
        for _ in range(8):
            simulator_instructions = []
            reference = QuantumCircuit(n_qubits)
            for _ in range(24):
                available = list(one_qubit_gates)
                if n_qubits >= 2:
                    available.extend(two_qubit_gates)
                if n_qubits >= 3:
                    available.extend(three_qubit_gates)
                name = str(rng.choice(available))
                arity = 1 if name in one_qubit_gates else (2 if name in two_qubit_gates else 3)
                logical_qubits = tuple(int(q) for q in rng.choice(n_qubits, size=arity, replace=False))
                physical_qubits = tuple(n_qubits - 1 - q for q in logical_qubits)

                if name in {"rx", "ry", "rz", "p"}:
                    params = (float(rng.uniform(-np.pi, np.pi)),)
                elif name == "u":
                    params = tuple(float(x) for x in rng.uniform(-np.pi, np.pi, size=3))
                elif name in {"cp", "crx", "cry", "crz", "rxx", "rzz"}:
                    params = (float(rng.uniform(-np.pi, np.pi)),)
                else:
                    params = ()

                simulator_instructions.append((name, *logical_qubits, *params))
                getattr(reference, name)(*params, *physical_qubits)

            sim = QuantumSimulator(n_qubits)
            sim.run_circuit(simulator_instructions)
            expected = np.asarray(Statevector.from_instruction(reference).data)
            np.testing.assert_allclose(sim.state, expected, rtol=0.0, atol=2e-12)
