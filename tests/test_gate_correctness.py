"""Analytical correctness checks for the public statevector simulator API."""

import numpy as np
import pytest

from quantum_simulator import QuantumSimulator


def probabilities(n_qubits, operations):
    """Apply public gate operations and return computational-basis probabilities."""
    simulator = QuantumSimulator(n_qubits)
    for name, qubits, parameters in operations:
        simulator.apply_gate(name, qubits, *parameters)
    return simulator.get_probabilities()


def assert_probabilities(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "name,operations,expected",
    [
        ("I", [("i", 0, ())], [1, 0]),
        ("X", [("x", 0, ())], [0, 1]),
        ("Y", [("y", 0, ())], [0, 1]),
        # These phase gates are tested in an interferometer, not on |0> alone.
        ("Z", [("h", 0, ()), ("z", 0, ()), ("h", 0, ())], [0, 1]),
        ("H", [("h", 0, ())], [0.5, 0.5]),
        ("S", [("h", 0, ()), ("s", 0, ()), ("h", 0, ())], [0.5, 0.5]),
        ("Sdg", [("h", 0, ()), ("sdg", 0, ()), ("h", 0, ())], [0.5, 0.5]),
        (
            "T",
            [("h", 0, ()), ("t", 0, ()), ("h", 0, ())],
            [np.cos(np.pi / 8) ** 2, np.sin(np.pi / 8) ** 2],
        ),
        (
            "Tdg",
            [("h", 0, ()), ("tdg", 0, ()), ("h", 0, ())],
            [np.cos(np.pi / 8) ** 2, np.sin(np.pi / 8) ** 2],
        ),
        ("SX", [("sx", 0, ())], [0.5, 0.5]),
        ("SXdg", [("sxdg", 0, ())], [0.5, 0.5]),
    ],
)
def test_single_qubit_gate(name, operations, expected):
    assert_probabilities(probabilities(1, operations), expected)


@pytest.mark.parametrize(
    "name,operations,expected",
    [
        ("RX(pi)", [("rx", 0, (np.pi,))], [0, 1]),
        ("RY(pi)", [("ry", 0, (np.pi,))], [0, 1]),
        (
            "RZ(pi)",
            [("h", 0, ()), ("rz", 0, (np.pi,)), ("h", 0, ())],
            [0, 1],
        ),
    ],
)
def test_parameterized_single_qubit_gate(name, operations, expected):
    assert_probabilities(probabilities(1, operations), expected)


@pytest.mark.parametrize(
    "name,operations,expected",
    [
        ("CX", [("x", 0, ()), ("cx", [0, 1], ())], [0, 0, 0, 1]),
        ("CZ", [("h", 0, ()), ("x", 1, ()), ("cz", [0, 1], ()), ("h", 0, ())], [0, 0, 0, 1]),
        ("SWAP", [("x", 1, ()), ("swap", [0, 1], ())], [0, 0, 1, 0]),
        ("CH", [("x", 0, ()), ("ch", [0, 1], ())], [0, 0, 0.5, 0.5]),
        (
            "CP",
            [("h", 0, ()), ("x", 1, ()), ("cp", [0, 1], (np.pi,)), ("h", 0, ())],
            [0, 0, 0, 1],
        ),
        ("CRX(pi)", [("x", 0, ()), ("x", 1, ()), ("crx", [0, 1], (np.pi,))], [0, 0, 1, 0]),
        ("CRY(pi)", [("x", 0, ()), ("x", 1, ()), ("cry", [0, 1], (np.pi,))], [0, 0, 1, 0]),
        (
            "CRZ(pi)",
            [("h", 0, ()), ("x", 1, ()), ("crz", [0, 1], (np.pi,)), ("h", 0, ())],
            [0, 0.5, 0, 0.5],
        ),
    ],
)
def test_two_qubit_gate(name, operations, expected):
    assert_probabilities(probabilities(2, operations), expected)


@pytest.mark.parametrize(
    "name,operations,expected",
    [
        ("CCX", [("x", 0, ()), ("x", 1, ()), ("ccx", [0, 1, 2], ())], [0, 0, 0, 0, 0, 0, 0, 1]),
        ("CSWAP", [("x", 0, ()), ("x", 2, ()), ("cswap", [0, 1, 2], ())], [0, 0, 0, 0, 0, 0, 1, 0]),
    ],
)
def test_three_qubit_gate(name, operations, expected):
    assert_probabilities(probabilities(3, operations), expected)


def test_rxx_zero_is_identity():
    from gates import RXX
    np.testing.assert_allclose(RXX(0), np.eye(4), rtol=0.0, atol=1e-12)


def test_rxx_pi_flips_both_bits():
    sim = QuantumSimulator(2)
    sim.run_circuit([("rxx", 0, 1, np.pi)])
    assert_probabilities(sim.get_probabilities(), [0, 0, 0, 1])


@pytest.mark.parametrize("theta", [0.23, 0.8, np.pi / 2])
def test_rxx_state_matches_qiskit_aer_when_available(theta):
    pytest.importorskip("qiskit_aer")
    try:
        from qiskit import QuantumCircuit as AerCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        pytest.skip("Qiskit Aer comparison dependencies are unavailable")
    ours = QuantumSimulator(2)
    ours.run_circuit([("rxx", 0, 1, theta)])
    ref = AerCircuit(2)
    ref.rxx(theta, 0, 1)
    ref.save_statevector()
    backend = AerSimulator(method="statevector")
    expected = backend.run(transpile(ref, backend)).result().get_statevector().data
    np.testing.assert_allclose(ours.state, expected, rtol=0.0, atol=1e-12)


def test_public_quantum_circuit_rxx_and_validation():
    from quantum_simulator import QuantumCircuit
    circuit = QuantumCircuit(2)
    circuit.rxx(np.pi / 3, 0, 1)
    assert circuit.instructions == [("rxx", 0, 1, np.pi / 3)]
    with pytest.raises(ValueError):
        circuit.rxx(0.2, 0, 0)
    with pytest.raises(ValueError):
        circuit.rxx(0.2, 0, 2)
    with pytest.raises((TypeError, ValueError)):
        circuit.rxx("bad", 0, 1)
