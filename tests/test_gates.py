import numpy as np

from quantum_simulator.gates import (
    I,
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    SXdg,
    RX,
    RY,
    RZ,
    P,
    U1,
    U2,
    U3,
    CX,
    CZ,
    SWAP,
    CH,
    CP,
    CRX,
    CRY,
    CRZ,
    CCX,
    CSWAP,
    GATE_REGISTRY,
    GATE_QUBIT_COUNT,
    PARAMETERIZED_GATES,
    get_gate,
    get_gate_qubit_count,
    is_parameterized_gate,
    is_unitary,
)


def assert_unitary(matrix):
    identity = np.eye(matrix.shape[0], dtype=complex)
    result = matrix.conj().T @ matrix
    np.testing.assert_allclose(result, identity, atol=1e-12)


def test_basic_single_qubit_gates():
    for gate in (I, X, Y, Z, H, S, Sdg, T, Tdg, SX, SXdg):
        assert gate.shape == (2, 2)
        assert_unitary(gate)


def test_parameterized_gate_shapes():
    np.testing.assert_allclose(RX(np.pi / 2), RX(np.pi / 2))
    assert RX(np.pi / 2).shape == (2, 2)
    assert RY(np.pi / 2).shape == (2, 2)
    assert RZ(np.pi / 2).shape == (2, 2)
    assert P(np.pi / 3).shape == (2, 2)
    assert U1(np.pi / 3).shape == (2, 2)
    assert U2(0.2, 0.3).shape == (2, 2)
    assert U3(0.5, 0.2, 0.3).shape == (2, 2)


def test_two_qubit_gate_shapes():
    for gate in (CX, CZ, SWAP, CH, CP(np.pi / 4), CRX(np.pi / 3), CRY(np.pi / 3), CRZ(np.pi / 3)):
        assert gate.shape == (4, 4)
        assert_unitary(gate)


def test_three_qubit_gate_shapes():
    for gate in (CCX, CSWAP):
        assert gate.shape == (8, 8)
        assert_unitary(gate)


def test_gate_registry():
    assert "h" in GATE_REGISTRY
    assert "cx" in GATE_REGISTRY
    assert "ccx" in GATE_REGISTRY
    assert "sx" in GATE_REGISTRY
    assert "rx" in GATE_REGISTRY
    assert get_gate_qubit_count("cx") == 2
    assert get_gate_qubit_count("ccx") == 3
    assert get_gate_qubit_count("rx") == 1
    assert is_parameterized_gate("rx") is True
    assert is_parameterized_gate("h") is False
    assert is_unitary("cx") is True


def test_known_gate_effects():
    np.testing.assert_allclose(H @ np.array([1, 0], dtype=complex), np.array([1, 1], dtype=complex) / np.sqrt(2))
    np.testing.assert_allclose(X @ np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))
    np.testing.assert_allclose(Z @ np.array([0, 1], dtype=complex), np.array([0, -1], dtype=complex))


def test_gate_lookup():
    assert get_gate("h") is H
    assert get_gate("x") is X
    assert get_gate("cx") is CX
    assert get_gate("ccx") is CCX
