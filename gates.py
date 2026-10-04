"""Quantum gate library for the universal simulator.

This module intentionally keeps the gate definitions separate from simulator logic.
It exports both concrete matrices and a registry for lookup by canonical names
or common aliases.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Single-qubit fixed gates
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = S.conj().T
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
Tdg = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex)
SX = (1 / 2) * np.array(
    [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
    dtype=complex,
)
SXdg = SX.conj().T


# ---------------------------------------------------------------------------
# Parameterized single-qubit gates
# ---------------------------------------------------------------------------

def RX(theta):
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def RY(theta):
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def RZ(theta):
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ],
        dtype=complex,
    )


def P(theta):
    return np.diag([1, np.exp(1j * theta)]).astype(complex)


U1 = P


def U2(phi, lam):
    return (1 / np.sqrt(2)) * np.array(
        [
            [1, np.exp(-1j * lam)],
            [np.exp(1j * phi), -np.exp(1j * (phi + lam))],
        ],
        dtype=complex,
    )


def U3(theta, phi, lam):
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------

CX = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

CZ = np.diag([1, 1, 1, -1]).astype(complex)

SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)

CH = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
    ],
    dtype=complex,
)


def CP(theta):
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def CRX(theta):
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, s],
            [0, 0, s, c],
        ],
        dtype=complex,
    )


def CRY(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c],
        ],
        dtype=complex,
    )


def CRZ(theta):
    return np.diag([1, 1, np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


# ---------------------------------------------------------------------------
# Three-qubit gates
# ---------------------------------------------------------------------------

def CCX():
    """Return the standard Toffoli gate matrix on 3 qubits."""
    matrix = np.zeros((8, 8), dtype=complex)
    for idx in range(8):
        bits = list(f"{idx:03b}")
        if bits[0] == "1" and bits[1] == "1":
            bits[2] = "1" if bits[2] == "0" else "0"
        target_idx = int("".join(bits), 2)
        matrix[target_idx, idx] = 1.0
    return matrix


CCX = CCX()


def CSWAP():
    """Return the Fredkin gate matrix on 3 qubits."""
    matrix = np.zeros((8, 8), dtype=complex)
    for idx in range(8):
        bits = list(f"{idx:03b}")
        # Control is qubit 0 (leftmost bit in the computational basis string)
        if bits[0] == "1":
            swapped = bits[1:]
            swapped[0], swapped[1] = swapped[1], swapped[0]
            bits[1:] = swapped
        target_idx = int("".join(bits), 2)
        matrix[target_idx, idx] = 1.0
    return matrix


CSWAP = CSWAP()


# ---------------------------------------------------------------------------
# Registry and metadata
# ---------------------------------------------------------------------------

GATE_QUBIT_COUNT = {
    "i": 1,
    "id": 1,
    "identity": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "sx": 1,
    "sxdg": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "p": 1,
    "u1": 1,
    "u2": 1,
    "u3": 1,
    "cx": 2,
    "cnot": 2,
    "cz": 2,
    "swap": 2,
    "ch": 2,
    "cp": 2,
    "crx": 2,
    "cry": 2,
    "crz": 2,
    "ccx": 3,
    "toffoli": 3,
    "cswap": 3,
    "fredkin": 3,
}

PARAMETERIZED_GATES = {
    "rx": RX,
    "ry": RY,
    "rz": RZ,
    "p": P,
    "u1": U1,
    "u2": U2,
    "u3": U3,
    "cp": CP,
    "crx": CRX,
    "cry": CRY,
    "crz": CRZ,
}

GATE_REGISTRY = {
    "i": I,
    "id": I,
    "identity": I,
    "x": X,
    "y": Y,
    "z": Z,
    "h": H,
    "hadamard": H,
    "s": S,
    "sdg": Sdg,
    "t": T,
    "tdg": Tdg,
    "sx": SX,
    "sxdg": SXdg,
    "rx": RX,
    "ry": RY,
    "rz": RZ,
    "p": P,
    "u1": U1,
    "u2": U2,
    "u3": U3,
    "cx": CX,
    "cnot": CX,
    "cz": CZ,
    "swap": SWAP,
    "ch": CH,
    "cp": CP,
    "crx": CRX,
    "cry": CRY,
    "crz": CRZ,
    "ccx": CCX,
    "toffoli": CCX,
    "cswap": CSWAP,
    "fredkin": CSWAP,
}


def _normalize_gate_name(name):
    if not isinstance(name, str):
        raise TypeError(f"Gate name must be a string, received {type(name).__name__}")
    return name.strip().lower()


def get_gate(name):
    """Return the matrix or constructor for a registered gate."""
    normalized = _normalize_gate_name(name)
    if normalized not in GATE_REGISTRY:
        raise ValueError(f"Unsupported gate: {name!r}")
    return GATE_REGISTRY[normalized]


def get_gate_qubit_count(name):
    normalized = _normalize_gate_name(name)
    if normalized not in GATE_QUBIT_COUNT:
        raise ValueError(f"Unsupported gate: {name!r}")
    return GATE_QUBIT_COUNT[normalized]


def is_parameterized_gate(name):
    normalized = _normalize_gate_name(name)
    return normalized in PARAMETERIZED_GATES


def is_unitary(gate_or_name):
    """Return whether the supplied gate matrix or gate name is unitary."""
    if isinstance(gate_or_name, str):
        try:
            gate = get_gate(gate_or_name)
        except ValueError:
            return False
    elif isinstance(gate_or_name, np.ndarray):
        gate = gate_or_name
    else:
        return False

    if gate is None or not hasattr(gate, "shape"):
        return False

    if len(gate.shape) != 2 or gate.shape[0] != gate.shape[1]:
        return False

    identity = np.eye(gate.shape[0], dtype=complex)
    return bool(np.allclose(gate.conj().T @ gate, identity, atol=1e-12))


__all__ = [
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "SX",
    "SXdg",
    "RX",
    "RY",
    "RZ",
    "P",
    "U1",
    "U2",
    "U3",
    "CX",
    "CZ",
    "SWAP",
    "CH",
    "CP",
    "CRX",
    "CRY",
    "CRZ",
    "CCX",
    "CSWAP",
    "GATE_REGISTRY",
    "PARAMETERIZED_GATES",
    "GATE_QUBIT_COUNT",
    "get_gate",
    "get_gate_qubit_count",
    "is_parameterized_gate",
    "is_unitary",
]
