"""Small independently checkable quantum-learning foundations."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def state_from_ry(theta):
    """State RY(theta)|0> under the conventional computational basis."""
    return np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=float)


def basis_probabilities(state):
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1 or not np.isclose(np.vdot(state, state).real, 1.0, atol=1e-10):
        raise ValueError("A normalized one-dimensional statevector is required")
    return np.abs(state) ** 2


def ry_gradient_expectation(theta):
    """Exact d/dtheta <Z> for RY(theta)|0>, used as a gradient oracle."""
    return -np.sin(theta)


def validate_foundations():
    """Cross-check analytic one-qubit identities against PennyLane statevector."""
    device = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(device)
    def state_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.state()

    @qml.qnode(device)
    def z_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    bell_device = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(bell_device)
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    bell = np.asarray(bell_circuit())
    expected_bell = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)

    checks = []
    for theta in (0.0, np.pi / 3, np.pi / 2):
        theta_var = pnp.array(theta, requires_grad=True)
        state = np.asarray(state_circuit(theta))
        expected = state_from_ry(theta)
        probs = basis_probabilities(state)
        grad_value = qml.grad(z_circuit)(theta_var)
        if isinstance(grad_value, tuple):
            grad_value = grad_value[0]
        checks.append({
            "theta": float(theta),
            "state_norm": float(np.vdot(state, state).real),
            "state_max_abs_error": float(np.max(np.abs(state - expected))),
            "probabilities": probs.tolist(),
            "expected_probabilities": [float(np.cos(theta / 2) ** 2), float(np.sin(theta / 2) ** 2)],
            "z_expectation": float(z_circuit(theta)),
            "expected_z": float(np.cos(theta)),
            "z_gradient": float(grad_value),
            "expected_z_gradient": float(ry_gradient_expectation(theta)),
        })
    checks.append({"bell_state_norm": float(np.vdot(bell, bell).real),
                   "bell_state_max_abs_error": float(np.max(np.abs(bell - expected_bell))),
                   "bell_probabilities": (np.abs(bell) ** 2).tolist()})
    return checks
