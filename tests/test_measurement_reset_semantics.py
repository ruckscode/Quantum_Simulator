"""Analytical checks for measurement instruments and reset trajectories."""

import numpy as np

from quantum_simulator.quantum_simulator import QuantumSimulator


def test_readout_flip_changes_report_but_collapse_uses_physical_outcome():
    ideal = QuantumSimulator(1, seed=19)
    ideal.apply_gate("h", 0)
    physical_outcome = ideal.measure(0)

    noisy = QuantumSimulator(1, seed=19)
    noisy.apply_gate("h", 0)
    reported_outcome = noisy.measure(0, readout_error=1.0)

    assert reported_outcome == str(1 - int(physical_outcome))
    expected_state = np.array([1, 0], dtype=complex) if physical_outcome == "0" else np.array([0, 1], dtype=complex)
    np.testing.assert_allclose(noisy.state, expected_state, atol=1e-12)


def test_reset_maps_basis_and_superposition_inputs_to_zero():
    for initial_state in (np.array([0, 1], dtype=complex), np.array([1, 1], dtype=complex) / np.sqrt(2)):
        sim = QuantumSimulator(1, seed=31)
        sim.state = initial_state.copy()
        sim.reset_qubit(0)
        np.testing.assert_allclose(sim.state, [1, 0], atol=1e-12)


def test_entangled_reset_trajectories_match_analytical_mixed_channel_output():
    density = np.zeros((4, 4), dtype=complex)
    for seed in range(1024):
        sim = QuantumSimulator(2, seed=seed)
        sim.run_circuit([("h", 0), ("cx", 0, 1)])
        sim.reset_qubit(0)
        np.testing.assert_allclose(sim.get_probabilities()[2:], [0, 0], atol=1e-12)
        density += np.outer(sim.state, sim.state.conjugate()) / 1024

    # Resetting either half of |Phi+> yields |0><0| tensor I/2.
    expected = np.diag([0.5, 0.5, 0.0, 0.0]).astype(complex)
    np.testing.assert_allclose(density, expected, atol=0.06)
