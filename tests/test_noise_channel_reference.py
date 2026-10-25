"""Analytical ensemble checks for the implemented stochastic noise model."""

import numpy as np

from quantum_simulator.quantum_simulator import QuantumSimulator
from quantum_simulator.hardware_model import HardwareModel


def test_readout_confusion_probability_matches_seeded_binomial_expectation():
    probability_of_flip = 0.2
    sim = QuantumSimulator(1, seed=9182)
    observed_flips = sum(sim.measure(0, readout_error=probability_of_flip) == "1" for _ in range(2000))

    # A four-standard-deviation bound is a conservative finite-sample check,
    # not a claim that finite-shot frequencies equal the exact probability.
    standard_error = np.sqrt(probability_of_flip * (1 - probability_of_flip) / 2000)
    assert abs(observed_flips / 2000 - probability_of_flip) < 4 * standard_error
    np.testing.assert_allclose(sim.get_probabilities(), [1, 0], atol=1e-12)


def test_t1_relaxation_trajectory_ensemble_matches_amplitude_damping_kraus_channel():
    duration = 0.7
    t1 = 1.0
    gamma = 1 - np.exp(-duration / t1)
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=t1, t2=2 * t1)
    model.add_gate_calibration("i", qubits=(0,), duration=duration)

    initial = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(initial, initial.conjugate())
    k0 = np.diag([1, np.sqrt(1 - gamma)]).astype(complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    expected = k0 @ rho @ k0.conjugate().T + k1 @ rho @ k1.conjugate().T

    empirical = np.zeros((2, 2), dtype=complex)
    trajectories = 4096
    for seed in range(trajectories):
        sim = QuantumSimulator(1, seed=seed)
        sim.state = initial.copy()
        sim.run_circuit([("i", 0)], hardware_model=model)
        empirical += np.outer(sim.state, sim.state.conjugate()) / trajectories

    np.testing.assert_allclose(empirical, expected, rtol=0.0, atol=0.04)
