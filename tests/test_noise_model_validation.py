"""Focused checks for the hardware effects implemented by the simulator."""

import numpy as np

from quantum_simulator import QuantumSimulator
from quantum_simulator.hardware_model import HardwareModel


def _idle_model(*, t1=1e15, t2=1e15, duration=1.0):
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=t1, t2=t2)
    model.add_gate_calibration("i", qubits=(0,), duration=duration)
    return model


def test_zero_noise_hardware_matches_ideal_circuit():
    circuit = [("h", 0), ("x", 0)]
    ideal = QuantumSimulator(1, seed=13)
    default_hardware = QuantumSimulator(1, seed=13)
    zero_error_hardware = HardwareModel(1)
    zero_error_hardware.add_gate_calibration("h", qubits=(0,), error=0.0)
    zero_error_hardware.add_gate_calibration("x", qubits=(0,), error=0.0)

    ideal.run_circuit(circuit)
    default_hardware.run_circuit(circuit, hardware_model=HardwareModel(1))
    calibrated = QuantumSimulator(1, seed=13)
    calibrated.run_circuit(circuit, hardware_model=zero_error_hardware)

    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(ideal.state, expected, atol=1e-12)
    np.testing.assert_allclose(default_hardware.state, ideal.state, atol=1e-12)
    np.testing.assert_allclose(calibrated.state, ideal.state, atol=1e-12)


def test_readout_error_zero_and_one_on_deterministic_one_state():
    circuit = [("x", 0), ("measure", 0)]
    clean = QuantumSimulator(1, seed=21)
    inverted = QuantumSimulator(1, seed=21)
    clean_model = HardwareModel(1)
    inverted_model = HardwareModel(1)
    inverted_model.set_qubit_properties(0, readout_error=1.0)

    assert clean.run_circuit(circuit, hardware_model=clean_model) == ["1"]
    assert inverted.run_circuit(circuit, hardware_model=inverted_model) == ["0"]


def test_calibrated_gate_error_one_skips_deterministic_x_gate():
    circuit = [("x", 0)]
    ideal = QuantumSimulator(1, seed=31)
    failed = QuantumSimulator(1, seed=31)
    zero_error = HardwareModel(1)
    full_error = HardwareModel(1)
    zero_error.add_gate_calibration("x", qubits=(0,), error=0.0)
    full_error.add_gate_calibration("x", qubits=(0,), error=1.0)

    ideal.run_circuit(circuit)
    ideal_with_calibration = QuantumSimulator(1, seed=31)
    ideal_with_calibration.run_circuit(circuit, hardware_model=zero_error)
    failed.run_circuit(circuit, hardware_model=full_error)

    np.testing.assert_allclose(ideal_with_calibration.get_probabilities(), [0, 1], atol=1e-12)
    np.testing.assert_allclose(failed.get_probabilities(), [1, 0], atol=1e-12)


def test_t1_relaxation_moves_excited_state_toward_zero():
    circuit = [("x", 0), ("i", 0)]
    long_t1 = QuantumSimulator(1, seed=41)
    short_t1 = QuantumSimulator(1, seed=41)

    long_t1.run_circuit(circuit, hardware_model=_idle_model(t1=1e15))
    short_t1.run_circuit(circuit, hardware_model=_idle_model(t1=0.01))

    assert long_t1.get_probabilities()[1] > 1 - 1e-12
    # The implemented amplitude-damping trajectory has jump probability
    # gamma = 1 - exp(-duration/T1), essentially one for these parameters.
    expected_excited_population = np.exp(-1.0 / 0.01)
    assert expected_excited_population < 4e-44
    assert short_t1.get_probabilities()[0] > 1 - 1e-12


def test_t2_dephasing_reduces_interference_by_analytic_coherence_factor():
    duration = 1.0
    t2 = 0.25
    high_t2_model = _idle_model(t1=1e15, t2=1e15, duration=duration)
    low_t2_model = _idle_model(t1=1e15, t2=t2, duration=duration)
    shots = 1200
    low_t2_zeros = 0

    high_t2_check = QuantumSimulator(1, seed=53)
    high_t2_check.run_circuit(
        [("h", 0), ("i", 0), ("h", 0)], hardware_model=high_t2_model
    )
    np.testing.assert_allclose(high_t2_check.get_probabilities(), [1, 0], atol=1e-12)

    for seed in range(shots):
        sim = QuantumSimulator(1, seed=seed)
        outcome = sim.run_circuit(
            [("h", 0), ("i", 0), ("h", 0), ("measure", 0)],
            hardware_model=low_t2_model,
        )[0]
        low_t2_zeros += outcome == "0"

    # For T1 >> T2, this implementation applies a phase-flip channel with
    # coherence factor exp(-duration/T2). After H-I-H, P(0) is (1+factor)/2.
    expected_zero_probability = (1 + np.exp(-duration / t2)) / 2
    observed_zero_probability = low_t2_zeros / shots
    assert abs(observed_zero_probability - expected_zero_probability) < 0.04
    assert observed_zero_probability < 0.65
