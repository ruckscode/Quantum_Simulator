import numpy as np
import pytest

from quantum_simulator.quantum_simulator import QuantumSimulator
from quantum_simulator.hardware_model import HardwareModel


def test_initial_state():
    sim = QuantumSimulator(2)
    np.testing.assert_allclose(sim.state, np.array([1, 0, 0, 0], dtype=complex))
    assert sim.dim == 4


def test_single_qubit_hadamard_and_x():
    sim = QuantumSimulator(1)
    sim.apply_gate("h", 0)
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    np.testing.assert_allclose(sim.state, expected, atol=1e-12)

    sim.apply_gate("x", 0)
    np.testing.assert_allclose(sim.state, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), atol=1e-12)


@pytest.mark.parametrize("gate_name", ["i", "id", "identity"])
def test_run_circuit_identity_gate_aliases_are_noops(gate_name):
    sim = QuantumSimulator(1)
    sim.run_circuit([("h", 0)])
    before = sim.state.copy()

    sim.run_circuit([(gate_name, 0)])

    np.testing.assert_allclose(sim.state, before, atol=1e-12)


def test_cnot_bell_state():
    sim = QuantumSimulator(2)
    sim.run_circuit([("h", 0), ("cnot", 0, 1)])
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(sim.state, expected, atol=1e-12)


def test_measurement_and_reset():
    sim = QuantumSimulator(2)
    sim.run_circuit([("h", 0), ("measure", 0)])
    assert len(sim.measurement_results) == 1
    assert sim.measurement_results[0] in {"0", "1"}

    sim.reset_qubit(1)
    probs = sim.get_probabilities()
    assert np.isclose(probs.sum(), 1.0)


def test_run_shots_counts():
    sim = QuantumSimulator(1)
    counts = sim.run_shots([("h", 0)], shots=200)
    assert sum(counts.values()) == 200
    assert set(counts) <= {"0", "1"}


def test_barrier_is_noop():
    sim = QuantumSimulator(2)
    before = sim.state.copy()
    sim.run_circuit([("barrier", 0, 1)])
    np.testing.assert_allclose(sim.state, before, atol=1e-12)


def test_ideal_execution_is_unchanged_without_hardware_model():
    sim = QuantumSimulator(1, seed=7)
    sim.run_circuit([("x", 0)])
    np.testing.assert_allclose(sim.state, np.array([0, 1], dtype=complex), atol=1e-12)


def test_healthy_hardware_model_runs_circuit():
    sim = QuantumSimulator(2, seed=7)
    model = HardwareModel(2)
    sim.run_circuit([("h", 0), ("cnot", 0, 1)], hardware_model=model)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(sim.state, expected, atol=1e-12)
    np.testing.assert_allclose(sim.get_probabilities(), [0.5, 0.0, 0.0, 0.5], atol=1e-12)


def test_configured_gate_error_changes_only_calibrated_operation_qubits():
    sim = QuantumSimulator(2, seed=7)
    model = HardwareModel(2)
    model.add_gate_calibration("x", qubits=(0,), error=1.0)

    sim.run_circuit([("x", 0), ("x", 1)], hardware_model=model)

    # Gate-operation skip model: calibrated X(q0) is skipped with probability 1;
    # the uncalibrated X(q1) executes. No readout bit flip is involved.
    np.testing.assert_allclose(sim.get_probabilities(), [0, 1, 0, 0], atol=1e-12)


def test_readout_error_is_applied_only_when_configured():
    ideal = QuantumSimulator(2, seed=7)
    ideal_model = HardwareModel(2)
    assert ideal.run_circuit([("measure", 0)])[0] == "0"

    noisy = QuantumSimulator(2, seed=7)
    model = HardwareModel(2)
    model.set_qubit_properties(0, readout_error=1.0)
    # Readout bit-flip model: the state remains |00>, while measured q0 flips.
    assert noisy.run_circuit([("measure", 0)], hardware_model=model)[0] == "1"
    np.testing.assert_allclose(noisy.get_probabilities(), [1, 0, 0, 0], atol=1e-12)

    unrelated = QuantumSimulator(2, seed=7)
    assert unrelated.run_circuit([("measure", 1)], hardware_model=model)[0] == "0"


def _idle_calibration(model, duration=1.0):
    model.add_gate_calibration("i", qubits=(0,), duration=duration)


def test_very_large_t1_causes_negligible_relaxation():
    sim = QuantumSimulator(1, seed=7)
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=1e15, t2=1e15)
    _idle_calibration(model)
    sim.run_circuit([("x", 0), ("i", 0)], hardware_model=model)
    np.testing.assert_allclose(sim.get_probabilities(), [0, 1], atol=1e-12)


def test_very_large_t2_causes_negligible_dephasing():
    sim = QuantumSimulator(1, seed=7)
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=1e15, t2=1e15)
    _idle_calibration(model)
    sim.run_circuit([("h", 0), ("i", 0)], hardware_model=model)
    np.testing.assert_allclose(np.abs(sim.state[0] * sim.state[1].conjugate()), 0.5, atol=1e-12)


def test_finite_t1_relaxes_excited_state_toward_zero():
    sim = QuantumSimulator(1, seed=7)
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=0.01, t2=1e15)
    _idle_calibration(model)
    sim.run_circuit([("x", 0), ("i", 0)], hardware_model=model)
    np.testing.assert_allclose(sim.get_probabilities(), [1, 0], atol=1e-12)


def test_finite_t2_reduces_ensemble_coherence_without_population_relaxation():
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=1e15, t2=0.2)
    _idle_calibration(model, duration=1.0)
    density = np.zeros((2, 2), dtype=complex)
    for seed in range(400):
        sim = QuantumSimulator(1, seed=seed)
        sim.run_circuit([("h", 0), ("i", 0)], hardware_model=model)
        density += np.outer(sim.state, sim.state.conjugate()) / 400
    np.testing.assert_allclose(np.diag(density).real, [0.5, 0.5], atol=0.03)
    assert abs(density[0, 1]) < 0.5 * np.exp(-1 / 0.2) + 0.08
    assert abs(density[0, 1]) < 0.5


@pytest.mark.parametrize("bad_time", [None, float("nan"), float("inf")])
def test_invalid_coherence_times_do_not_change_state(bad_time):
    baseline = QuantumSimulator(1, seed=7)
    baseline.run_circuit([("x", 0), ("i", 0)])
    sim = QuantumSimulator(1, seed=7)
    model = HardwareModel(1)
    model.set_qubit_properties(0, t1=0.01, t2=0.01)
    model.get_qubit_properties(0).t1 = bad_time
    model.get_qubit_properties(0).t2 = bad_time
    _idle_calibration(model)
    sim.run_circuit([("x", 0), ("i", 0)], hardware_model=model)
    np.testing.assert_allclose(sim.state, baseline.state, atol=1e-12)


def test_healthy_and_faulted_gate_operation_counts_are_reproducible():
    circuit = [("x", 0), ("measure", 0)]
    healthy = HardwareModel(1)
    faulty = HardwareModel(1)
    faulty.add_gate_calibration("x", qubits=(0,), error=1.0)

    def outcomes(model):
        counts = {"0": 0, "1": 0}
        for seed in range(12):
            sim = QuantumSimulator(1, seed=seed)
            result = sim.run_circuit(circuit, hardware_model=model)[0]
            counts[result] += 1
        return counts

    assert outcomes(healthy) == {"0": 0, "1": 12}
    # A probability-one gate-operation skip leaves |0>; readout error is zero.
    assert outcomes(faulty) == {"0": 12, "1": 0}


def test_configured_readout_bit_flip_counts_leave_state_probabilities_unchanged():
    model = HardwareModel(2)
    model.set_qubit_properties(0, readout_error=1.0)
    counts = {"0": 0, "1": 0}
    for seed in range(12):
        sim = QuantumSimulator(2, seed=seed)
        result = sim.run_circuit([("measure", 0)], hardware_model=model)[0]
        counts[result] += 1
        np.testing.assert_allclose(sim.get_probabilities(), [1, 0, 0, 0], atol=1e-12)
    assert counts == {"0": 0, "1": 12}
