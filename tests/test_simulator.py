import numpy as np
import pytest

from quantum_simulator.quantum_simulator import QuantumSimulator
from quantum_simulator.hardware_model import HardwareModel


def test_initial_state():
    sim = QuantumSimulator(2)
    np.testing.assert_allclose(sim.state, np.array([1, 0, 0, 0], dtype=complex))
    assert sim.dim == 4


@pytest.mark.parametrize("bad_qubit", [0.5, "0", True, np.float64(1.0)])
def test_qubit_indices_reject_lossy_or_boolean_coercion(bad_qubit):
    sim = QuantumSimulator(2)
    before = sim.state.copy()
    with pytest.raises(TypeError):
        sim.apply_gate("x", bad_qubit)
    np.testing.assert_array_equal(sim.state, before)


@pytest.mark.parametrize("instruction", [
    ("cx", 0, 0),
    ("cz", 1, 1),
    ("rxx", 0, 0, 0.7),
])
def test_duplicate_multi_qubit_operands_are_rejected_without_state_change(instruction):
    sim = QuantumSimulator(2)
    sim.apply_gate("h", 0)
    before = sim.state.copy()
    with pytest.raises(ValueError, match="distinct"):
        sim.run_circuit([instruction])
    np.testing.assert_array_equal(sim.state, before)


def test_boolean_qubit_count_is_rejected_and_numpy_integer_is_supported():
    with pytest.raises(ValueError, match="positive integer"):
        QuantumSimulator(True)
    sim = QuantumSimulator(np.int64(2))
    sim.apply_gate("x", np.int64(1))
    np.testing.assert_array_equal(sim.state, [0, 1, 0, 0])


def test_working_array_estimate_and_opt_in_budget_guard():
    estimate = QuantumSimulator.estimate_working_array_bytes(4)
    assert estimate == 3 * 16 * 2**4
    sim = QuantumSimulator(4, memory_budget_bytes=np.int64(estimate))
    assert sim.memory_budget_bytes == estimate
    with pytest.raises(MemoryError, match="configured budget"):
        QuantumSimulator(4, memory_budget_bytes=estimate - 1)


def test_working_array_estimate_rejects_unaddressable_statevector_size():
    with pytest.raises(MemoryError, match="addressable array size"):
        QuantumSimulator.estimate_working_array_bytes(59)
    with pytest.raises(MemoryError, match="addressable array size"):
        QuantumSimulator.estimate_working_array_bytes(10**12)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_memory_budget_must_be_a_positive_integer_or_none(budget):
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        QuantumSimulator(1, memory_budget_bytes=budget)


def test_run_shots_reuses_configured_memory_budget_for_each_trajectory():
    budget = QuantumSimulator.estimate_working_array_bytes(1)
    sim = QuantumSimulator(1, memory_budget_bytes=budget)
    assert sum(sim.run_shots([("h", 0)], shots=32, seed=11).values()) == 32


@pytest.mark.parametrize("angle", [np.nan, np.inf, -np.inf])
def test_nonfinite_gate_parameters_are_rejected_without_state_change(angle):
    sim = QuantumSimulator(1)
    before = sim.state.copy()
    with pytest.raises(ValueError, match="finite"):
        sim.apply_gate("rx", 0, angle)
    np.testing.assert_array_equal(sim.state, before)


@pytest.mark.parametrize("matrix", [
    np.array([[1, 0], [0, 2]], dtype=complex),
    np.array([[1, 0], [0, np.nan]], dtype=complex),
    np.array([1, 0], dtype=complex),
])
def test_invalid_custom_gate_matrices_are_rejected_without_state_change(matrix):
    sim = QuantumSimulator(1)
    before = sim.state.copy()
    with pytest.raises(ValueError):
        sim.apply_gate(matrix, 0)
    np.testing.assert_array_equal(sim.state, before)


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


@pytest.mark.parametrize("bad_shots", [0, -1, 1.5, True])
def test_run_shots_rejects_nonpositive_or_noninteger_counts(bad_shots):
    with pytest.raises(ValueError, match="positive integer"):
        QuantumSimulator(1).run_shots([], shots=bad_shots)


def test_run_shots_replays_generator_circuit_for_every_shot():
    circuit = [("h", 0), ("measure", 0)]
    expected = QuantumSimulator(1).run_shots(circuit, shots=256, seed=991)
    actual = QuantumSimulator(1).run_shots((instruction for instruction in circuit), shots=256, seed=991)
    assert actual == expected
    assert set(actual) == {"0", "1"}


def test_run_circuit_traced_records_each_sequential_actual_state():
    simulator = QuantumSimulator(1, seed=1)
    instructions = [("x", 0), ("h", 0)]

    clean_states, actual_states, labels, trace = simulator.run_circuit_traced(
        instructions, instructions
    )

    assert len(trace) == len(instructions)
    assert labels == ["x", "h"]
    assert len(actual_states) == len(instructions) + 1
    np.testing.assert_allclose(actual_states[0], [1, 0])
    np.testing.assert_allclose(actual_states[1], [0, 1])
    np.testing.assert_allclose(actual_states[2], [1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert not np.allclose(actual_states[0], actual_states[1])
    assert not np.allclose(actual_states[1], actual_states[2])


@pytest.mark.parametrize("expected,actual,expected_pairs", [
    ([('x', 0), ('h', 0)], [('x', 0), ('h', 0)], [(('x', 0), ('x', 0)), (('h', 0), ('h', 0))]),
    ([('x', 0), ('h', 0)], [('x', 0), ('z', 0), ('h', 0)], [(('x', 0), ('x', 0)), (None, ('z', 0)), (('h', 0), ('h', 0))]),
    ([('x', 0), ('z', 0), ('h', 0)], [('x', 0), ('h', 0)], [(('x', 0), ('x', 0)), (('z', 0), None), (('h', 0), ('h', 0))]),
])
def test_traced_alignment_records_matches_insertions_and_deletions(expected, actual, expected_pairs):
    _, _, _, trace = QuantumSimulator(1).run_circuit_traced(expected, actual)
    assert [(row['expected'], row['actual']) for row in trace] == expected_pairs
    assert all({'step', 'operation', 'operands', 'expected', 'actual', 'difference'} <= row.keys() for row in trace)


def test_traced_alignment_keeps_later_operations_aligned_after_replacement():
    _, _, _, trace = QuantumSimulator(1).run_circuit_traced(
        [('x', 0), ('h', 0), ('z', 0)], [('x', 0), ('y', 0), ('z', 0)]
    )
    assert [(row['expected'], row['actual']) for row in trace] == [
        (('x', 0), ('x', 0)), (('h', 0), ('y', 0)), (('z', 0), ('z', 0))
    ]


def test_traced_differences_are_probability_distances_and_snapshots_progress():
    clean, actual, _, trace = QuantumSimulator(1).run_circuit_traced(
        [('x', 0), ('h', 0)], [('x', 0), ('z', 0)]
    )
    assert not np.allclose(clean[0], clean[1])
    assert not np.allclose(actual[1], actual[2])
    assert trace[-1]['difference'] == pytest.approx(
        np.abs(np.abs(clean[-1]) ** 2 - np.abs(actual[-1]) ** 2).sum() / 2
    )


def test_run_shots_hadamard_samples_both_outcomes():
    counts = QuantumSimulator(1).run_shots([("h", 0)], shots=1000, seed=7)
    assert set(counts) == {"0", "1"}


def test_run_shots_x_returns_only_one():
    counts = QuantumSimulator(1).run_shots([("x", 0)], shots=1000, seed=7)
    assert counts == {"1": 1000}


def test_run_shots_bell_state_samples_only_correlated_outcomes():
    counts = QuantumSimulator(2).run_shots(
        [("h", 0), ("cx", 0, 1)], shots=1000, seed=7
    )
    assert set(counts) == {"00", "11"}


def test_run_shots_same_seed_produces_identical_counts():
    circuit = [("h", 0)]
    first = QuantumSimulator(1).run_shots(circuit, shots=1000, seed=17)
    second = QuantumSimulator(1).run_shots(circuit, shots=1000, seed=17)
    assert first == second


def test_run_shots_different_seeds_can_produce_different_counts():
    circuit = [("h", 0)]
    first = QuantumSimulator(1).run_shots(circuit, shots=1000, seed=17)
    second = QuantumSimulator(1).run_shots(circuit, shots=1000, seed=18)
    assert first != second


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


def test_run_shots_applies_hardware_model_to_each_execution():
    circuit = [("x", 0), ("measure", 0)]
    ideal = QuantumSimulator(1).run_shots(circuit, shots=512, seed=23)
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=1.0)
    affected = QuantumSimulator(1).run_shots(circuit, shots=512, seed=23, hardware_model=hardware)

    assert ideal == {"1": 512}
    assert affected == {"0": 512}


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
