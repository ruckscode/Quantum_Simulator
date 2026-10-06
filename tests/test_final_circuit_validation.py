"""Focused end-to-end validation of representative public circuit workflows."""

from math import pi

import pytest

from quantum_simulator import QuantumCircuit


SHOTS = 1000
SEED = 20261005
PROBABILITY_TOLERANCE = 0.06


def _build_circuits():
    cases = []

    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure(0)
    cases.append(("X state", circuit, {"1": 1.0}, True))

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure(0)
    cases.append(("H state", circuit, {"0": 0.5, "1": 0.5}, False))

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    cases.append(("Bell state", circuit, {"00": 0.5, "11": 0.5}, False))

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    cases.append(("3-qubit GHZ", circuit, {"000": 0.5, "111": 0.5}, False))

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.measure_all()
    cases.append(("4-qubit GHZ", circuit, {"0000": 0.5, "1111": 0.5}, False))

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.x(1)
    circuit.measure_all()
    cases.append(("Bell with X on both qubits", circuit, {"00": 0.5, "11": 0.5}, False))

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.swap(0, 1)
    circuit.measure_all()
    cases.append(("SWAP circuit", circuit, {"01": 1.0}, True))

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.cz(0, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.measure_all()
    cases.append(("CZ interference circuit", circuit, {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}, False))

    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.x(1)
    circuit.ccx(0, 1, 2)
    circuit.measure_all()
    cases.append(("Toffoli circuit", circuit, {"111": 1.0}, True))

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(pi, 0)
    circuit.h(0)
    circuit.measure(0)
    cases.append(("Parameterized rotation/interference circuit", circuit, {"1": 1.0}, True))

    return cases


@pytest.mark.parametrize("name,circuit,expected,deterministic", _build_circuits())
def test_final_public_circuit_validation(name, circuit, expected, deterministic):
    result = circuit.run(shots=SHOTS, seed=SEED)
    counts = result.get_counts()
    probabilities = {outcome: count / SHOTS for outcome, count in counts.items()}

    print(f"{name}: PASS")
    print(f"  counts: {counts}")
    print(f"  observed probabilities: {probabilities}")
    print(f"  expected probabilities: {expected}")

    assert sum(counts.values()) == SHOTS
    if deterministic:
        expected_outcome = next(iter(expected))
        assert counts == {expected_outcome: SHOTS}
    else:
        for outcome, probability in expected.items():
            assert abs(probabilities.get(outcome, 0.0) - probability) <= PROBABILITY_TOLERANCE
        assert set(probabilities).issubset(expected)
