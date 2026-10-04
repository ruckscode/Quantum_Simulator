import pytest

from quantum_simulator import QuantumCircuit


def test_circuit_methods_append_execution_engine_instructions():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.x(1)
    circuit.y(0)
    circuit.z(1)
    circuit.s(2)
    circuit.sdg(0)
    circuit.t(1)
    circuit.tdg(2)
    circuit.sx(0)
    circuit.sxdg(1)
    circuit.rx(0, 0.1)
    circuit.ry(1, 0.2)
    circuit.rz(2, 0.3)
    circuit.cx(0, 1)
    circuit.cz(1, 2)
    circuit.swap(0, 2)
    circuit.ccx(0, 1, 2)
    circuit.reset(1)
    circuit.barrier()
    circuit.measure(0)
    circuit.measure(2, 1)
    circuit.measure_all()

    assert circuit.n_qubits == 3
    assert circuit.instructions == [
        ("h", 0), ("x", 1), ("y", 0), ("z", 1), ("s", 2), ("sdg", 0),
        ("t", 1), ("tdg", 2), ("sx", 0), ("sxdg", 1), ("rx", 0, 0.1),
        ("ry", 1, 0.2), ("rz", 2, 0.3), ("cx", 0, 1), ("cz", 1, 2),
        ("swap", 0, 2), ("ccx", 0, 1, 2), ("reset", 1), ("barrier",),
        ("measure", 0), ("measure", 2, [1]), ("measure", [0, 1, 2]),
    ]


@pytest.mark.parametrize("n_qubits", [0, -1, True, 1.5, "2"])
def test_constructor_rejects_non_positive_integer_qubit_counts(n_qubits):
    with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
        QuantumCircuit(n_qubits)


@pytest.mark.parametrize("qubit", [-1, 2, 1.5, "0", True])
def test_single_qubit_methods_reject_invalid_qubits(qubit):
    with pytest.raises(ValueError, match="Qubit index"):
        QuantumCircuit(2).h(qubit)


@pytest.mark.parametrize(
    "operation, qubits",
    [
        ("cx", (0, 0)),
        ("cz", (1, 1)),
        ("swap", (0, 0)),
        ("ccx", (0, 1, 0)),
        ("ccx", (1, 1, 0)),
    ],
)
def test_multi_qubit_operations_reject_reused_qubits(operation, qubits):
    circuit = QuantumCircuit(3)
    with pytest.raises(ValueError, match="distinct qubit indices"):
        getattr(circuit, operation)(*qubits)


def test_multi_qubit_operations_reject_out_of_range_qubits():
    with pytest.raises(ValueError, match="out of range"):
        QuantumCircuit(2).cx(0, 2)


def test_measure_rejects_invalid_qubit_and_classical_bit():
    circuit = QuantumCircuit(2)
    with pytest.raises(ValueError, match="Qubit index"):
        circuit.measure(2)
    with pytest.raises(ValueError, match="Classical bit index"):
        circuit.measure(0, -1)
