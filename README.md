# Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs

## 1. Project Overview

This project provides a statevector quantum simulator, a circuit-building API, distribution comparison, and an evidence-based diagnosis workflow. It compares expected and observed execution behavior, applies a configured finite-shot tolerance, and combines circuit-level and hardware-calibration evidence to identify a likely program fault or hardware anomaly. When the evidence cannot distinguish plausible causes, the diagnosis can remain ambiguous or report insufficient evidence.

The core package is `quantum_simulator` under `src/quantum_simulator`. IBM Quantum support lives in the separate, optional `ibm_validation.py` integration.

## 2. Problem Statement

Quantum program outcomes can differ from their expected distributions because of program defects, hardware behavior, sampling variation, or incomplete evidence. A distribution difference alone does not identify its cause. This project brings together simulation, statistical comparison, program-operation comparison, and hardware calibration evidence to support testing and fault localization without claiming certainty when the evidence is insufficient.

## 3. Objectives

- Represent and execute supported quantum circuits with the local simulator.
- Compare expected and observed measurement distributions using statistical distance metrics and a finite-shot tolerance.
- Localize differences between expected and observed circuit operations.
- Evaluate hardware calibration evidence relevant to the circuit.
- Preserve evidence and report `NO_ANOMALY`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY`, `AMBIGUOUS`, or `INSUFFICIENT_EVIDENCE` as supported by the available inputs.
- Keep the core simulator independent of IBM credentials, an IBM account, a specific IBM backend, and live IBM connectivity.

## 4. Architecture

```text
Quantum Program
      |
      v
Circuit Representation
      |
      v
Ideal / hardware-aware execution
      |
      v
Statistical comparison
      |
      v
Anomaly detection
      |
      v
Program / hardware localization
      |
      v
Diagnosis
```

`diagnose_circuit()` runs the supplied circuit ideally and, when a `HardwareModel` is supplied, runs it with that model. The resulting distributions and circuit/calibration context are passed into the existing diagnosis components. The returned diagnosis includes evidence and missing-evidence details.

## 5. Core Components

- `QuantumCircuit`: public builder for gates, measurement, and local execution.
- `QuantumSimulator`: statevector execution, supported gate operations, measurement, and shot sampling.
- `HardwareModel`: backend-independent representation of qubit properties, gate calibrations, and connectivity.
- `statistics.py`: distribution and count comparisons, including total variation distance and finite-shot tolerance.
- `fault_localizer.py`: comparison of expected and observed operations and program-fault candidates.
- `hardware_fault_localizer.py`: calibration-backed hardware candidates relevant to the circuit.
- `diagnosis_engine.py`: integrated diagnosis through `diagnose_circuit()` and `diagnose_execution()`.
- `validation_framework.py`, `demo.py`, and benchmark adapters: controlled local validation, demonstration, and dataset reporting.
- `ibm_validation.py`: optional IBM backend discovery, execution, calibration inspection, and calibration conversion.

## 6. Installation

The project requires Python 3.10 or newer. The core runtime dependency is NumPy; pytest is used for tests.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pytest
```

To use the package from another project, install it from the repository root:

```powershell
python -m pip install .
```

The core simulator does not require Qiskit, `qiskit-ibm-runtime`, IBM credentials, network access, or an IBM account.

## 7. Basic QuantumCircuit Usage

```python
from quantum_simulator import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

result = circuit.run(shots=1000, seed=42)
print(result.get_counts())
```

`QuantumCircuit` supports the gates and special operations implemented in the package, including single-qubit gates and rotations, `cx`, `cz`, `swap`, `ccx`, `reset`, barriers, and measurement. See `src/quantum_simulator/quantum_circuit.py` for the current builder methods.

## 8. Running Simulations

`QuantumCircuit.run()` returns a result with a `get_counts()` method. The simulator can also be used directly:

```python
from quantum_simulator import QuantumSimulator

simulator = QuantumSimulator(2, seed=42)
counts = simulator.run_shots(
    [("h", 0), ("cx", 0, 1), ("measure", [0, 1])],
    shots=1000,
)
print(counts)
```

Run the offline demonstration from the repository root with:

```powershell
.\.venv\Scripts\python.exe demo.py
```

The demo uses controlled local scenarios and does not claim its observations came from physical hardware.

## 9. Diagnosis Workflow

The public `diagnose_circuit()` function accepts either a `QuantumCircuit` or a supported instruction sequence. It returns a `DiagnosisResult` with category, anomaly status, statistical result, localization candidates, evidence, and missing evidence.

```python
from quantum_simulator import diagnose_circuit

circuit = [("h", 0), ("cx", 0, 1), ("measure", [0, 1])]
diagnosis = diagnose_circuit(circuit, shots=1000, seed=42, threshold=0.05)

print(diagnosis.category)
print(diagnosis.summary)
print(diagnosis.evidence)
```

When no hardware model is supplied, the ideal distribution is compared against the same ideal execution; hardware calibration evidence is unavailable. Supplying a model enables the model-aware execution and calibration analysis:

```python
from quantum_simulator import diagnose_circuit
from quantum_simulator.hardware_model import HardwareModel

hardware = HardwareModel(2)
hardware.set_qubit_properties(1, readout_error=0.5)
circuit = [("h", 0), ("cx", 0, 1), ("measure", [0, 1])]

diagnosis = diagnose_circuit(
    circuit,
    hardware_model=hardware,
    shots=1000,
    seed=42,
)
print(diagnosis.category)
```

The finite-shot tolerance is applied by the statistical comparison. An anomaly can be detected without enough evidence to attribute it to a specific cause.

## 10. Program Fault Localization

The program localizer compares expected and observed instruction sequences and can report operation-level candidates, such as changed, inserted, or removed operations. The integrated `diagnose_execution()` workflow accepts expected and observed distributions and optional expected and observed circuits:

```python
from quantum_simulator.diagnosis_engine import diagnose_execution

result = diagnose_execution(
    expected_distribution={"0": 0.0, "1": 1.0},
    observed_distribution={"0": 1.0, "1": 0.0},
    expected_circuit=[("x", 0), ("measure", 0)],
    observed_circuit=[("h", 0), ("measure", 0)],
    shots=1000,
)
print(result.category)
print(result.program_evidence_candidates)
```

This lower-level workflow accepts supplied execution distributions. `diagnose_circuit()` itself analyzes the circuit it is given; it does not currently accept a separate reference circuit argument.

## 11. Hardware Fault Localization

Hardware localization evaluates calibration metrics and connectivity in relation to the supplied circuit. The integrated diagnosis can report relevant qubit, edge, or gate candidates when calibration evidence crosses the configured threshold. Calibration evidence alone does not guarantee that a hardware anomaly caused an observed difference; the diagnosis may be ambiguous or report insufficient evidence.

## 12. HardwareModel

`HardwareModel` is the simulator’s generic hardware representation. It stores qubit count, coupling information, backend name and calibration timestamp, qubit T1/T2 and readout error, and available gate/edge properties. It has no IBM dependency. Values left at model defaults are generic model values, not measurements of a real backend.

```python
from quantum_simulator.hardware_model import HardwareModel

hardware = HardwareModel(
    2,
    coupling_map=[(0, 1)],
    backend_name="example-backend",
)
hardware.set_qubit_properties(1, readout_error=0.02, t1=120.0, t2=80.0)
hardware.add_gate_calibration("cx", qubits=(0, 1), error=0.01, duration=0.2)
```

The units supplied to `HardwareModel` are used as values; callers should keep units consistent with the calibration data they provide.

## 13. Optional IBM Quantum Integration

IBM Quantum is an **optional external integration**. The core simulator remains usable without IBM credentials, an IBM account, a particular IBM backend, live backend access, or an internet connection.

When IBM integration is used, the caller explicitly selects a backend and authenticates through external IBM configuration or credentials. The integration retrieves calibration information dynamically from that selected backend, can submit an IBM execution, and can analyze returned results with the project’s statistics and diagnosis components. The calibration capture can also be converted into a `HardwareModel`. Calibration values and available fields can change over time; some backend fields may be unavailable.

The adapter is in `ibm_validation.py`. Its offline validation path uses local fixtures and does not submit an IBM job. Live integration requires compatible Qiskit and Qiskit IBM Runtime installations, valid external access, network connectivity, and an explicitly selected backend. No backend is a permanent or default project configuration.

The repository contains an IBM validation report for an offline fixture. It is not a live hardware result. Any earlier backend experiment, including one performed on `ibm_marrakesh`, is a historical validation snapshot only and is not needed to run the project.

## 14. Bugs4Q / QMutBench Validation

The repository includes separate adapters for Bugs4Q and QMutBench data. They normalize supported dataset inputs, run supported cases through project functionality, and produce benchmark-specific reports. Mutation metadata is not used as a diagnosis label. Benchmark metrics are not claimed without suitable independent, reliable ground truth.

The checked-in QMutBench data contains a small set of QASM source/mutant examples. The Bugs4Q adapter and integration tests are present; external/official dataset material may not be available in every checkout. Synthetic test fixtures and controlled local scenarios are not benchmark ground truth.

Run benchmark reporting from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m benchmark_validation --root .
```

## 15. Validation Results

The latest verified full project suite completed with **280 passed** when pytest used a writable repository-local temporary directory. The normal Windows pytest command encountered **20 setup errors** because its default temporary directory was inaccessible; these were environment-only errors. No genuine test failures were found.

The historical benchmark reports record focused Bugs4Q integration for 8 selected cases (3 diagnosed `NO_ANOMALY`, 5 `PROGRAM_FAULT`) and a QMutBench pilot with 12 paired cases (11 executed and diagnosed, 1 unsupported). These counts are integration results, not accuracy or fault-attribution guarantees. The reports do not claim benchmark accuracy, precision, recall, F1, or mutation score.

## 16. Project Structure

```text
src/quantum_simulator/    packaged simulator, circuit, hardware, statistics, diagnosis
tests/                    simulator, diagnosis, integration, and validation tests
experiments/              controlled local experiment configurations and runner
reports/                  historical demo, experiment, benchmark, and IBM adapter reports
data/qmutbench/           included QMutBench QASM examples
demo.py                   offline diagnostic demonstration
ibm_validation.py          optional IBM integration and offline validation
benchmark_validation.py   combined benchmark report runner
bugs4q_validation.py      Bugs4Q dataset adapter
qmutbench_validation.py  QMutBench dataset adapter
```

Top-level compatibility modules are retained alongside the `src/quantum_simulator` package for repository-local imports and existing tools.

## 17. Current Limitations

- Hardware-aware execution and localization are limited to the behavior implemented by `HardwareModel` and the simulator; this is not a general physical noise simulator.
- IBM integration depends on external IBM access when used. Backend availability and calibration data can change, and some calibration fields may be unavailable.
- A detected distribution difference does not by itself establish a cause. Diagnosis can remain ambiguous or report insufficient evidence when multiple causes are plausible or evidence is incomplete.
- `diagnose_circuit()` does not currently take a separate reference circuit; supplied expected/observed execution data and circuit comparisons are supported by the lower-level diagnosis workflow.
- The benchmark adapters do not claim accuracy, precision, recall, F1, mutation score, or guaranteed fault attribution without independent reliable ground truth.

## 18. Future Work

Potential future work includes adding explicit reference-circuit support to the high-level workflow, extending hardware execution/noise modeling, and evaluating localization against independent benchmark ground truth. These are future possibilities, not current capabilities.

## 19. Testing

Run the complete suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

If the environment’s default pytest temporary directory is inaccessible, configure pytest to use a writable temporary directory. Offline IBM tests do not require credentials or a live IBM job.

## 20. License / Repository Information

No license declaration or repository URL is provided in the project metadata, so this README does not state a license or external repository address.

## Current Project Status

The implementation and validation suite are complete. The repository is in finalization and documentation stage.
