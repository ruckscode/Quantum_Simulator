import os
from pathlib import Path
import subprocess
import sys


def test_repository_root_import_uses_src_package():
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import quantum_simulator; "
        "from quantum_simulator import QuantumCircuit, QuantumSimulator; "
        "from pathlib import Path; "
        "expected = Path('src/quantum_simulator/__init__.py').resolve(); "
        "assert Path(quantum_simulator.__file__).resolve() == expected; "
        "assert QuantumCircuit.__module__ == 'quantum_simulator.quantum_circuit'; "
        "assert QuantumSimulator.__module__ == 'quantum_simulator.quantum_simulator'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=os.environ.copy(),
        check=True,
    )


def test_legacy_core_imports_resolve_to_packaged_modules():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import diagnosis_engine
import fault_localizer
import gates
import hardware_fault_localizer
import hardware_model
import quantum_simulator
from quantum_simulator import diagnosis_engine as packaged_diagnosis
from quantum_simulator import fault_localizer as packaged_localizer
from quantum_simulator import gates as packaged_gates
from quantum_simulator import hardware_fault_localizer as packaged_hardware_localizer
from quantum_simulator import hardware_model as packaged_hardware_model
assert diagnosis_engine is packaged_diagnosis
assert fault_localizer is packaged_localizer
assert gates is packaged_gates
assert hardware_fault_localizer is packaged_hardware_localizer
assert hardware_model is packaged_hardware_model
assert diagnosis_engine.DiagnosisCategory is packaged_diagnosis.DiagnosisCategory
assert hardware_model.HardwareModel is packaged_hardware_model.HardwareModel
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=os.environ.copy(),
        check=True,
    )


def test_statistics_compatibility_preserves_stdlib_and_project_helpers():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import statistics
from statistics import compare_counts, compare_distributions
from quantum_simulator.statistics import (
    compare_counts as packaged_compare_counts,
    compare_distributions as packaged_compare_distributions,
)

assert statistics.mean([1, 2, 6]) == 3
assert compare_counts is packaged_compare_counts
assert compare_distributions is packaged_compare_distributions
assert compare_counts({"0": 2}, {"0": 2}).total_variation_distance == 0
assert compare_distributions({"0": 1.0}, {"0": 1.0}).anomaly_detected is False
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=os.environ.copy(),
        check=True,
    )


def test_core_import_isolated_from_optional_ibm_runtime():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name == "qiskit_ibm_runtime" or name.startswith("qiskit_ibm_runtime."):
        raise AssertionError("core attempted an IBM Runtime import")
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from quantum_simulator import QuantumCircuit, QuantumSimulator, diagnose_circuit
assert QuantumCircuit and QuantumSimulator and diagnose_circuit
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=os.environ.copy(),
        check=True,
    )
