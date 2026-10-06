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
