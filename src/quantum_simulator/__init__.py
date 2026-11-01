"""Reusable core for quantum circuit simulation and diagnosis."""

from .quantum_simulator import QuantumSimulator
from .quantum_circuit import QuantumCircuit, SimulationResult
from .diagnosis_engine import (
    DiagnosisCategory,
    DiagnosisResult,
    diagnose_circuit,
    diagnose_execution,
)
from .hardware_model import HardwareModel
from .ml_diagnosis import ClassifierPrediction, ObservableClassifier
from .observable_model_adapter import PortableLogisticRegressionAdapter
from .explainability import explain_diagnosis

__all__ = [
    "DiagnosisCategory",
    "DiagnosisResult",
    "HardwareModel",
    "ClassifierPrediction",
    "ObservableClassifier",
    "PortableLogisticRegressionAdapter",
    "QuantumCircuit",
    "QuantumSimulator",
    "SimulationResult",
    "diagnose_circuit",
    "diagnose_execution",
    "explain_diagnosis",
]
