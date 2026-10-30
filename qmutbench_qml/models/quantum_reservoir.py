"""Fixed-quantum-feature reservoir classifier with a classical linear readout."""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression


class QuantumReservoirClassifier:
    """Random fixed circuit features; only the class-balanced readout is trained."""

    def __init__(self, n_qubits=4, seed=20261026, c=1.0):
        self.n_qubits = n_qubits
        self.seed = seed
        self.estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=500,
                                           solver="lbfgs", random_state=seed)
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None, seed=seed)
        self.reservoir_weights = np.random.default_rng(seed).uniform(-np.pi, np.pi, (2, n_qubits, 3))

        @qml.qnode(self.device, interface="numpy")
        def circuit(features, weights):
            qml.AngleEmbedding(np.pi * features, wires=range(n_qubits), rotation="Y")
            for layer in range(2):
                for wire in range(n_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

        self.circuit = circuit

    def _transform(self, x):
        started = time.perf_counter()
        with qml.Tracker(self.device) as tracker:
            result = np.asarray([self.circuit(row, self.reservoir_weights) for row in x], dtype=float)
        self.last_transform_seconds = time.perf_counter() - started
        self.last_transform_executions = int(tracker.totals.get("executions", 0))
        return result

    def fit(self, x, y):
        started = time.perf_counter()
        features = self._transform(x)
        self.estimator.fit(features, y)
        self.training_seconds = time.perf_counter() - started
        self.training_executions = self.last_transform_executions
        return self

    def predict_proba(self, x):
        features = self._transform(x)
        self.inference_seconds = self.last_transform_seconds
        self.inference_executions = self.last_transform_executions
        return self.estimator.predict_proba(features)

    def predict(self, x):
        return self.estimator.predict(self._transform(x))
