"""Quantum-kernel SVC using PennyLane statevector overlap estimation."""
from __future__ import annotations

import time
import math

import pennylane as qml
from sklearn.svm import SVC


class QuantumSupportVectorClassifier:
    def __init__(self, n_qubits=4, seed=20261026, c=1.0):
        self.n_qubits = n_qubits
        self.seed = seed
        self.estimator = SVC(C=c, kernel="precomputed", class_weight="balanced", random_state=seed)
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None, seed=seed)

        def feature_map(x):
            for wire in range(n_qubits):
                qml.RY(math.pi * x[wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            for wire in range(n_qubits):
                qml.RZ(math.pi * x[wire], wires=wire)

        @qml.qnode(self.device, interface="numpy")
        def overlap(x, z):
            qml.adjoint(feature_map)(x)
            feature_map(z)
            return qml.probs(wires=range(n_qubits))

        self.overlap = overlap

    def kernel_matrix(self, left, right):
        import numpy as np
        result = np.empty((len(left), len(right)), dtype=float)
        with qml.Tracker(self.device) as tracker:
            for i, x in enumerate(left):
                for j, z in enumerate(right):
                    result[i, j] = float(self.overlap(x, z)[0])
        self.last_kernel_executions = int(tracker.totals.get("executions", 0))
        return result

    def fit(self, x, y):
        started = time.perf_counter()
        self.train_kernel = self.kernel_matrix(x, x)
        self.estimator.fit(self.train_kernel, y)
        self.training_seconds = time.perf_counter() - started
        self.training_executions = self.last_kernel_executions
        self.train_size = len(x)
        return self

    def decision_function(self, x):
        started = time.perf_counter()
        kernel = self.kernel_matrix(x, self._train_x)
        self.inference_seconds = time.perf_counter() - started
        self.inference_executions = self.last_kernel_executions
        return self.estimator.decision_function(kernel)

    def predict(self, x):
        return self.estimator.predict(self.kernel_matrix(x, self._train_x))

    def fit_with_features(self, x, y):
        self._train_x = x.copy()
        return self.fit(x, y)
