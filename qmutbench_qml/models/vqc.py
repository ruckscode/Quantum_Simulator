"""Trainable PennyLane variational quantum classifier on a statevector simulator."""
from __future__ import annotations

import time

import pennylane as qml
from pennylane import numpy as pnp


class VariationalQuantumClassifier:
    def __init__(self, n_qubits=4, depth=2, steps=20, learning_rate=0.08, seed=20261026,
                 data_reuploading=False):
        self.n_qubits = n_qubits
        self.depth = depth
        self.steps = steps
        self.learning_rate = learning_rate
        self.seed = seed
        self.data_reuploading = data_reuploading
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None, seed=seed)

        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(features, weights):
            for layer in range(depth):
                if layer == 0 or data_reuploading:
                    qml.AngleEmbedding(pnp.pi * features, wires=range(n_qubits), rotation="Y")
                for wire in range(n_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def fit(self, x, y):
        import numpy as np
        x = pnp.array(x, requires_grad=False)
        y = pnp.array(y, requires_grad=False)
        class_counts = pnp.array([pnp.sum(y == label) for label in (0, 1)], requires_grad=False)
        class_weights = len(y) / (2.0 * class_counts)
        sample_weights = pnp.where(y == 1, class_weights[1], class_weights[0])
        rng = np.random.default_rng(self.seed)
        weights = pnp.array(rng.normal(0, 0.15, (self.depth, self.n_qubits, 3)), requires_grad=True)
        initial_weights = np.asarray(weights).copy()
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        def objective(params):
            z_values = pnp.stack([self.circuit(row, params) for row in x])
            p_detected = pnp.clip((1.0 - z_values) / 2.0, 1e-7, 1.0 - 1e-7)
            row_loss = -(y * pnp.log(p_detected) + (1.0 - y) * pnp.log(1.0 - p_detected))
            return pnp.mean(sample_weights * row_loss)

        self.loss_history = []
        started = time.perf_counter()
        with qml.Tracker(self.device) as tracker:
            for _ in range(self.steps):
                weights, cost = optimizer.step_and_cost(objective, weights)
                self.loss_history.append(float(cost))
        self.training_seconds = time.perf_counter() - started
        self.training_executions = int(tracker.totals.get("executions", 0))
        self.initial_weights = initial_weights
        self.weights = weights
        return self

    def predict_proba(self, x):
        import numpy as np
        started = time.perf_counter()
        with qml.Tracker(self.device) as tracker:
            z = np.asarray([float(self.circuit(row, self.weights)) for row in x])
        self.inference_seconds = time.perf_counter() - started
        self.inference_executions = int(tracker.totals.get("executions", 0))
        p1 = np.clip((1.0 - z) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class DataReuploadingClassifier(VariationalQuantumClassifier):
    """VQC architecture that re-encodes inputs before every variational layer."""

    def __init__(self, n_qubits=4, depth=2, steps=20, learning_rate=0.08, seed=20261026):
        super().__init__(n_qubits=n_qubits, depth=depth, steps=steps,
                         learning_rate=learning_rate, seed=seed, data_reuploading=True)
